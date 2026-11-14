"""Data for the Modal stages: fetch, convert and build mixes in Modal containers, never on the laptop.

    # once: the secrets (values from your own accounts; nothing is stored in the repo)
    modal secret create kaggle KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN      # or KAGGLE_USERNAME=… KAGGLE_KEY=…
    modal secret create huggingface HF_TOKEN=hf_…                       # WildGuardMix is gated: accept AI2's terms on the Hub first

    uv run --extra bench modal run scripts/modal_data.py --fetch-stage-b         # Stage B mix (Kaggle notebook output) → /vol/mix/stage_b
    uv run --extra bench modal run scripts/modal_data.py --convert stage_d       # Stage D sources → /vol/public (one container each)
    uv run --extra bench modal run scripts/modal_data.py --convert toxigen --limit 200   # smoke: 200 rows per split, streamed
    uv run --extra bench modal run scripts/modal_data.py --build stage_d         # configs/mixes/stage_d.yaml → /vol/mix/stage_d
    uv run --extra bench modal run scripts/modal_data.py --upload-stand data/stand/records/stand.jsonl   # Stage C: company records → private volume
    uv run --extra bench modal run scripts/modal_data.py --build stage_c         # after Stage D: stand + D replay + B replay

Everything lands in the private volume ``jevlite-stage-c`` (mounted at ``/vol``), where ``scripts/modal_train.py``
reads mixes from (``/vol/mix/<mix>``). Raw downloads and the Hub cache go to ``jevlite-hf-cache``.

- ``--fetch-stage-b``: the files of ``stage_b/`` in the output of the Kaggle notebook ``mrgrin548/stage-b-mix``
  (the mix ``stage_b-full-s42-r1`` was trained on), through the Kaggle API (secret ``kaggle``).
- ``--convert``: ``scripts/convert_public.py`` per dataset; ``stage_d`` = the 5 training sources + 3 eval-only
  ones (``jevlite.data.public.safety``), all with the ``huggingface`` secret (WildGuardMix is gated).
- ``--build``: ``scripts/build_mix.py`` with the stage's mix and epochs from ``configs/base.yaml``,
  ``$STAGE_B_MIX`` / ``$STAGE_D_MIX`` / ``$STAND_RECORDS`` pointing at the volume, ``--batch-size`` = the
  H100 micro-batch of ``modal_train.py`` (for the loss-weight hint). ``mix_report.md`` is printed and
  saved to ``data/mix/<mix>/mix_report.md`` (a few KB: the only file that comes back).
- ``--export-labels <mix>``: the mix's eval records (``test_in``, ``test_ood``, ``dev``, ``calib``) without
  their states → ``data/mix/<mix>/labels.jsonl``: targets and slicing fields only (ids, sources, criteria),
  enough for ``scripts/eval.py`` on a run's ``preds.jsonl``, a few % of the size, and no texts. For a mix with
  company data (``stage_c``) the stand records' prompts stay out too.
- ``--upload-stand``: the converted stand records (company data) → ``/vol/stand/stand.jsonl``; the volume
  is private, and the file never goes anywhere else.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

HF_CACHE = "/root/.cache/huggingface"
VOL = "/vol"
CONFIGS = "/root/configs"
SCRIPTS = "/root/scripts"
KAGGLE_KERNEL = ("mrgrin548", "stage-b-mix")
STAGE_D_SOURCES = ("aegis2", "wildguardmix", "toxigen", "measuring_hate_speech", "civil_comments",
                   "toxic_chat", "beavertails", "openai_moderation")
GATED = {"wildguardmix"}
MICRO_BATCH_H100 = 16  # modal_train.MICRO_BATCH["H100"]

app = modal.App("jevlite-data")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(".", extras=["bench"], frozen=True)
    .env({"HF_HOME": HF_CACHE, "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
          "HF_DATASETS_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_python_source("jevlite")
    .add_local_dir("configs", CONFIGS)
    .add_local_dir("scripts", SCRIPTS)
)
hf_cache = modal.Volume.from_name("jevlite-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("jevlite-stage-c", create_if_missing=True)
VOLUMES = {HF_CACHE: hf_cache, VOL: data}


def _kaggle_headers() -> dict:
    import base64

    if token := os.environ.get("KAGGLE_API_TOKEN"):
        return {"Authorization": f"Bearer {token}"}
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        raise RuntimeError("secret 'kaggle' needs KAGGLE_API_TOKEN, or KAGGLE_USERNAME and KAGGLE_KEY")
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()}


@app.function(image=image, cpu=2.0, memory=4096, timeout=2 * 3600, volumes=VOLUMES,
              secrets=[modal.Secret.from_name("kaggle")])
def do_fetch_stage_b(prefix: str = "stage_b/") -> list[str]:
    import json
    import shutil
    import urllib.parse
    import urllib.request

    owner, slug = KAGGLE_KERNEL
    files, token = [], ""
    while True:
        query = urllib.parse.urlencode({"userName": owner, "kernelSlug": slug, "pageSize": 100, "pageToken": token})
        req = urllib.request.Request(f"https://www.kaggle.com/api/v1/kernels/output?{query}", headers=_kaggle_headers())
        with urllib.request.urlopen(req, timeout=60) as resp:
            page = json.load(resp)
        files += [f for f in page.get("files", []) if f.get("fileName", "").startswith(prefix)]
        token = page.get("nextPageToken") or ""
        if not token:
            break
    if not files:
        raise RuntimeError(f"no {prefix}* files in the output of kaggle kernel {owner}/{slug}")
    out = Path(VOL) / "mix"
    done = []
    for f in files:
        dest = out / f["fileName"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(f["url"], timeout=600) as resp, tmp.open("wb") as fh:
            shutil.copyfileobj(resp, fh, length=1 << 20)
        tmp.rename(dest)
        done.append(f"{f['fileName']} ({dest.stat().st_size / 1e6:.1f} MB)")
        print(done[-1], flush=True)
    data.commit()
    return done


def _convert(name: str, limit: int) -> str:
    out = f"{VOL}/public" if not limit else f"{VOL}/public_smoke"
    cmd = ["python", f"{SCRIPTS}/convert_public.py", "--datasets", name, "--out", out, "--raw-dir", f"{HF_CACHE}/raw"]
    if limit:
        cmd += ["--limit", str(limit)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    hf_cache.commit()
    data.commit()
    if proc.returncode:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-30:])
        raise RuntimeError(f"{name}: convert_public.py failed ({proc.returncode})\n{tail}")
    return "\n".join(line for line in proc.stdout.splitlines() if line.startswith("["))  # the converter's own summary


@app.function(image=image, cpu=4.0, memory=32768, timeout=3 * 3600, volumes=VOLUMES,
              secrets=[modal.Secret.from_name("huggingface")])  # authenticated Hub requests: higher rate limits
def do_convert(name: str, limit: int = 0) -> str:
    return _convert(name, limit)


@app.function(image=image, cpu=4.0, memory=32768, timeout=3 * 3600, volumes=VOLUMES,
              secrets=[modal.Secret.from_name("huggingface")])
def do_convert_gated(name: str, limit: int = 0) -> str:
    return _convert(name, limit)


@app.function(image=image, cpu=8.0, memory=65536, timeout=4 * 3600, volumes=VOLUMES)
def do_build(stage: str) -> str:
    import yaml

    data.reload()
    stage_cfg = yaml.safe_load(Path(f"{CONFIGS}/base.yaml").read_text())["train"][stage]
    mix_cfg = Path(CONFIGS) / stage_cfg["mix"]
    out = Path(VOL) / "mix" / mix_cfg.stem
    env = {**os.environ, "STAGE_B_MIX": f"{VOL}/mix/stage_b", "STAGE_D_MIX": f"{VOL}/mix/stage_d",
           "STAND_RECORDS": f"{VOL}/stand/stand.jsonl"}
    cmd = ["python", f"{SCRIPTS}/build_mix.py", "--config", str(mix_cfg), "--out", str(out),
           "--epochs", str(stage_cfg.get("epochs", 1)), "--batch-size", str(MICRO_BATCH_H100)]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env, text=True)
    hf_cache.commit()
    data.commit()
    if proc.returncode:
        raise RuntimeError(f"build_mix.py failed ({proc.returncode}); its output is above")
    return (out / "mix_report.md").read_text()


LABEL_DROP = ("state", "user_question")


@app.function(image=image, cpu=2.0, memory=8192, timeout=3600, volumes=VOLUMES)
def do_export_labels(mix: str) -> list[str]:
    import json

    data.reload()
    out = []
    for split in ("test_in", "test_ood", "dev", "calib"):
        path = Path(VOL) / "mix" / mix / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                for k in LABEL_DROP:
                    row.pop(k, None)
                if row.get("source") == "stand":
                    row.pop("prompt", None)
                row["state"] = "-"  # Record needs a state; evaluation does not read it
                row.setdefault("prompt", "-")
                out.append(json.dumps(row, ensure_ascii=False))
    return out


def _upload_stand(local: str) -> None:
    src = Path(local)
    if not src.is_file():
        raise FileNotFoundError(src)
    with data.batch_upload(force=True) as batch:
        batch.put_file(str(src), "/stand/stand.jsonl")
    print(f"uploaded {src} → volume jevlite-stage-c:/stand/stand.jsonl")


@app.local_entrypoint()
def main(fetch_stage_b: bool = False, convert: str = "", limit: int = 0, build: str = "", upload_stand: str = "",
         export_labels: str = ""):
    if upload_stand:
        _upload_stand(upload_stand)
    if fetch_stage_b:
        for line in do_fetch_stage_b.remote():
            print(line)
    if convert:
        names = list(STAGE_D_SOURCES) if convert == "stage_d" else [n.strip() for n in convert.split(",") if n.strip()]
        calls = {n: (do_convert_gated if n in GATED else do_convert).spawn(n, limit) for n in names}
        failed = []
        for n, call in calls.items():
            try:
                print(f"--- {n}\n{call.get()}")
            except Exception as e:  # noqa: BLE001 — report every source
                failed.append(n)
                print(f"--- {n}: FAILED\n{e}")
        if failed:
            raise SystemExit(f"failed: {', '.join(failed)}")
    if build:
        report = do_build.remote(build)
        import yaml

        mix = Path(yaml.safe_load(Path("configs/base.yaml").read_text())["train"][build]["mix"]).stem
        local = Path("data/mix") / mix / "mix_report.md"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(report)
        print(report + f"\n→ {local}")
    if export_labels:
        lines = do_export_labels.remote(export_labels)
        local = Path("data/mix") / export_labels / "labels.jsonl"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("\n".join(lines) + "\n")
        print(f"{len(lines)} eval records (no states) → {local} ({local.stat().st_size / 1e6:.1f} MB)")
    if not (fetch_stage_b or convert or build or upload_stand or export_labels):
        print("nothing to do: --fetch-stage-b, --convert, --build, --upload-stand, --export-labels")
