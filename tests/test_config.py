import pytest
import yaml

from jevlite.config import grad_accum_steps, load_config


@pytest.mark.parametrize("stack", ["apple_silicon", "cuda"])
def test_stack_configs_load(stack):
    cfg = load_config(stack)
    assert cfg["stack"] == stack
    assert cfg["model"]["base"] == "answerdotai/ModernBERT-large"


def test_stacks_differ_only_in_allowed_keys():
    a, c = load_config("apple_silicon"), load_config("cuda")
    for key in ("runtime", "env", "tuning", "train", "stack"):
        a.pop(key), c.pop(key)
    assert a == c
    a, c = load_config("apple_silicon"), load_config("cuda")
    assert a["tuning"]["lora"] == c["tuning"]["lora"]
    assert {k: v for k, v in a["train"].items() if k != "per_device_batch_size"} == {
        k: v for k, v in c["train"].items() if k != "per_device_batch_size"
    }


def test_effective_batch_is_equal_across_stacks():
    a, c = load_config("apple_silicon"), load_config("cuda")
    assert grad_accum_steps(a) * a["train"]["per_device_batch_size"] == 32
    assert grad_accum_steps(c, world_size=2) * c["train"]["per_device_batch_size"] * 2 == 32


def test_disallowed_stack_override_rejected(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"train": {"warmup_ratio": 0.06}}))
    (tmp_path / "cuda.yaml").write_text(yaml.safe_dump({"train": {"warmup_ratio": 0.1}}))
    with pytest.raises(ValueError, match="train.warmup_ratio"):
        load_config("cuda", config_dir=tmp_path)


def test_experiment_overrides_apply():
    cfg = load_config("apple_silicon", overrides={"encoding": {"max_len": 1024}})
    assert cfg["encoding"]["max_len"] == 1024
    assert cfg["encoding"]["head_ratio"] == 0.5
