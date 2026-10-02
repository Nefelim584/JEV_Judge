"""Classification with labels as options: intents, topics, emotions (``DATASETS.md`` section 0).

Label sets are larger than ``MAX_CANDIDATES`` or than a request would list, so every row gets the gold
label plus a random number of random distractors (``sample_candidates``).
"""

from __future__ import annotations

import csv
import json
import tarfile
from typing import Iterable, Iterator, Sequence

from ..unified import Record
from .base import Converter, Template, fetch, humanize, pick_template, sample_candidates

INTENT = (
    Template("want", "What does the user want?"),
    Template("intent", "What is the intent of this message?"),
    Template("request", "Which request is the user making?"),
)
TOPIC = (
    Template("about", "What kind of entity is this article about?"),
    Template("category", "Which category does this article belong to?"),
    Template("subject", "The subject of the text is a"),
)
EMOTION = (
    Template("express", "Which emotion does the text express?"),
    Template("feeling", "How does the writer feel?"),
    Template("emotion", "The main emotion in this comment is"),
)


def label_record(
    conv: Converter, *, key: Sequence, source_split: str, state: str, labels: Sequence[str], gold: int,
    templates: Sequence[Template], k_max: int = 16,
) -> list[Record]:
    if not state.strip():
        return []
    candidates, target = sample_candidates(labels, gold, conv.name, *key, k_max=k_max)
    tpl, split = pick_template(templates, conv.target_split(source_split, *key), conv.name, *key)
    return [
        conv.record(
            key=key, split=split, state=state, type="choice", prompt=tpl.prompt, candidates=candidates,
            target=target, template=f"{conv.family}/{tpl.id}", n_labels=len(labels),
        )
    ]


def _hf_with_names(hf_id: str, config: str | None, split: str, label_col: str, streaming: bool) -> Iterator[dict]:
    """Rows of an HF dataset with the ClassLabel names attached as ``label_names``."""
    from datasets import load_dataset

    ds = load_dataset(hf_id, config, split=split, streaming=streaming)
    feature = ds.features[label_col]
    names = feature.feature.names if hasattr(feature, "feature") else feature.names
    for i, row in enumerate(ds):
        yield {**row, "idx": i, "label_names": names}


class CLINC150(Converter):
    """150 intents plus out-of-scope, which becomes the option "none of these"."""

    name = "clinc150"
    domain = "intents"
    family = "intent"
    license = "CC BY 3.0"
    origin = "hf:clinc/clinc_oos/plus"
    splits = {"train": "train", "test": "test_in"}

    def rows(self, source_split: str) -> Iterable[dict]:
        return _hf_with_names("clinc/clinc_oos", "plus", source_split, "intent", self.streaming)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        labels = ["none of these" if n == "oos" else humanize(n) for n in row["label_names"]]
        return label_record(
            self, key=(source_split, row["idx"]), source_split=source_split, state=row["text"],
            labels=labels, gold=row["intent"], templates=INTENT,
        )


class MASSIVE(Converter):
    """The en-US part of MASSIVE 1.1 (60 intents)."""

    name = "massive"
    domain = "voice_assistant"
    family = "intent"
    license = "CC BY 4.0"
    origin = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
    splits = {"train": "train", "test": "test_in"}

    def _all_rows(self) -> list[dict]:
        with tarfile.open(fetch(self.origin)) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("data/en-US.jsonl"))
            return [json.loads(line) for line in tar.extractfile(member).read().decode().splitlines() if line.strip()]

    def rows(self, source_split: str) -> Iterator[dict]:
        rows = self._all_rows()
        labels = sorted({r["intent"] for r in rows})
        for r in rows:
            if r["partition"] == source_split:
                yield {**r, "label_names": labels}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        labels = [humanize(n) for n in row["label_names"]]
        return label_record(
            self, key=(source_split, row["id"]), source_split=source_split, state=row["utt"],
            labels=labels, gold=row["label_names"].index(row["intent"]), templates=INTENT,
        )


class BANKING77(Converter):
    """77 banking intents. Held out: its test split goes to ``test_ood`` only."""

    name = "banking77"
    domain = "banking"
    family = "intent"
    license = "CC BY 4.0"
    origin = "https://github.com/PolyAI-LDN/task-specific-datasets"
    held_out = True
    splits = {"test": "test_ood"}

    _URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{split}.csv"

    def rows(self, source_split: str) -> Iterator[dict]:
        path = fetch(self._URL.format(split=source_split), f"banking77_{source_split}.csv")
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        labels = sorted({r["category"] for r in rows})
        for i, r in enumerate(rows):
            yield {**r, "idx": i, "label_names": labels}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        labels = [humanize(n) for n in row["label_names"]]
        return label_record(
            self, key=(source_split, row["idx"]), source_split=source_split, state=row["text"],
            labels=labels, gold=row["label_names"].index(row["category"]), templates=INTENT,
        )


class DBpedia14(Converter):
    name = "dbpedia14"
    domain = "wiki"
    family = "topic"
    license = "CC BY-SA 3.0"
    origin = "hf:fancyzhx/dbpedia_14"
    splits = {"train": "train", "test": "test_in"}

    # "EducationalInstitution" → "educational institution"
    _NAMES = {
        "Company": "company", "EducationalInstitution": "educational institution", "Artist": "artist",
        "Athlete": "athlete", "OfficeHolder": "office holder", "MeanOfTransportation": "means of transportation",
        "Building": "building", "NaturalPlace": "natural place", "Village": "village", "Animal": "animal",
        "Plant": "plant", "Album": "album", "Film": "film", "WrittenWork": "written work",
    }

    def rows(self, source_split: str) -> Iterable[dict]:
        return _hf_with_names("fancyzhx/dbpedia_14", None, source_split, "label", self.streaming)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        labels = [self._NAMES.get(n, n) for n in row["label_names"]]
        return label_record(
            self, key=(source_split, row["idx"]), source_split=source_split,
            state=f"{row['title'].strip()}\n{row['content'].strip()}", labels=labels, gold=row["label"],
            templates=TOPIC, k_max=14,
        )


class GoEmotions(Converter):
    """Reddit comments with 28 emotion labels; single-label comments only."""

    name = "go_emotions"
    domain = "reddit"
    family = "emotion"
    license = "Apache-2.0"
    origin = "hf:google-research-datasets/go_emotions/simplified"
    splits = {"train": "train", "test": "test_in"}

    def rows(self, source_split: str) -> Iterable[dict]:
        return _hf_with_names("google-research-datasets/go_emotions", "simplified", source_split, "labels", self.streaming)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        if len(row["labels"]) != 1:
            return []
        return label_record(
            self, key=(source_split, row["id"]), source_split=source_split, state=row["text"],
            labels=row["label_names"], gold=row["labels"][0], templates=EMOTION,
        )
