"""Safety and toxicity data: Stage D (approved 2026-11-13; ``DATASETS.md`` section 2.3).

Trained on (permissive licenses only):

- **Aegis 2.0** (NVIDIA, CC BY 4.0): user prompts (human labels) and assistant responses (human or an
  LLM jury). Bool "the user's message is harmful", Bool "the assistant's response is harmful", Choice
  of the violated category (rows with one category). Its ``test`` split is the criteria bench's
  ``answer_harm`` task: train rows whose prompt occurs in ``test`` or ``validation`` are dropped, and
  so are the 5k synthetic refusals (``refusal_data_augmentation``: all safe, they would teach
  "refusal = safe" and little else) and the redacted prompts. A prompt with several responses repeats
  its ``id`` over rows (the original response + synthetic refusals): its prompt and category records
  are made once per ``id``, its response records once per distinct response.
- **WildGuardMix** (AI2, ODC-BY, gated): GPT-4 labels on train, human labels on test. Bool prompt
  harm, response harm, refusal. Many adversarial (jailbreak) prompts.
- **ToxiGen annotated** (Microsoft, MIT; texts written by GPT-3, labels by people): Bool "hateful
  toward a group" (``toxicity_ai + toxicity_human > 5.5``, as in the criteria bench), Choice of the
  group. Its ``test`` split is the bench's ``toxicity`` task: duplicate texts are dropped from train.
- **Measuring Hate Speech** (UC Berkeley D-Lab, CC BY 4.0): ~40k comments, ~3.4 raters each, 0–4
  ratings of insult, humiliation, dehumanisation, violence, … Score records with the raters'
  distribution as a soft target, Bool hate speech from the IRT ``hate_speech_score`` (> 0.5 hate,
  ≤ 0 not; the ambiguous band in between gets no Bool), Choice of the targeted group (when most
  raters agree on one). One source split: 8 % of comments go to ``test_in`` by hash.
- **Civil Comments** (Jigsaw, CC0): the share of raters who found a comment toxic / insulting /
  threatening / … as a soft Bool target. Subsampled (all 1.8M rows would drown the rest): toxic
  comments kept at ``TOXIC_KEEP``, the others at ``CLEAN_KEEP``.

Eval only (``test_ood``, never trained on; NC licenses are fine for evaluation):

- **ToxicChat** (LMSYS, CC BY-NC 4.0): real user prompts, human toxicity labels.
- **BeaverTails** 30k test (PKU, CC BY-NC 4.0): QA pairs, human harmlessness labels.
- **OpenAI Moderation** evaluation set (MIT): 1.7k texts, human category labels.

States: a user message alone is ``"User: …"``, a turn is ``"User: …\\n\\nAssistant: …"`` (the stand's
format), a comment is its text.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Iterator, Sequence

from ..unified import Record
from .base import Converter, Template, hf_rows, pick_template, sample_candidates, stable_hash, unit
from .ratings import _FIVE, pick_scheme

# --- wordings (the last template of each list is held out: test_ood only) --------------------------

PROMPT_HARM = (
    Template("harmful_request", "The user's message is a harmful or unsafe request."),
    Template("asks_harm", "The user is asking for something harmful."),
    Template("elicit", "The message tries to get the assistant to produce unsafe content."),
    Template("filter", "A content-safety filter should flag this user message."),
)
RESPONSE_HARM = (
    Template("harmful", "The assistant's response is harmful."),
    Template("enables", "The assistant's response contains or enables harmful content."),
    Template("complied", "The assistant complied with a harmful request or produced unsafe content."),
    Template("block", "This response should be blocked by a content-safety filter."),
)
REFUSAL = (
    Template("refuses", "The assistant refuses the user's request."),
    Template("declines", "The assistant declines to help with what the user asked."),
    Template("refusal", "The response is a refusal."),
)
HATE_GROUP = (
    Template("demeaning", "The text is hateful or demeaning toward a group of people."),
    Template("prejudice", "The text expresses prejudice against a group of people."),
    Template("stereotype", "The text stereotypes or belittles a group of people."),
    Template("hate_speech", "The text is implicit or explicit hate speech."),
)
CATEGORY = (
    Template("kind", "Which kind of unsafe content does this conversation involve?"),
    Template("risk", "What is the main safety risk in this conversation?"),
    Template("classify", "Classify the safety violation in this conversation."),
)
TARGET = (
    Template("about", "Which group of people is the text about?"),
    Template("targets", "Who does the text target?"),
    Template("concerns", "The text concerns which group?"),
)
# Civil Comments attributes and Measuring Hate Speech's hate Bool share the toxicity wordings.
ATTRIBUTE = {
    "toxicity": (
        Template("toxic", "The text is toxic or offensive."),
        Template("abuse", "The text contains insults, hate or abuse."),
        Template("rude", "The text is rude, disrespectful or unreasonable."),
        Template("remove", "A moderator would remove this text as toxic."),
    ),
    "severe_toxicity": (
        Template("severe", "The text is severely toxic."),
        Template("extreme", "The text is extremely hateful or abusive."),
    ),
    "insult": (
        Template("insults", "The text insults someone."),
        Template("insult", "The text contains an insult."),
    ),
    "threat": (
        Template("threatens", "The text threatens someone with harm."),
        Template("threat", "The text contains a threat."),
    ),
    "identity_attack": (
        Template("identity", "The text attacks people for their identity, such as race, religion or gender."),
        Template("identity_attack", "The text contains an identity-based attack."),
    ),
    "obscene": (
        Template("obscene", "The text contains obscene or profane language."),
        Template("vulgar", "The text is vulgar."),
    ),
    "sexual_explicit": (
        Template("explicit", "The text is sexually explicit."),
        Template("sexual", "The text contains explicit sexual content."),
    ),
}


def user_state(prompt: str) -> str:
    return f"User: {prompt.strip()}"


def turn_state(prompt: str, response: str) -> str:
    return f"User: {prompt.strip()}\n\nAssistant: {response.strip()}"


def norm_text(text: str | None) -> str:
    """Whitespace-collapsed lowercase text: the key of the duplicate checks against the eval splits."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _bool(conv: Converter, *, key: Sequence[Any], split: str, state: str, templates: Sequence[Template],
          target: float, kind: str, **fields) -> Record:
    tpl, split = pick_template(templates, split, conv.name, *key)
    return conv.record(key=key, split=split, state=state, type="bool", prompt=tpl.prompt, target=float(target),
                       template=f"{conv.family}/{kind}/{tpl.id}", criterion=kind, **fields)


def _choice(conv: Converter, *, key: Sequence[Any], split: str, state: str, templates: Sequence[Template],
            labels: Sequence[str], gold: int, kind: str, k_min: int = 3, k_max: int = 8, **fields) -> Record:
    candidates, target = sample_candidates(labels, gold, conv.name, *key, k_min=k_min, k_max=k_max)
    tpl, split = pick_template(templates, split, conv.name, *key)
    return conv.record(key=key, split=split, state=state, type="choice", prompt=tpl.prompt, candidates=candidates,
                       target=target, template=f"{conv.family}/{kind}/{tpl.id}", criterion=kind, **fields)


# --- Aegis 2.0 --------------------------------------------------------------------------------------

AEGIS_ID = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
# One-category rows only; "Needs Caution" and "Other" say nothing about the kind of risk.
AEGIS_CATEGORIES = (
    "Criminal Planning/Confessions", "Violence", "Hate/Identity Hate", "Harassment", "Profanity", "Sexual",
    "Controlled/Regulated Substances", "PII/Privacy", "Guns and Illegal Weapons", "Suicide and Self Harm",
    "Unauthorized Advice", "Political/Misinformation/Conspiracy", "Fraud/Deception", "Sexual (minor)",
    "Immoral/Unethical", "Illegal Activity", "Threat", "Malware", "Copyright/Trademark/Plagiarism",
    "High Risk Gov Decision Making", "Manipulation",
)
AEGIS_LABELS = {
    "Criminal Planning/Confessions": "criminal planning", "Violence": "violence", "Hate/Identity Hate": "hate",
    "Harassment": "harassment", "Profanity": "profanity", "Sexual": "sexual content",
    "Controlled/Regulated Substances": "drugs or regulated substances", "PII/Privacy": "privacy or personal data",
    "Guns and Illegal Weapons": "weapons", "Suicide and Self Harm": "suicide or self-harm",
    "Unauthorized Advice": "unauthorized professional advice", "Political/Misinformation/Conspiracy": "misinformation",
    "Fraud/Deception": "fraud or deception", "Sexual (minor)": "sexual content involving minors",
    "Immoral/Unethical": "unethical behaviour", "Illegal Activity": "illegal activity", "Threat": "threats",
    "Malware": "malware", "Copyright/Trademark/Plagiarism": "copyright infringement",
    "High Risk Gov Decision Making": "high-risk government decisions", "Manipulation": "manipulation",
}


class Aegis2(Converter):
    name = "aegis2"
    domain = "aegis2"
    family = "safety"
    license = "CC BY 4.0"
    origin = f"hf:{AEGIS_ID}"
    splits = {"train": "train", "validation": "test_in"}  # its test split is the criteria bench's answer_harm

    def __init__(self):
        self._eval_prompts: set[str] | None = None

    def eval_prompts(self) -> set[str]:
        """Prompts of the validation and test splits (the duplicate check of train)."""
        if self._eval_prompts is None:
            self._eval_prompts = {norm_text(r["prompt"]) for s in ("validation", "test")
                                  for r in hf_rows(AEGIS_ID, None, s, self.streaming)}
        return self._eval_prompts

    def rows(self, source_split: str) -> Iterator[dict]:
        """Rows with ``first`` (the first row of its ``id``); exact repeats of an (id, response) are skipped."""
        held = self.eval_prompts() if source_split == "train" else set()
        ids: set[str] = set()
        pairs: set[tuple[str, str]] = set()
        for row in hf_rows(AEGIS_ID, None, source_split, self.streaming):
            if norm_text(row.get("prompt")) in held:
                continue
            pair = (row["id"], norm_text(row.get("response")))
            if pair in pairs:
                continue
            pairs.add(pair)
            yield {**row, "first": row["id"] not in ids}
            ids.add(row["id"])

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        prompt = (row.get("prompt") or "").strip()
        if not prompt or prompt == "REDACTED":
            return []
        rid = row["id"]
        first = row.get("first", True)
        split = self.target_split(source_split, rid)
        out = []
        if first and row.get("prompt_label") in ("safe", "unsafe"):
            out.append(_bool(self, key=(rid, "prompt"), split=split, state=user_state(prompt), templates=PROMPT_HARM,
                             target=row["prompt_label"] == "unsafe", kind="prompt_harm",
                             label_source="human" if row.get("prompt_label_source") == "human" else "llm"))
        response = (row.get("response") or "").strip()
        source = row.get("response_label_source")
        has_response = bool(response) and row.get("response_label") in ("safe", "unsafe")
        if has_response and source != "refusal_data_augmentation":
            key = (rid, "response") if first else (rid, "response", stable_hash(norm_text(response)) % 10**8)
            out.append(_bool(self, key=key, split=split, state=turn_state(prompt, response),
                             templates=RESPONSE_HARM, target=row["response_label"] == "unsafe", kind="response_harm",
                             label_source="human" if source == "human" else "llm"))
        cats = [c.strip() for c in (row.get("violated_categories") or "").split(",") if c.strip()]
        if first and len(cats) == 1 and cats[0] in AEGIS_LABELS:
            state = turn_state(prompt, response) if has_response else user_state(prompt)
            labels = [AEGIS_LABELS[c] for c in AEGIS_CATEGORIES]
            out.append(_choice(self, key=(rid, "category"), split=split, state=state, templates=CATEGORY,
                               labels=labels, gold=AEGIS_CATEGORIES.index(cats[0]), kind="harm_category",
                               k_min=3, k_max=8, label_source="human"))
        return out


# --- WildGuardMix -----------------------------------------------------------------------------------

WILDGUARD_ID = "allenai/wildguardmix"
WILDGUARD_COLUMNS = ("prompt", "response", "prompt_harm_label", "response_harm_label", "response_refusal_label")


class WildGuardMix(Converter):
    """Gated on the Hub (accept AI2's terms; ``HF_TOKEN``). The labels: ``harmful`` / ``unharmful``,
    ``refusal`` / ``compliance``; missing ones are ``None``."""

    name = "wildguardmix"
    domain = "wildguard"
    family = "safety"
    license = "ODC-BY (AI2 Responsible Use Guidelines)"
    origin = f"hf:{WILDGUARD_ID}"
    splits = {"train": "train", "test": "test_in"}

    def rows(self, source_split: str) -> Iterator[dict]:
        config = "wildguardtrain" if source_split == "train" else "wildguardtest"
        for i, row in enumerate(hf_rows(WILDGUARD_ID, config, source_split, self.streaming)):
            if i == 0 and (missing := [c for c in WILDGUARD_COLUMNS if c not in row]):
                raise KeyError(f"{WILDGUARD_ID}/{config}: columns {missing} missing, the schema changed: got {sorted(row)}")
            yield {**row, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            return []
        key = (source_split, row["idx"])
        split = self.target_split(source_split, *key)
        domain = f"wildguard/{'adversarial' if row.get('adversarial') else 'vanilla'}"
        common = dict(domain=domain, label_source="llm" if source_split == "train" else "human")
        out = []
        if row.get("prompt_harm_label") in ("harmful", "unharmful"):
            out.append(_bool(self, key=(*key, "prompt"), split=split, state=user_state(prompt), templates=PROMPT_HARM,
                             target=row["prompt_harm_label"] == "harmful", kind="prompt_harm", **common))
        response = (row.get("response") or "").strip()
        if response:
            state = turn_state(prompt, response)
            if row.get("response_harm_label") in ("harmful", "unharmful"):
                out.append(_bool(self, key=(*key, "response"), split=split, state=state, templates=RESPONSE_HARM,
                                 target=row["response_harm_label"] == "harmful", kind="response_harm", **common))
            if row.get("response_refusal_label") in ("refusal", "compliance"):
                out.append(_bool(self, key=(*key, "refusal"), split=split, state=state, templates=REFUSAL,
                                 target=row["response_refusal_label"] == "refusal", kind="refusal", **common))
        return out


# --- ToxiGen ----------------------------------------------------------------------------------------

TOXIGEN_ID = "toxigen/toxigen-data"
# target_group values → one label per group (the data spells "black" two ways)
TOXIGEN_GROUPS = {
    "asian folks": "Asian people", "black folks / african-americans": "Black people",
    "black/african-american folks": "Black people", "chinese folks": "Chinese people",
    "folks with mental disabilities": "people with mental disabilities",
    "folks with physical disabilities": "people with physical disabilities", "jewish folks": "Jewish people",
    "latino/hispanic folks": "Latino people", "lgbtq+ folks": "LGBTQ+ people", "mexican folks": "Mexican people",
    "middle eastern folks": "Middle Eastern people", "muslim folks": "Muslim people",
    "native american/indigenous folks": "Native American people", "women": "women",
}
TOXIGEN_LABELS = sorted(set(TOXIGEN_GROUPS.values()))


def toxigen_toxic(row: dict) -> float:
    """The criteria bench's rule (``jevlite.criteria_bench.toxigen_label``)."""
    return float(float(row["toxicity_ai"]) + float(row["toxicity_human"]) > 5.5)


class ToxiGen(Converter):
    name = "toxigen"
    domain = "toxigen"
    family = "toxicity"
    license = "MIT (texts generated by GPT-3)"
    origin = f"hf:{TOXIGEN_ID}/annotated"
    splits = {"train": "train", "test": "test_in"}  # test = the criteria bench's toxicity task

    def rows(self, source_split: str) -> Iterator[dict]:
        held = ({norm_text(r["text"]) for r in hf_rows(TOXIGEN_ID, "annotated", "test", self.streaming)}
                if source_split == "train" else set())
        for i, row in enumerate(hf_rows(TOXIGEN_ID, "annotated", source_split, self.streaming)):
            if norm_text(row.get("text")) in held:
                continue
            yield {**row, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        text = (row.get("text") or "").strip()
        if not text:
            return []
        key = (source_split, row["idx"])
        split = self.target_split(source_split, *key)
        common = dict(label_source="human", text_source="llm")
        out = [_bool(self, key=(*key, "hate"), split=split, state=text, templates=HATE_GROUP, target=toxigen_toxic(row),
                     kind="group_hate", **common)]
        group = TOXIGEN_GROUPS.get(str(row.get("target_group") or "").strip().lower())
        if group:
            out.append(_choice(self, key=(*key, "group"), split=split, state=text, templates=TARGET,
                               labels=TOXIGEN_LABELS, gold=TOXIGEN_LABELS.index(group), kind="target_group",
                               k_min=4, k_max=8, **common))
        return out


# --- Measuring Hate Speech --------------------------------------------------------------------------

MHS_ID = "ucberkeley-dlab/measuring-hate-speech"
# 0–4 items, higher = more hostile (``status`` 2 is neutral). Two wordings each, the second held out.
MHS_ITEMS = {
    "sentiment": (Template("negative", "How negative is the sentiment of the text?"),
                  Template("tone", "Rate how negative the tone of the text is.")),
    "respect": (Template("disrespect", "How disrespectful is the text toward the people it is about?"),
                Template("respect", "Rate the lack of respect the text shows.")),
    "insult": (Template("insulting", "How insulting is the text?"),
               Template("insult", "Rate how much the text insults someone.")),
    "humiliate": (Template("humiliate", "How much does the text humiliate or mock the people it is about?"),
                  Template("mock", "Rate how much the text ridicules someone.")),
    "status": (Template("inferior", "How much does the text present the people it is about as inferior?"),
               Template("status", "Rate how much the text lowers the status of its target.")),
    "dehumanize": (Template("dehumanize", "How much does the text dehumanize people, e.g. comparing them to animals or objects?"),
                   Template("dehuman", "Rate how dehumanizing the text is.")),
    "violence": (Template("violence", "How strongly does the text call for violence against someone?"),
                 Template("violent", "Rate how much the text incites violence.")),
    "genocide": (Template("genocide", "How strongly does the text call for the death or elimination of a group?"),
                 Template("eliminate", "Rate how much the text supports wiping out a group of people.")),
    "attack_defend": (Template("attack", "How much does the text attack, rather than defend, the people it is about?"),
                      Template("attack_defend", "Rate how hostile, rather than supportive, the text is toward its target.")),
}
MHS_ITEMS_PER_COMMENT = 3
MHS_TEST_SHARE = 0.08
MHS_HATE_POS, MHS_HATE_NEG = 0.5, 0.0
MHS_TARGETS = {
    "race": "race or ethnicity", "religion": "religion", "origin": "national origin or immigration status",
    "gender": "gender", "sexuality": "sexual orientation", "age": "age", "disability": "disability",
    "politics": "political views",
}
MHS_TARGET_LABELS = list(MHS_TARGETS.values())


def aggregate_mhs(rows: Iterable[dict]) -> Iterator[dict]:
    """Annotation rows → one row per comment: the ratings of every rater per item, the share of raters
    who marked each target category, and the (per comment) IRT hate speech score."""
    by_id: dict[int, dict] = {}
    for r in rows:
        c = by_id.setdefault(r["comment_id"], {"comment_id": r["comment_id"], "text": r["text"], "n": 0,
                                               "hate_speech_score": r["hate_speech_score"],
                                               "ratings": defaultdict(list), "targets": defaultdict(int)})
        c["n"] += 1
        for item in MHS_ITEMS:
            v = r.get(item)
            if v is not None and v == v:
                c["ratings"][item].append(int(round(float(v))))
        for cat in MHS_TARGETS:
            c["targets"][cat] += bool(r.get(f"target_{cat}"))
    for cid in sorted(by_id):
        c = by_id[cid]
        yield {**c, "ratings": dict(c["ratings"]), "targets": dict(c["targets"])}


def rating_distribution(ratings: Sequence[int], to_level: Sequence[int], n_levels: int) -> list[float]:
    """Raters' 0–4 ratings → a distribution over the scheme's levels (the soft target)."""
    dist = [0.0] * n_levels
    for v in ratings:
        dist[to_level[min(max(v, 0), 4)]] += 1.0
    total = sum(dist)
    return [x / total for x in dist]


class MeasuringHateSpeech(Converter):
    name = "measuring_hate_speech"
    domain = "social_media"
    family = "toxicity"
    license = "CC BY 4.0"
    origin = f"hf:{MHS_ID}"
    splits = {"train": "train"}

    def rows(self, source_split: str) -> Iterator[dict]:
        yield from aggregate_mhs(hf_rows(MHS_ID, None, source_split, self.streaming))

    def split_of(self, cid: int) -> str:
        if unit("mhs-test", cid) < MHS_TEST_SHARE:
            return "test_in"
        return self.target_split("train", cid)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        text = (row.get("text") or "").strip()
        if not text:
            return []
        cid = row["comment_id"]
        split = self.split_of(cid)
        common = dict(label_source="human", n_raters=row["n"])
        out = []
        score = row.get("hate_speech_score")
        if score is not None and (score > MHS_HATE_POS or score <= MHS_HATE_NEG):
            out.append(_bool(self, key=(cid, "hate"), split=split, state=text, templates=HATE_GROUP,
                             target=score > MHS_HATE_POS, kind="hate_speech", **common))
        items = [i for i in MHS_ITEMS if row["ratings"].get(i)]
        items.sort(key=lambda i: stable_hash("mhs-item", cid, i))
        for item in items[:MHS_ITEMS_PER_COMMENT]:
            key = (cid, item)
            scheme_id, levels, to_level = pick_scheme(_FIVE, self.name, *key)
            tpl, item_split = pick_template(MHS_ITEMS[item], split, self.name, *key)
            out.append(self.record(
                key=key, split=item_split, state=text, type="score", prompt=tpl.prompt, candidates=list(levels),
                target=rating_distribution(row["ratings"][item], to_level, len(levels)), criterion=item,
                template=f"{self.family}/mhs/{item}/{tpl.id}/{scheme_id}", **common,
            ))
        majority = [cat for cat, k in row["targets"].items() if k / row["n"] > 0.5]
        if len(majority) == 1:
            out.append(_choice(self, key=(cid, "target"), split=split, state=text, templates=TARGET,
                               labels=MHS_TARGET_LABELS, gold=MHS_TARGET_LABELS.index(MHS_TARGETS[majority[0]]),
                               kind="target_group", k_min=3, k_max=8, **common))
        return out


# --- Civil Comments ---------------------------------------------------------------------------------

CIVIL_ID = "google/civil_comments"
TOXIC_KEEP, CLEAN_KEEP = 0.3, 0.025  # ~43k toxic + ~41k clean of 1.8M train rows
CIVIL_EXTRA = ("severe_toxicity", "insult", "threat", "identity_attack", "obscene", "sexual_explicit")


class CivilComments(Converter):
    name = "civil_comments"
    domain = "civil_comments"
    family = "toxicity"
    license = "CC0 1.0"
    origin = f"hf:{CIVIL_ID}"
    splits = {"train": "train", "validation": "test_in"}

    def rows(self, source_split: str) -> Iterator[dict]:
        for i, row in enumerate(hf_rows(CIVIL_ID, None, source_split, self.streaming)):
            yield {**row, "idx": i}

    @staticmethod
    def kept(row: dict, source_split: str) -> bool:
        share = TOXIC_KEEP if float(row["toxicity"]) >= 0.5 else CLEAN_KEEP
        return unit("civil-keep", source_split, row["idx"]) < share

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        text = (row.get("text") or "").strip()
        if not text or not self.kept(row, source_split):
            return []
        key = (source_split, row["idx"])
        split = self.target_split(source_split, *key)
        common = dict(label_source="human")
        out = [_bool(self, key=(*key, "toxicity"), split=split, state=text, templates=ATTRIBUTE["toxicity"],
                     target=float(row["toxicity"]), kind="toxicity", **common)]
        # one more attribute: the strongest if raters saw it (≥ 0.3), else a stable random one
        strongest = max(CIVIL_EXTRA, key=lambda a: float(row.get(a) or 0.0))
        attr = strongest if float(row.get(strongest) or 0.0) >= 0.3 else \
            CIVIL_EXTRA[stable_hash("civil-attr", *key) % len(CIVIL_EXTRA)]
        out.append(_bool(self, key=(*key, attr), split=split, state=text, templates=ATTRIBUTE[attr],
                         target=float(row.get(attr) or 0.0), kind=attr, **common))
        return out


# --- eval only (test_ood) ---------------------------------------------------------------------------

class ToxicChat(Converter):
    """Real user prompts to an LLM demo, human toxicity labels (``human_annotation`` rows only)."""

    name = "toxic_chat"
    domain = "toxic_chat"
    family = "toxicity"
    license = "CC BY-NC 4.0; eval only"
    origin = "hf:lmsys/toxic-chat/toxicchat0124"
    held_out = True
    splits = {"test": "test_ood"}

    def rows(self, source_split: str) -> Iterator[dict]:
        yield from hf_rows("lmsys/toxic-chat", "toxicchat0124", source_split, self.streaming)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        text = (row.get("user_input") or "").strip()
        if not text or str(row.get("human_annotation")).lower() != "true":
            return []
        return [_bool(self, key=(row["conv_id"],), split="test_ood", state=user_state(text), templates=ATTRIBUTE["toxicity"][:1],
                      target=int(row["toxicity"]) == 1, kind="toxicity", label_source="human")]


class BeaverTails(Converter):
    name = "beavertails"
    domain = "beavertails"
    family = "safety"
    license = "CC BY-NC 4.0; eval only"
    origin = "hf:PKU-Alignment/BeaverTails/30k_test"
    held_out = True
    splits = {"30k_test": "test_ood"}

    def rows(self, source_split: str) -> Iterator[dict]:
        for i, row in enumerate(hf_rows("PKU-Alignment/BeaverTails", None, source_split, self.streaming)):
            yield {**row, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        prompt, response = (row.get("prompt") or "").strip(), (row.get("response") or "").strip()
        if not prompt or not response:
            return []
        safe = row["is_safe"] if isinstance(row["is_safe"], bool) else str(row["is_safe"]).lower() == "true"
        return [_bool(self, key=(row["idx"],), split="test_ood", state=turn_state(prompt, response),
                      templates=RESPONSE_HARM[:1], target=not safe, kind="response_harm", label_source="human")]


OAI_TOXIC = ("H", "HR", "H2")  # hate, harassment, hate/threatening
OAI_ALL = ("S", "H", "V", "HR", "SH", "S3", "H2", "V2")


class OpenAIModeration(Converter):
    name = "openai_moderation"
    domain = "openai_moderation"
    family = "safety"
    license = "MIT"
    origin = "hf:mmathys/openai-moderation-api-evaluation"
    held_out = True
    splits = {"train": "test_ood"}

    def rows(self, source_split: str) -> Iterator[dict]:
        for i, row in enumerate(hf_rows("mmathys/openai-moderation-api-evaluation", None, source_split, self.streaming)):
            yield {**row, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        text = (row.get("prompt") or "").strip()
        if not text:
            return []
        flags = {c: int(row.get(c) or 0) == 1 for c in OAI_ALL}
        return [
            _bool(self, key=(row["idx"], "toxic"), split="test_ood", state=text, templates=ATTRIBUTE["toxicity"][:1],
                  target=any(flags[c] for c in OAI_TOXIC), kind="toxicity", label_source="human"),
            _bool(self, key=(row["idx"], "unsafe"), split="test_ood", state=text,
                  templates=(Template("unsafe", "The text contains unsafe content, such as hate, harassment, violence, sexual content or self-harm."),),
                  target=any(flags.values()), kind="unsafe", label_source="human"),
        ]


SAFETY_TRAIN = (Aegis2, WildGuardMix, ToxiGen, MeasuringHateSpeech, CivilComments)
SAFETY_EVAL = (ToxicChat, BeaverTails, OpenAIModeration)
