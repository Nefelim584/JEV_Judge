"""Scorers of the criteria bench (``jevlite.criteria_bench``): jev-lite, general judges and specialists.

Every scorer maps items to P(positive) — P(the item's statement is true) — so that one set of metrics
fits all. Specialists whose own output means "risk" (Granite Guardian on groundedness, HHEM's
"hallucinated", LettuceDetect's hallucinated tokens) are flipped where the task's positive is the safe
side. A scorer may return ``nan`` for an item it could not read (counted as missing).

| Scorer | Model | Tasks | Read-out |
|---|---|---|---|
| ``jevlite`` | ``hub:Nefelim584/jev-lite@v0.2.0`` or a Stage C ``final/`` | all | Bool question, bf16, model-layer calibration |
| ``nli`` | DeBERTa-v3-large NLI (``baselines.nli``) | all | P(entailment), statement as hypothesis, 512 tokens |
| ``llm_judge`` | Qwen3-8B (no thinking) | all | P(Yes) vs P(No) of the first answer token |
| ``granite_guardian`` | ibm-granite/granite-guardian-3.3-8b | all | P(yes) at the ``<score>`` token; pre-baked criteria, ``custom_criteria`` for custom tasks |
| ``hhem`` | vectara/hallucination_evaluation_model (HHEM-2.1-open) | grounding | P(consistent) of (passage, claim) |
| ``lettucedetect`` | KRLabsOrg/lettucedect-large-modernbert-en-v1 | grounding | 1 − max hallucination prob over answer tokens |
| ``toxic_bert`` | unitary/toxic-bert | the harm and toxicity tasks | max sigmoid over its 6 labels, on the reply / text |

LettuceDetect is read without its package (which pulls an OpenAI client and RAG tooling): the QA
prompt template and the (prompt, answer) pair encoding below follow ``lettucedetect`` 0.2.3
(``detectors/transformer.py``, ``prompts/qa_prompt_en.txt``; MIT), one pass up to 8192 tokens
instead of its 4096-token passage chunking.
"""

from __future__ import annotations

import math
from string import Template
from typing import Sequence

import numpy as np

from ..criteria_bench import Item, iter_batches
from ..schema import BoolQuestion

SCORERS = ("jevlite", "nli", "llm_judge", "granite_guardian", "hhem", "lettucedetect", "toxic_bert")
DEFAULT_JEVLITE = "hub:Nefelim584/jev-lite@v0.2.0"
DEFAULT_LLM = "Qwen/Qwen3-8B"
GRANITE = "ibm-granite/granite-guardian-3.3-8b"
HHEM = "vectara/hallucination_evaluation_model"
LETTUCE = "KRLabsOrg/lettucedect-large-modernbert-en-v1"
TOXIC_BERT = "unitary/toxic-bert"

NAN = float("nan")


def _by_length(it: Item) -> int:
    return len(it.state)


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


def p_yes(logits: np.ndarray, yes_ids: Sequence[int], no_ids: Sequence[int]) -> float:
    """P(yes) from one step's logits, restricted to the yes / no token variants."""
    z = np.asarray(logits, dtype=np.float64)
    a, b = _logsumexp(z[list(yes_ids)]), _logsumexp(z[list(no_ids)])
    return 1.0 / (1.0 + math.exp(b - a))


def p_yes_from_generation(tokens: Sequence[int], step_logits: Sequence[np.ndarray], yes_ids: Sequence[int],
                          no_ids: Sequence[int]) -> float:
    """The first generated yes / no token decides; P(yes) is read at that step. ``nan`` if none came."""
    wanted = set(yes_ids) | set(no_ids)
    for tok, z in zip(tokens, step_logits):
        if int(tok) in wanted:
            return p_yes(z, yes_ids, no_ids)
    return NAN


def single_token_ids(tokenizer, words: Sequence[str]) -> list[int]:
    """Ids of the variants of ``words`` that are one token (``"yes"``, ``" yes"``, ``"Yes"``, …)."""
    ids = []
    for w in words:
        for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            enc = tokenizer.encode(v, add_special_tokens=False)
            if len(enc) == 1 and enc[0] not in ids:
                ids.append(enc[0])
    if not ids:
        raise ValueError(f"none of {words} is a single token")
    return ids


def split_on_oom(fn, batch: list) -> list[float]:
    """``fn(batch)``; on CUDA out-of-memory the batch is halved and retried, and an item that does not fit
    alone gets ``nan`` (counted as missing). Batches are sorted by length, so only the longest split."""
    import torch

    try:
        return fn(batch)
    except torch.cuda.OutOfMemoryError:
        pass  # leave the except block first: its traceback still holds the failed batch's tensors
    torch.cuda.empty_cache()
    if len(batch) == 1:
        return [NAN]
    mid = len(batch) // 2
    return split_on_oom(fn, batch[:mid]) + split_on_oom(fn, batch[mid:])


class Scorer:
    name: str = "scorer"
    tasks: frozenset[str] | None = None  # None: every task, custom ones included

    def supports(self, item: Item) -> bool:
        return self.tasks is None or item.task in self.tasks

    batch_size: int = 1

    def score(self, items: Sequence[Item]) -> list[float]:
        """P(positive) per item, in order: length-sorted batches through ``_batch``, split on OOM."""
        out: dict[str, float] = {}
        for batch in iter_batches(items, self.batch_size, _by_length):
            out.update(zip((it.id for it in batch), split_on_oom(self._batch, batch), strict=True))
        return [out[it.id] for it in items]

    def _batch(self, batch: list[Item]) -> list[float]:
        raise NotImplementedError


# --- general judges --------------------------------------------------------------------------------

class JevLiteScorer(Scorer):
    def __init__(self, spec: str = DEFAULT_JEVLITE, device: str = "cuda", batch_size: int = 8, bf16: bool = True):
        import torch

        from ..inference import Predictor
        from ..model import resolve_checkpoint

        self.predictor = Predictor.from_final(resolve_checkpoint(spec), device=device,
                                              amp_dtype=torch.bfloat16 if bf16 else None)
        parts = [p for p in spec.removeprefix("hub:").split("/") if p]
        label = parts[-2] if len(parts) > 1 and parts[-1] == "final" else parts[-1]
        self.name = f"jev-lite ({label})"
        self.batch_size = batch_size

    def _batch(self, batch: list[Item]) -> list[float]:
        rows, _ = self.predictor._probs([(it.state, BoolQuestion(id="q", prompt=it.prompt)) for it in batch])
        return [float(r[0]) for r in rows]


class NLIScorer(Scorer):
    def __init__(self, device: str = "cuda", batch_size: int = 32):
        from .nli import NLIZeroShot

        self.nli = NLIZeroShot(device=device, max_len=512, batch_size=batch_size, fp16=device == "cuda")
        self.name = "NLI DeBERTa-v3-large (zero-shot)"

    def score(self, items: Sequence[Item]) -> list[float]:
        res = self.nli.predict_many([(it.state, [BoolQuestion(id="q", prompt=it.prompt)]) for it in items])
        return [float(r[0][0]) for r in res]


LLM_SYSTEM = "You are a careful evaluator. You judge a statement only by the text you are given."
LLM_USER = (
    "Text:\n<<<\n{state}\n>>>\n\nStatement: {prompt}\n\n"
    "Judge the statement only by what the text says or shows. If the text does not support it, the answer is No.\n"
    "Is the statement true? Answer with one word: Yes or No."
)


class LLMJudgeScorer(Scorer):
    """An open instruction model as a plain LLM-as-a-judge: the same statement as jev-lite, P(Yes)
    from the logits of the first answer token (no sampling, no thinking)."""

    def __init__(self, model: str = DEFAULT_LLM, device: str = "cuda", batch_size: int = 8, max_len: int = 8192):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.device, self.batch_size, self.max_len = device, batch_size, max_len
        self.yes, self.no = single_token_ids(self.tok, ["yes"]), single_token_ids(self.tok, ["no"])
        self.name = f"LLM judge {model.rsplit('/', 1)[-1]}"

    def _fit(self, text: str, budget: int) -> str:
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) <= budget:
            return text
        head = budget // 2
        return self.tok.decode(ids[:head]) + "\n[…]\n" + self.tok.decode(ids[-(budget - head):])

    def _chat(self, it: Item) -> str:
        msgs = [{"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": LLM_USER.format(state=self._fit(it.state, self.max_len - 512), prompt=it.prompt)}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _batch(self, batch: list[Item]) -> list[float]:
        import torch

        enc = self.tok([self._chat(it) for it in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with torch.no_grad():
            # logits_to_keep=1: the full-vocabulary logits of every position would not fit at 8k × batch
            logits = self.model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float().cpu().numpy()
        return [p_yes(z, self.yes, self.no) for z in logits]


class GraniteGuardianScorer(Scorer):
    """Granite Guardian 3.3: a guard LLM with pre-baked criteria; "yes" means the risk is present.

    grounding → ``groundedness`` (passage as a document; P(supported) = 1 − P(yes)); answer_harm →
    ``harm`` on the assistant turn; toxicity → ``social_bias`` on the text as a user turn; a custom
    task → its statement as ``custom_criteria`` (P(yes) = P(statement true)).
    """

    PREBAKED = {"grounding": ("groundedness", True), "answer_harm": ("harm", False), "toxicity": ("social_bias", False),
                "beavertails": ("harm", False), "toxic_chat": ("harm", False), "oai_moderation": ("harm", False)}
    RESPONSE_TASKS = ("answer_harm", "beavertails")

    def __init__(self, model: str = GRANITE, device: str = "cuda", batch_size: int = 8, max_new_tokens: int = 32):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.device, self.batch_size, self.max_new_tokens = device, batch_size, max_new_tokens
        self.yes, self.no = single_token_ids(self.tok, ["yes"]), single_token_ids(self.tok, ["no"])
        self.name = "Granite Guardian 3.3 8B"

    @classmethod
    def inputs(cls, it: Item) -> tuple[list[dict], dict, list[dict] | None, bool]:
        """``(messages, guardian_config, documents, invert)`` for one item."""
        if it.task in cls.PREBAKED:
            criteria, invert = cls.PREBAKED[it.task]
            config = {"criteria_id": criteria}
            if it.task == "grounding":
                return ([{"role": "user", "content": it.question}, {"role": "assistant", "content": it.answer}], config,
                        [{"doc_id": "0", "text": it.context}], invert)
            if it.task in cls.RESPONSE_TASKS:
                return [{"role": "user", "content": it.user}, {"role": "assistant", "content": it.answer}], config, None, invert
            return [{"role": "user", "content": it.state}], config, None, invert
        config = {"custom_criteria": it.prompt}
        docs = [{"doc_id": "0", "text": it.context}] if it.context else None
        if it.answer:
            user = it.user or it.question or "(no user message)"
            return [{"role": "user", "content": user}, {"role": "assistant", "content": it.answer}], config, docs, False
        return [{"role": "user", "content": it.state}], config, docs, False

    def _chat(self, it: Item) -> tuple[str, bool]:
        messages, config, docs, invert = self.inputs(it)
        kw = {"documents": docs} if docs else {}
        text = self.tok.apply_chat_template(messages, guardian_config=config, think=False, tokenize=False,
                                            add_generation_prompt=True, **kw)
        return text, invert

    def _batch(self, batch: list[Item]) -> list[float]:
        import torch

        chats = [self._chat(it) for it in batch]
        # no truncation: it would cut the end of the chat (the criteria and the generation prompt)
        enc = self.tok([c for c, _ in chats], return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      output_scores=True, return_dict_in_generate=True,
                                      pad_token_id=self.tok.pad_token_id)
        n_in = enc["input_ids"].shape[1]
        steps = torch.stack(gen.scores, dim=1).float().cpu().numpy()  # (batch, steps, vocab)
        out = []
        for row, (_, invert) in enumerate(chats):
            p = p_yes_from_generation(gen.sequences[row, n_in:].tolist(), steps[row], self.yes, self.no)
            out.append(1.0 - p if invert and not math.isnan(p) else p)
        return out


# --- specialists -----------------------------------------------------------------------------------

class HHEMScorer(Scorer):
    tasks = frozenset({"grounding"})

    def __init__(self, device: str = "cuda", batch_size: int = 8):
        from transformers import AutoModelForSequenceClassification

        self.model = AutoModelForSequenceClassification.from_pretrained(HHEM, trust_remote_code=True).to(device).eval()
        self.batch_size = batch_size
        self.name = "HHEM-2.1-open"

    def _batch(self, batch: list[Item]) -> list[float]:
        return [float(p) for p in self.model.predict([(it.context, it.prompt) for it in batch]).float().cpu().tolist()]


LETTUCE_QA = Template(
    "Briefly answer the following question:\n${question}\n"
    "Bear in mind that your response should be strictly based on the following ${num_passages} passages:\n${context}\n"
    'In case the passages do not contain the necessary information to answer the question, please reply with: '
    '"Unable to answer based on given passages."\noutput:'
)


def lettuce_prompt(context: str, question: str) -> str:
    return LETTUCE_QA.substitute(question=question, num_passages=1, context=f"passage 1: {context}")


class LettuceDetectScorer(Scorer):
    tasks = frozenset({"grounding"})

    def __init__(self, device: str = "cuda", max_len: int = 8192):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(LETTUCE)
        self.model = AutoModelForTokenClassification.from_pretrained(LETTUCE).to(device).eval()
        self.device, self.max_len = device, max_len
        self.name = "LettuceDetect large (ModernBERT)"

    def _batch(self, batch: list[Item]) -> list[float]:  # one item per call (batch_size 1)
        import torch

        (it,) = batch
        enc = self.tok(lettuce_prompt(it.context, it.question), it.answer, truncation="only_first",
                       max_length=self.max_len, return_tensors="pt").to(self.device)
        n_answer = len(self.tok(it.answer, add_special_tokens=False)["input_ids"])
        total = enc["input_ids"].shape[1]
        start = total - n_answer - 1  # [CLS] prompt [SEP] answer [SEP]
        with torch.no_grad(), torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16,
                                             enabled=torch.device(self.device).type == "cuda"):
            logits = self.model(**enc).logits[0].float()
        hall = torch.softmax(logits, dim=-1)[start : total - 1, 1]
        return [1.0 - float(hall.max()) if hall.numel() else NAN]


class ToxicBertScorer(Scorer):
    tasks = frozenset({"answer_harm", "toxicity", "beavertails", "toxic_chat", "oai_moderation"})

    def __init__(self, device: str = "cuda", batch_size: int = 32):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(TOXIC_BERT)
        self.model = AutoModelForSequenceClassification.from_pretrained(TOXIC_BERT).to(device).eval()
        self.device, self.batch_size = device, batch_size
        self.name = "toxic-bert (Detoxify)"

    def _batch(self, batch: list[Item]) -> list[float]:
        import torch

        enc = self.tok([it.answer or it.state for it in batch], truncation=True, max_length=512, padding=True,
                       return_tensors="pt").to(self.device)
        with torch.no_grad():
            return [float(x) for x in torch.sigmoid(self.model(**enc).logits.float()).max(dim=-1).values.cpu().tolist()]


def load_scorer(name: str, device: str = "cuda", jevlite: str = DEFAULT_JEVLITE, llm: str = DEFAULT_LLM) -> Scorer:
    if name == "jevlite":
        return JevLiteScorer(jevlite, device)
    if name == "nli":
        return NLIScorer(device)
    if name == "llm_judge":
        return LLMJudgeScorer(llm, device)
    if name == "granite_guardian":
        return GraniteGuardianScorer(device=device)
    if name == "hhem":
        return HHEMScorer(device)
    if name == "lettucedetect":
        return LettuceDetectScorer(device)
    if name == "toxic_bert":
        return ToxicBertScorer(device)
    raise ValueError(f"unknown scorer {name!r}, expected one of {SCORERS}")


__all__ = ["SCORERS", "Scorer", "load_scorer", "lettuce_prompt", "p_yes", "p_yes_from_generation", "single_token_ids",
           "split_on_oom"]
