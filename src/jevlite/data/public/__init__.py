"""Converters from public datasets to the unified format (TODO Phase 3, ``DATASETS.md`` section 0).

Not converted yet: FEVER, HoVer and FEVEROUS (their evidence text needs Wikipedia dumps) and
Natural Questions (the full release is ~45 GB; SQuAD 2.0 and ClapNQ cover answerability for now).
"""

from __future__ import annotations

from .base import Converter
from .classification import BANKING77, CLINC150, MASSIVE, DBpedia14, GoEmotions
from .factcheck import ContractNLI, TabFact, VitaminC
from .mcqa import ARC, CommonsenseQA, CosmosQA, OpenBookQA, SocialIQa
from .nli import PAWS, QNLI, SNLI, WANLI, MultiNLI
from .qa import BoolQ, ClapNQ, SQuAD2
from .ratings import HelpSteer2, HelpSteer3, SummarizeFromFeedback

CONVERTERS: dict[str, type[Converter]] = {
    c.name: c
    for c in (
        # Bool / Choice-3
        VitaminC, TabFact, MultiNLI, WANLI, SNLI, QNLI, PAWS, BoolQ, SQuAD2, ClapNQ,
        # Choice
        ARC, OpenBookQA, CommonsenseQA, CosmosQA, SocialIQa, CLINC150, MASSIVE, DBpedia14, GoEmotions,
        # Score and preferences
        HelpSteer2, HelpSteer3, SummarizeFromFeedback,
        # test_ood only
        ContractNLI, BANKING77,
    )
}


def get_converter(name: str) -> Converter:
    if name not in CONVERTERS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(CONVERTERS)}")
    return CONVERTERS[name]()
