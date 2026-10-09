"""Converters from public datasets to the unified format (TODO Phase 3, ``DATASETS.md`` section 0).

FEVER, HoVer, FEVEROUS and Natural Questions download large Wikipedia dumps (~75 GB on disk at the
peak, FEVEROUS alone 64 GB): ``all`` includes them, so convert them where the disk allows it
(``notebooks/kaggle/05_convert_wiki_sources.ipynb``).
"""

from __future__ import annotations

from .base import Converter
from .classification import BANKING77, CLINC150, MASSIVE, DBpedia14, GoEmotions
from .factcheck import FEVER, FEVEROUS, ContractNLI, HoVer, TabFact, VitaminC
from .mcqa import ARC, CommonsenseQA, CosmosQA, OpenBookQA, SocialIQa
from .nli import PAWS, QNLI, SNLI, WANLI, MultiNLI
from .qa import BoolQ, ClapNQ, NaturalQuestions, SQuAD2
from .ratings import HelpSteer2, HelpSteer3, SummarizeFromFeedback

CONVERTERS: dict[str, type[Converter]] = {
    c.name: c
    for c in (
        # Bool / Choice-3
        FEVER, VitaminC, HoVer, FEVEROUS, TabFact, MultiNLI, WANLI, SNLI, QNLI, PAWS, BoolQ, SQuAD2, NaturalQuestions, ClapNQ,
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
