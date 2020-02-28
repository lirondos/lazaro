import re
from abc import ABC, abstractmethod
from typing import Sequence, Dict, List, NamedTuple, Tuple, Counter

import regex
from spacy.tokens import Token

UPPERCASE_RE = regex.compile(r"[\p{Lu}\p{Lt}]")
LOWERCASE_RE = regex.compile(r"\p{Ll}")
DIGIT_RE = re.compile(r"\d")

PUNC_REPEAT_RE = regex.compile(r"\p{P}+")


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        raise NotImplementedError


class EntityEncoder(ABC):
    @abstractmethod
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        raise NotImplementedError


class PRF1(NamedTuple):
    precision: float
    recall: float
    f1: float


class ScoringEntity(NamedTuple):
    tokens: Tuple[str, ...]
    entity_type: str


class ScoringCounts(NamedTuple):
    true_positives: Counter[ScoringEntity]
    false_positives: Counter[ScoringEntity]
    false_negatives: Counter[ScoringEntity]
