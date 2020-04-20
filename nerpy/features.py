import re
from abc import ABCMeta, abstractmethod
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

# For Unicode properties
import regex
from attr import attrs

from nerpy.document import Document, Sentence, Token
from nerpy.embeddings import SqliteWordEmbeddings

# Sentinel for dict lookup
_NOTHING = object()

_RE_NUMERIC = re.compile(r"[\d.,]+$")
_RE_DIGIT = re.compile(r"\d")
# We use the regex package since we want to use Unicode properties
_RE_PUNC = regex.compile(r"\p{p}")

# pylint: disable=invalid-name
FeatureSink = MutableMapping[str, float]
ItemFeatures = Mapping[str, float]
SequenceFeatures = Sequence[ItemFeatures]
CorpusFeatures = Sequence[SequenceFeatures]
SequenceLabels = Sequence[str]
CorpusLabels = Sequence[SequenceLabels]


class FeatureExtractor(metaclass=ABCMeta):
    @abstractmethod
    def extract(self, token: Token, index: int, output: FeatureSink) -> None:
        raise NotImplementedError()


class WordEmbeddingFeatures(FeatureExtractor):

    FEATURE = "v"
    OOV = "OOV"

    def __init__(self, path: str, *, scale: float = 1.0, cache_size: int = 10000):
        self.scale = scale
        self.word_vectors = SqliteWordEmbeddings(path)
        self._feature_keys_cache: Dict[int, List[str]] = {}
        # Store normalized form or None to indicate no match
        self._word_casing: Dict[str, Optional[str]] = {}
        # Cache vectors
        # We look up directly from the vectors if there is no scaling, or add our own wrapper if there is scaling.
        self._embedding_cache = lru_cache(cache_size)(
            self.word_vectors.__getitem__
            if self.scale == 1.0
            else self._scaled_word_vector
        )

    def extract(self, token: Token, index: int, output: FeatureSink) -> None:
        text = token.text
        # Do nothing for punc
        if _is_punc(text):
            return

        # Optimization: avoid repeated lookups
        word_vectors = self.word_vectors

        # Optimization: Using get and try/except result in indistinguishable performance
        norm_text = self._word_casing.get(text, _NOTHING)
        if norm_text is _NOTHING:
            # Haven't determined casing yet
            if text in word_vectors:
                # Keep normal casing
                norm_text = text
            else:
                lower_text = text.lower()
                norm_text = lower_text if lower_text in word_vectors else None
            self._word_casing[text] = norm_text

        if norm_text is None:
            # No normalized form in vectors, thus OOV
            _add_feature_with_value(self.FEATURE, index, self.OOV, output)
            return

        # Since we will only miss once ever for each relative index features are generated over,
        # use an exception rather than a conditional.
        try:
            feature_keys = self._feature_keys_cache[index]
        except KeyError:
            feature_keys = [
                f"{self.FEATURE}[{index}]={i}" for i in range(word_vectors.dim)
            ]
            self._feature_keys_cache[index] = feature_keys

        # Optimization: Update over zip is much faster than any kind of comprehension
        # We suppress type warnings because we know norm_text must be a str at this point
        output.update(zip(feature_keys, self._embedding_cache(norm_text)))  # type: ignore

    def _scaled_word_vector(self, word: str) -> Sequence[float]:
        vec = self.word_vectors[word]
        # We cannot use in-place multiply because vec is read-only
        # Ignore type warning because we know this is an ndarray even though the formal return type is
        # Sequence[float]. Things are this way because it's hard to type annotate an ndarray return.
        return vec * self.scale  # type: ignore


class BrownClusterFeatures(FeatureExtractor):

    FEATURE = "bc"
    OOV = "OOV"

    def __init__(
        self,
        clusters_path: str,
        *,
        use_full_paths: bool = False,
        use_prefixes: bool = False,
        prefixes: Optional[Sequence[int]] = None,
    ) -> None:
        if not (use_full_paths or use_prefixes):
            raise ValueError("Either use_full_paths or use_prefixes must be set to True")
        if use_full_paths and use_prefixes and not prefixes:
            raise ValueError(
                "Cannot use full paths and all possible prefixes as it would create redundant features. "
                "Disable use_full_paths or specify values for the prefixes."
            )
        if prefixes and not use_prefixes:
            raise ValueError("Cannot specify prefixes without specifying use_prefixes")
        if prefixes is not None and not prefixes:
            raise ValueError(
                "Cannot specify empty sequence of prefixes. Use None instead."
            )
        if prefixes:
            for prefix in prefixes:
                if not isinstance(prefix, int) or prefix <= 0:
                    raise ValueError("Prefixes must be positive integers")

        self.use_full_paths = use_full_paths
        self.use_prefixes = use_prefixes
        self.prefixes = prefixes

        self.paths: Dict[str, str] = {}
        with open(clusters_path, encoding="utf8") as cluster_file:
            for idx, line in enumerate(cluster_file):
                splits = line.split()
                if len(splits) != 3:
                    raise ValueError(
                        f"Invalid format on line {idx + 1} of file {clusters_path}: {repr(line)}"
                    )
                path, token, _ = splits
                self.paths[token] = path

    def extract(self, token: Token, index: int, output: FeatureSink) -> None:
        try:
            path = self.paths[token.text]
        except KeyError:
            _add_feature_with_value(self.FEATURE, index, self.OOV, output)
            return

        if self.use_full_paths:
            _add_feature_with_value(self.FEATURE, index, path, output)

        if self.use_prefixes:
            max_idx = len(path) + 1
            prefix_idxs: Iterable[int]
            if self.prefixes:
                prefix_idxs = [idx for idx in self.prefixes if idx < max_idx]
            else:
                prefix_idxs = range(1, max_idx)
            for idx in prefix_idxs:
                _add_feature_with_value(self.FEATURE, index, path[:idx], output)


class TokenIdentity(FeatureExtractor):

    FEATURE = "tkn"

    def __init__(self, *, lowercase: bool = False):
        self.lowercase = lowercase

    def extract(self, token: Token, index: int, output: FeatureSink) -> None:
        _add_feature_with_value(
            self.FEATURE,
            index,
            token.text if not self.lowercase else token.text.lower(),
            output,
        )


class IsCapitalized(FeatureExtractor):

    FEATURE = "cap"

    def extract(self, token: Token, index: int, output: FeatureSink) -> None:
        value = token.text[0].isupper()
        if value:
            _add_feature_with_value(self.FEATURE, index, value, output)


class IsPunc(FeatureExtractor):

    FEATURE = "punc"

    def extract(self, token: Token, index: int, output: FeatureSink) -> None:
        value = _is_punc(token.text)
        if value:
            _add_feature_with_value(self.FEATURE, index, value, output)


class AllCaps(FeatureExtractor):

    FEATURE = "all_caps"

    def extract(self, token: Token, index: int, output: FeatureSink) -> None:
        value = token.text.isupper()
        if value:
            _add_feature_with_value(self.FEATURE, index, value, output)


class AllNumeric(FeatureExtractor):

    FEATURE = "all_num"

    def extract(self, token: Token, index: int, output: FeatureSink) -> None:
        value = bool(_RE_DIGIT.search(token.text) and _RE_NUMERIC.match(token.text))
        if value:
            _add_feature_with_value(self.FEATURE, index, value, output)


class ContainsNumber(FeatureExtractor):

    FEATURE = "cntns_num"

    def extract(self, token: Token, index: int, output: FeatureSink) -> None:
        value = bool(_RE_DIGIT.search(token.text))
        if value:
            _add_feature_with_value(self.FEATURE, index, value, output)


class LengthValue(FeatureExtractor):

    FEATURE = "len_val"

    def extract(self, token: Token, index: int, output: FeatureSink) -> None:
        _add_feature_with_value(self.FEATURE, index, len(token.text), output)


class LengthWeight(FeatureExtractor):

    FEATURE = "len_weight"

    def extract(self, token: Token, index: int, output: FeatureSink) -> None:
        _add_feature_without_value(self.FEATURE, index, output, len(token.text))


class POS(FeatureExtractor):

    FEATURE = "pos"

    def extract(self, token: Token, index: int, output: FeatureSink) -> None:
        if token.pos_tag:
            _add_feature_with_value(self.FEATURE, index, token.pos_tag, output)


class Prefix(FeatureExtractor):

    FEATURE = "pfx"

    def __init__(self, min_length: int, max_length: int):
        self.min_length = min_length
        self.max_length = max_length

    def extract(self, token: Token, index: int, output: FeatureSink) -> None:
        token_text = token.text
        for i in range(self.min_length, self.max_length + 1):
            if i > len(token_text):
                break
            _add_feature_with_value(self.FEATURE, index, token_text[:i], output)


class Suffix(FeatureExtractor):

    FEATURE = "sfx"

    def __init__(self, min_length: int, max_length: int):
        self.min_length = min_length
        self.max_length = max_length

    def extract(self, token: Token, index: int, output: FeatureSink) -> None:
        token_text = token.text
        for i in range(self.min_length, self.max_length + 1):
            if i > len(token_text):
                break
            _add_feature_with_value(self.FEATURE, index, token_text[-i:], output)


class WordShape(FeatureExtractor):

    FEATURE = "shape"

    def extract(self, token: Token, index: int, output: FeatureSink) -> None:
        chars = []
        # Because Python doesn't support full unicode properties, we can't easily do a regex like
        # "match lowercase letters". Instead we just go character by character
        for char in token.text:
            if char.isalpha():
                if char.isupper():
                    chars.append("A")
                else:
                    chars.append("a")
            elif char.isdigit():
                chars.append("0")
            else:
                chars.append(char)
        _add_feature_with_value(self.FEATURE, index, "".join(chars), output)


class SentenceFeatureExtractor:

    BIAS = "b"
    FEATURE_CLASSES = {
        "token_identity": TokenIdentity,
        "is_capitalized": IsCapitalized,
        "is_punc": IsPunc,
        "all_caps": AllCaps,
        "all_numeric": AllNumeric,
        "contains_number": ContainsNumber,
        "length_value": LengthValue,
        "length_weight": LengthWeight,
        "pos": POS,
        "word_shape": WordShape,
        "suffix": Suffix,
        "word_vectors": WordEmbeddingFeatures,
        "brown_clusters": BrownClusterFeatures,
        "prefix": Prefix,
    }

    def __init__(self, feature_params: Mapping):

        self.window_features: dict = {}

        for feature_set in feature_params:
            window = feature_params[feature_set]["window"]
            window_features = []
            for feature in feature_params[feature_set]:
                if feature == "window":
                    continue

                if feature not in self.FEATURE_CLASSES:
                    raise ValueError(f"Invalid feature name: {feature}")
                feature_class = self.FEATURE_CLASSES[feature]
                feature_kwargs = feature_params[feature_set][feature]
                window_features.append(feature_class(**feature_kwargs))

            for position in window:
                if position in window:
                    if position not in self.window_features:
                        self.window_features[position] = []
                    self.window_features[position].extend(window_features)

    def extract(self, sentence: Sentence, _doc: Document) -> SequenceFeatures:
        sentence_features: List[Mapping[str, float]] = []
        tokens = sentence.tokens
        max_i = len(tokens) - 1

        for idx, _ in enumerate(tokens):
            token_features = {self.BIAS: 1.0}

            for position in self.window_features:
                position_index = idx + position
                position_feature_extractors = self.window_features[position]
                if 0 <= position_index <= max_i:
                    position_token = tokens[position_index]
                    for extractor in position_feature_extractors:
                        extractor.extract(position_token, position, token_features)

            sentence_features.append(token_features)

        return sentence_features


@attrs(auto_attribs=True, frozen=True)
class ExtractedFeatures:
    extractor: SentenceFeatureExtractor
    features: CorpusFeatures
    labels: Optional[CorpusLabels]


def _add_feature_with_value(
    label: str, index: int, value: Any, output: FeatureSink, weight: float = 1.0
) -> None:
    output[f"{label}[{index}]={value}"] = weight


def _add_feature_without_value(
    label: str, index: int, output: FeatureSink, weight: float = 1.0
) -> None:
    output[f"{label}[{index}]"] = weight


def _is_punc(s: str) -> bool:
    return bool(_RE_PUNC.match(s))
