from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

from attr import Attribute, attrib, attrs, converters, evolve
from attr.validators import instance_of
from frozendict import frozendict

# Make a str instance validator for convenience
# pylint: disable=invalid-name
_instance_of_str = instance_of(str)


def _validator_nonempty_str(inst: Any, attr: Attribute, value: Any) -> None:
    _instance_of_str(inst, attr, value)

    if not value or not isinstance(value, str):
        raise ValueError("Empty or non-string value: {!r}".format(value))


def _validator_nonnegative(_inst: Any, _attr: Attribute, value: Any) -> None:
    if value < 0:
        raise ValueError("Negative value: {}".format(value))


@attrs(frozen=True, slots=True)
class MentionType:
    type: str = attrib(validator=_validator_nonempty_str)


def _convert_entity_types(types: Union[str, Sequence[str]]) -> Tuple[str, ...]:
    if isinstance(types, str):
        # Return as singleton tuple
        return (types,)
    else:
        return tuple(types)


@attrs(frozen=True, slots=True)
class EntityType(Sequence[str]):
    types: Tuple[str, ...] = attrib(converter=_convert_entity_types)

    # PyCharm doesn't understand .validator
    # noinspection PyUnresolvedReferences
    @types.validator
    def _validate_types(self, attr: Attribute, value: Any) -> None:
        if not value:
            raise ValueError("Empty value: {!r}".format(value))

        for item in value:
            _validator_nonempty_str(self, attr, item)

    @overload
    def __getitem__(self, index: int) -> str:
        raise NotImplementedError

    @overload
    def __getitem__(self, index: slice) -> Tuple[str, ...]:
        raise NotImplementedError

    def __getitem__(self, i: Union[int, slice]) -> Union[str, Tuple[str, ...]]:
        return self.types[i]

    def __len__(self) -> int:
        return len(self.types)

    def __str__(self) -> str:
        return ".".join(self.types)


@attrs(frozen=True, slots=True)
class Token(Sequence[str]):
    _POS_TAG = "pos"
    _CHUNK_TAG = "chunk"
    _LEMMAS = "lemmas"

    text: str = attrib(validator=_validator_nonempty_str)
    # Mypy gives a false positive on this validator
    index: int = attrib(validator=_validator_nonnegative, eq=False)  # type: ignore
    # Mypy attrs plugin raises false positives on converters and validators
    properties: Mapping[str, Any] = attrib(
        converter=converters.optional(frozendict),  # type: ignore
        default=None,
    )

    @property
    def pos_tag(self) -> Optional[str]:
        return self.properties.get(self._POS_TAG) if self.properties else None

    @property
    def chunk_tag(self) -> Optional[str]:
        return self.properties.get(self._CHUNK_TAG) if self.properties else None

    @property
    def lemmas(self) -> Optional[Tuple[str, ...]]:
        return self.properties.get(self._LEMMAS) if self.properties else None

    @classmethod
    def create(
        cls,
        text: str,
        index: int,
        *,
        pos_tag: Optional[str] = None,
        chunk_tag: Optional[str] = None,
        lemmas: Optional[Sequence[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> "Token":
        if properties is None and (
            pos_tag is not None or chunk_tag is not None or lemmas is not None
        ):
            properties = {}

        if pos_tag is not None:
            properties[cls._POS_TAG] = pos_tag
        if chunk_tag is not None:
            properties[cls._CHUNK_TAG] = chunk_tag
        if lemmas is not None:
            properties[cls._LEMMAS] = tuple(lemmas)

        return Token(text, index, properties)

    @overload
    def __getitem__(self, index: int) -> str:
        raise NotImplementedError

    @overload
    def __getitem__(self, index: slice) -> str:
        raise NotImplementedError

    def __getitem__(self, i: Union[int, slice]) -> str:
        return self.text[i]

    def __iter__(self) -> Iterator[str]:
        return iter(self.text)

    def __len__(self) -> int:
        return len(self.text)


@attrs(frozen=True, slots=True)
class Sentence(Sequence[Token]):
    tokens: Tuple[Token, ...] = attrib(converter=tuple)
    # Mypy gives a false positive on this validator
    index: int = attrib(validator=_validator_nonnegative, eq=False)  # type: ignore

    @staticmethod
    def from_tokens(tokens: Sequence[Token], sentence_index: int) -> "Sentence":
        # Check token indices
        for i, token in enumerate(tokens):
            if token.index != i:
                raise ValueError(
                    "Expected index {} for token, got index {}".format(i, token.index)
                )

        # Mypy does not recognize tokens as an Iterable
        return Sentence(tokens, sentence_index)  # type: ignore

    @overload
    def __getitem__(self, index: int) -> Token:
        raise NotImplementedError

    @overload
    def __getitem__(self, index: slice) -> Tuple[Token, ...]:
        raise NotImplementedError

    def __getitem__(self, i: Union[int, slice]) -> Union[Token, Tuple[Token, ...]]:
        return self.tokens[i]

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)


@attrs(frozen=True, slots=True)
class Mention:
    sentence_index: int = attrib(validator=_validator_nonnegative)
    start: int = attrib(validator=_validator_nonnegative)
    end: int = attrib(validator=_validator_nonnegative)
    mention_type: MentionType = attrib(validator=instance_of(MentionType))
    entity_type: EntityType = attrib(validator=instance_of(EntityType))

    @end.validator
    def _validate_end(self, _attr: Attribute, value: int) -> None:
        if value <= self.start:
            raise ValueError("End token index must be greater than start")

    @staticmethod
    def create(
        sentence: Sentence,
        tokens: Sequence[Token],
        mention_type: MentionType,
        entity_type: EntityType,
    ) -> "Mention":
        sentence_idx = sentence.index

        token_idxs = [token.index for token in tokens]
        if not token_idxs:
            raise ValueError("Token sequence is empty")

        # Inclusive start index
        first_token_index = token_idxs[0]
        # Exclusive end index
        last_token_index = tokens[-1].index + 1

        if token_idxs != list(
            range(first_token_index, first_token_index + len(token_idxs))
        ):
            raise ValueError("Tokens are not in correct order: {}".format(token_idxs))

        return Mention(
            sentence_idx, first_token_index, last_token_index, mention_type, entity_type
        )

    def __len__(self) -> int:
        return self.end - self.start

    def tokens(self, document: "Document") -> Tuple[Token, ...]:
        return document.sentences[self.sentence_index].tokens[self.start : self.end]

    def tokenized_text(self, document: "Document") -> str:
        return " ".join([token.text for token in self.tokens(document)])


@attrs(frozen=True, slots=True)
class Document(Sequence[Sentence]):
    id: str = attrib(validator=_validator_nonempty_str)
    sentences: Tuple[Sentence, ...] = attrib(converter=tuple)
    # TODO: Make these sorted, coordinate with DocumentBuilder initialization
    mentions: Tuple[Mention, ...] = attrib(converter=tuple, default=())
    # TODO: Consider making this the primary mention representation so we don't store mentions twice
    _sentence_mentions: Tuple[Tuple[Mention, ...], ...] = attrib(init=False)

    @_sentence_mentions.default
    def _sentence_mentions_default(self) -> Tuple[Tuple[Mention, ...], ...]:
        sentence_mentions: List[List[Mention]] = [[] for _ in range(len(self.sentences))]
        # TODO: Sort mentions before iteration so that order is guaranteed
        for mention in self.mentions:
            sentence_mentions[mention.sentence_index].append(mention)
        return tuple(tuple(mentions) for mentions in sentence_mentions)

    @overload
    def __getitem__(self, index: int) -> Sentence:
        raise NotImplementedError

    @overload
    def __getitem__(self, index: slice) -> Tuple[Sentence, ...]:
        raise NotImplementedError

    def __getitem__(self, i: Union[int, slice]) -> Union[Sentence, Tuple[Sentence, ...]]:
        return self.sentences[i]

    def __iter__(self) -> Iterator[Sentence]:
        return iter(self.sentences)

    def __len__(self) -> int:
        return len(self.sentences)

    def sentences_with_mentions(self) -> Iterable[Tuple[Sentence, Tuple[Mention, ...]]]:
        for sentence, mentions in zip(self.sentences, self._sentence_mentions):
            yield sentence, mentions

    def mentions_for_sentence(self, sentence: Sentence):
        sentence_idx = sentence.index
        if self.sentences[sentence_idx] != sentence:
            raise ValueError("Sentence is not from this document")
        return self._sentence_mentions[sentence_idx]

    def copy_with_mentions(self, mentions: Iterable[Mention]) -> "Document":
        return evolve(self, mentions=mentions)

    def copy_without_mentions(self) -> "Document":
        return evolve(self, mentions=())

    def __str__(self):
        lines: List[str] = []
        for sentence in self:
            lines.append(" ".join(token.text for token in sentence))
        return "\n".join(lines)


@attrs(eq=False)
class DocumentBuilder:
    id: str = attrib(validator=_validator_nonempty_str)
    sentences: List[Sentence] = attrib(factory=list, init=False)
    # TODO: Should mentions be an ordered set?
    mentions: List[Mention] = attrib(factory=list, init=False)
    sentence_idx: int = attrib(default=0, init=False)

    def add_mention(self, mention: Mention) -> "DocumentBuilder":
        self.mentions.append(mention)
        return self

    def add_mentions(self, mentions: Iterable[Mention]) -> "DocumentBuilder":
        self.mentions.extend(mentions)
        return self

    def create_sentence(self, tokens: Sequence[Token]) -> Sentence:
        sentence = Sentence.from_tokens(tokens, self.sentence_idx)
        self.sentences.append(sentence)
        self.sentence_idx += 1
        return sentence

    def build(self) -> Document:
        # Mypy does not recognize tokens as an Iterable
        return Document(self.id, self.sentences, self.mentions)  # type: ignore
