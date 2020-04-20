from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Sequence, Type

from attr import attrs

from nerpy.document import EntityType, Mention, MentionType, Sentence

LABEL_DELIM = "-"


class MentionEncoder(metaclass=ABCMeta):
    """Convert mentions to and from a sequence of label strings."""

    @abstractmethod
    def encode_mentions(
        self, sentence: Sentence, mentions: Sequence[Mention]
    ) -> Sequence[str]:
        raise NotImplementedError()

    @abstractmethod
    def decode_mentions(
        self, sentence: Sentence, labels: Sequence[str]
    ) -> Sequence[Mention]:
        raise NotImplementedError()


@attrs(auto_attribs=True)
class _EncoderToken:
    entity_type: Optional[EntityType] = None
    first_token: bool = False
    last_token: bool = False
    first_token_after_same_type_mention = False


class AbstractMentionEncoder(MentionEncoder, metaclass=ABCMeta):
    BEGIN_PREFIX = "B"
    INSIDE_PREFIX = "I"
    LAST_PREFIX = "L"
    UNIT_PREFIX = "U"
    OUTSIDE = "O"

    @abstractmethod
    def _encode_token(self, token: _EncoderToken) -> str:
        """Encode the tokens using the mention encoding."""

    def encode_mentions(
        self, sentence: Sentence, mentions: Sequence[Mention]
    ) -> Sequence[str]:
        encoder_tokens = [_EncoderToken() for _ in sentence.tokens]
        for mention in mentions:
            start = mention.start
            end = mention.end
            entity_type = mention.entity_type

            # Set entity type on all tokens
            for idx in range(start, end):
                token = encoder_tokens[idx]
                assert token.entity_type is None, (
                    "Token at index {} already has entity type {}, "
                    "refusing to overwrite it with entity type {}".format(
                        idx, token.entity_type, entity_type
                    )
                )
                token.entity_type = entity_type

            # Set first and last. Note that both will be True for single-token mentions.
            encoder_tokens[start].first_token = True
            # Exclusive end offset, so subtract one
            encoder_tokens[end - 1].last_token = True

        # Do a final pass to mark special tokens for IOB encoding
        prev_entity_type = None
        for token in encoder_tokens:
            if token.entity_type == prev_entity_type and token.first_token:
                token.first_token_after_same_type_mention = True

            prev_entity_type = token.entity_type

        return tuple(self._encode_token(token) for token in encoder_tokens)

    def decode_mentions(
        self, sentence: Sentence, labels: Sequence[str]
    ) -> Sequence[Mention]:
        if len(sentence) != len(labels):
            raise ValueError(
                f"Sentence is of length {len(sentence)} but {len(labels)} labels provided"
            )

        mentions = []
        entity_type: Optional[EntityType] = None
        mention_start: Optional[int] = None

        sentence_idx = sentence.index
        for idx, _ in enumerate(sentence.tokens):
            label = labels[idx]

            if label.startswith(self.BEGIN_PREFIX):
                # Clear out any started mention
                if entity_type:
                    assert mention_start is not None
                    mention = Mention(
                        sentence_idx, mention_start, idx, MentionType("name"), entity_type
                    )
                    mentions.append(mention)

                # Start new mention
                entity_type = _extract_entity_type(label)
                mention_start = idx
            elif label.startswith(self.LAST_PREFIX):
                # Clear out any started mention if the type is different
                if entity_type and entity_type != _extract_entity_type(label):
                    assert mention_start is not None
                    mention = Mention(
                        sentence_idx, mention_start, idx, MentionType("name"), entity_type
                    )
                    mentions.append(mention)
                    entity_type = None
                    mention_start = None

                # Fill in entity type if needed
                # This can occur if a last is predicted without a preceding begin or inside
                if not entity_type:
                    mention_start = idx
                    entity_type = _extract_entity_type(label)

                assert mention_start is not None
                mention = Mention(
                    sentence_idx, mention_start, idx + 1, MentionType("name"), entity_type
                )
                mentions.append(mention)
                mention_start = None
                entity_type = None
            elif label.startswith(self.UNIT_PREFIX):
                # Clear out any previously started mention
                if entity_type:
                    assert mention_start is not None
                    mention = Mention(
                        sentence_idx, mention_start, idx, MentionType("name"), entity_type
                    )
                    mentions.append(mention)

                # Unit mention
                entity_type = _extract_entity_type(label)
                mention = Mention(
                    sentence_idx, idx, idx + 1, MentionType("name"), entity_type
                )
                mentions.append(mention)
                mention_start = None
                entity_type = None
            elif label.startswith(self.INSIDE_PREFIX):
                # Clear out any started mention if the type is different
                if entity_type and entity_type != _extract_entity_type(label):
                    assert mention_start is not None
                    mention = Mention(
                        sentence_idx, mention_start, idx, MentionType("name"), entity_type
                    )
                    mentions.append(mention)
                    entity_type = None
                    mention_start = None

                # Start mention if needed
                # This can occur if a last is predicted without a preceding begin or inside
                if mention_start is None:
                    mention_start = idx
                    entity_type = _extract_entity_type(label)
            elif label == self.OUTSIDE:
                # Close any non-ended mention
                # This will happen if a mention doesn't end with last
                if mention_start is not None:
                    mention = Mention(
                        sentence_idx,
                        mention_start,
                        idx,  # Previous token must be the last one, and index is exclusive
                        MentionType("name"),
                        entity_type,
                    )
                    mentions.append(mention)
                    mention_start = None
                    entity_type = None
            else:
                raise ValueError(f"Unknown label: {repr(label)}")

        # Close any dangling mention at end of sentence
        if mention_start is not None:
            mention = Mention(
                sentence_idx,
                mention_start,
                len(sentence.tokens),  # Final index of sentence
                MentionType("name"),
                entity_type,
            )
            mentions.append(mention)

        return mentions


class BILOU(AbstractMentionEncoder):
    def _encode_token(self, token: _EncoderToken) -> str:
        if token.entity_type:
            if token.first_token:
                if token.last_token:
                    # Both first and last means unit
                    prefix = self.UNIT_PREFIX
                else:
                    prefix = self.BEGIN_PREFIX
            elif token.last_token:
                prefix = self.LAST_PREFIX
            else:
                prefix = self.INSIDE_PREFIX
            return _create_label(prefix, token.entity_type)
        else:
            return self.OUTSIDE


class IOBES(BILOU):
    LAST_PREFIX = "E"
    UNIT_PREFIX = "S"


class BMES(IOBES):
    INSIDE_PREFIX = "M"


class BIOU(AbstractMentionEncoder):
    def _encode_token(self, token: _EncoderToken) -> str:
        if token.entity_type:
            if token.first_token:
                if token.last_token:
                    # Both first and last means unit
                    prefix = self.UNIT_PREFIX
                else:
                    prefix = self.BEGIN_PREFIX
            else:
                prefix = self.INSIDE_PREFIX
            return _create_label(prefix, token.entity_type)
        else:
            return self.OUTSIDE


class IO(AbstractMentionEncoder):
    def _encode_token(self, token: _EncoderToken) -> str:
        if token.entity_type:
            return _create_label(self.INSIDE_PREFIX, token.entity_type)
        else:
            return self.OUTSIDE


class IOB(AbstractMentionEncoder):
    def _encode_token(self, token: _EncoderToken) -> str:
        if token.entity_type:
            if token.first_token_after_same_type_mention:
                prefix = self.BEGIN_PREFIX
            else:
                prefix = self.INSIDE_PREFIX
            return _create_label(prefix, token.entity_type)
        else:
            return self.OUTSIDE


class BIO(AbstractMentionEncoder):
    def _encode_token(self, token: _EncoderToken) -> str:
        if token.entity_type:
            if token.first_token:
                prefix = self.BEGIN_PREFIX
            else:
                prefix = self.INSIDE_PREFIX
            return _create_label(prefix, token.entity_type)
        else:
            return self.OUTSIDE


# Declared mid-file so it can refer to classes in file
_ENCODING_NAMES: Dict[str, Type[MentionEncoder]] = {
    "IOB": IOB,
    "IOB1": IOB,
    "BIO": BIO,
    "IOB2": BIO,
    "IO": IO,
    "BILOU": BILOU,
    "BIOU": BIOU,
    "BMES": BMES,
    "IOBES": IOBES,
    "BIOES": IOBES,
}
SUPPORTED_ENCODINGS: Sequence[str] = tuple(sorted(_ENCODING_NAMES))


def get_mention_encoder(name: str) -> Type[MentionEncoder]:
    name = name.upper()
    if name in _ENCODING_NAMES:
        return _ENCODING_NAMES[name]
    else:
        raise ValueError(f"Unknown encoder {repr(name)}")


def _extract_entity_type(label: str) -> EntityType:
    splits = label.split(LABEL_DELIM)
    if len(splits) == 2:
        return EntityType(splits[1])
    else:
        raise ValueError("Cannot parse label {!r}".format(label))


def _create_label(prefix: str, entity_type: EntityType) -> str:
    return prefix + LABEL_DELIM + entity_type.types[0]
