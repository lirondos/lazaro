import re
from typing import Generator, Match, Optional, Pattern, TextIO, Tuple

from attr import attrs

from nerpy import Document, DocumentBuilder, EntityType, Mention, MentionType, Token


def split_keeping_delims(
    string: str, delim_pattern: Pattern
) -> Generator[Tuple[str, bool], None, None]:
    last_end = 0
    for match in delim_pattern.finditer(string):
        # Yield up to this match if needed
        start, end = match.span()
        if start > last_end:
            yield string[last_end:start], False

        # Yield the match itself if non-empty
        if start != end:
            yield match.group(0), True

        # Update end
        last_end = end

    # Yield whatever remains
    if last_end != len(string):
        yield string[last_end:], False


@attrs(frozen=True)
class OntoNotesIngester:
    _DOC_START = "<DOC"
    _DOC_END = "</DOC>"
    # Ignore anything before the first @
    _PATTERN_DOCNO = re.compile(r'<DOC DOCNO=".+?@(.+?)">')
    _PATTERN_NEWLINE = re.compile("\n")
    _PATTERN_SPACE = re.compile(" ")
    # We use * to help catch malformed empty tags
    _PATTERN_SPACE_OR_ENAMEX = re.compile(" |<ENAMEX.*?>.*?</ENAMEX>")
    _PATTERN_ENAMEX = re.compile(
        r'<ENAMEX TYPE="(?P<type>.+?)"(?: S_OFF="(?P<s_off>\d+)")?(?: E_OFF="(?P<e_off>\d+)")?>'
        "(?P<name>.+?)"
        "</ENAMEX>$"
    )
    _PATTERN_PUNC_REPLACEMENT = re.compile(r"-[A-Z]{3}-")

    _PUNC_TOKEN_MAP = {
        "-LCB-": "{",
        "-RCB-": "}",
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LAB-": "<",
        "-RAB-": ">",
        "-AMP-": "&",
    }

    @classmethod
    def _token_text(cls, token: str) -> str:
        return cls._PATTERN_PUNC_REPLACEMENT.sub(cls._replace_punc, token)

    @classmethod
    def _replace_punc(cls, match: Match) -> str:
        return cls._PUNC_TOKEN_MAP[match.group(0)]

    def ingest(self, source: TextIO, document_id: Optional[str] = None) -> Document:
        # Extract doc id
        try:
            doc_header = next(source)
        except StopIteration:
            raise ValueError("Empty file")

        if doc_header.startswith(self._DOC_START):
            if document_id is None:
                match = self._PATTERN_DOCNO.match(doc_header)
                if not match:
                    raise ValueError("Could not extract docid from document")
                document_id = match.group(1).replace("@", "_")
        else:
            raise ValueError("No opening DOC tag")

        # Initialize builder
        builder = DocumentBuilder(document_id)

        sentence_index = -1
        for source_sentence in source:
            source_sentence = source_sentence.strip()
            if not source_sentence:
                # Skip empty lines
                continue

            if source_sentence.startswith(self._DOC_END):
                break

            # Create tokens and mentions
            sentence_index += 1
            tokens = []
            token_idx = -1

            for token, is_token_delim in split_keeping_delims(
                source_sentence, self._PATTERN_SPACE_OR_ENAMEX
            ):
                if is_token_delim:
                    if token != " ":
                        # Should match ENAMEX
                        match = self._PATTERN_ENAMEX.match(token)
                        if not match:
                            raise ValueError("Could not match ENAMEX: " + repr(token))

                        # Create the tokens
                        name_match = match.group("name")
                        name_tokens = []
                        for name, is_name_delim in split_keeping_delims(
                            name_match, self._PATTERN_SPACE
                        ):
                            if not is_name_delim:
                                token_idx += 1
                                token_text = self._token_text(name)
                                name_tokens.append(Token(token_text, token_idx))

                        # Add the name tokens
                        tokens.extend(name_tokens)

                        if not name_tokens:
                            raise ValueError(
                                "No tokens created for ENAMEX: " + repr(token)
                            )

                        entity_type = EntityType(match.group("type"))
                        mention = Mention(
                            sentence_index,
                            name_tokens[0].index,
                            name_tokens[-1].index + 1,
                            MentionType("name"),
                            entity_type,
                        )
                        builder.add_mention(mention)
                else:
                    # Normal token, compute span and offsets
                    token_idx += 1
                    token_text = self._token_text(token)
                    tokens.append(Token(token_text, token_idx))

            # Add sentence to document
            builder.create_sentence(tokens)

        document = builder.build()

        return document
