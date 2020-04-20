from typing import Iterable, List, Optional, TextIO, Tuple

from attr import attrib, attrs

from nerpy import Document, DocumentBuilder, MentionEncoder, Token

DOCSTART = "-DOCSTART-"


@attrs(frozen=True)
class CoNLLIngester:
    mention_encoder: MentionEncoder = attrib()
    ignore_comments: bool = attrib(default=False, kw_only=True)

    def ingest(self, document_id: str, source: TextIO) -> List[Document]:
        documents = []
        document_counter = 1
        builder = DocumentBuilder(document_id + "_" + str(document_counter))

        for source_sentence in self._parse_file(
            source, ignore_comments=self.ignore_comments
        ):
            sentence_tokens: List[Token] = []
            sentence_labels: List[str] = []

            if source_sentence[0].is_docstart:
                # We should only receive DOCSTART in a sentence by itself
                assert (
                    len(source_sentence) == 1
                ), "Received -DOCSTART- as part of a sentence"

                # End current document and start a new one
                # We skip this if the builder is empty, which will happen for the very
                # first document in the corpus (as there is no previous document to end).
                if builder.sentences:
                    document = builder.build()
                    documents.append(document)
                    document_counter += 1
                    builder = DocumentBuilder(document_id + "_" + str(document_counter))
                continue

            # Create mentions from tokens in sentence
            for idx, token in enumerate(source_sentence):
                new_token = Token.create(
                    token.text,
                    idx,
                    pos_tag=token.pos_tag,
                    chunk_tag=token.chunk_tag,
                    lemmas=token.lemmas,
                )
                sentence_tokens.append(new_token)
                sentence_labels.append(token.ne_tag)

            sentence = builder.create_sentence(sentence_tokens)

            mentions = self.mention_encoder.decode_mentions(sentence, sentence_labels)
            builder.add_mentions(mentions)

        document = builder.build()
        documents.append(document)

        return documents

    @classmethod
    def _parse_file(
        cls, input_file: TextIO, *, ignore_comments: bool = False
    ) -> Iterable[Tuple["_CoNLLToken", ...]]:
        sentence: list = []
        line_num = 0
        for line in input_file:
            line_num += 1
            line = line.strip()

            if ignore_comments and line.startswith("#"):
                continue

            if not line:
                # Clear out sentence if there's anything in it
                if sentence:
                    yield tuple(sentence)
                    sentence = []
                # Always skip empty lines
                continue

            token = cls._CoNLLToken.from_line(line)
            # Skip document starts, but ensure sentence is empty when we reach them
            if token.is_docstart:
                if sentence:
                    raise ValueError(
                        "Encountered DOCSTART at line {} while still in sentence".format(
                            line_num
                        )
                    )
                else:
                    # Yield it by itself
                    yield (token,)
            else:
                sentence.append(token)

        # Finish the last sentence if needed
        if sentence:
            yield tuple(sentence)

    @attrs(frozen=True)
    class _CoNLLToken:
        text: str = attrib()
        pos_tag: Optional[str] = attrib()
        lemmas: Optional[Tuple[str, ...]] = attrib()
        chunk_tag: Optional[str] = attrib()
        ne_tag: str = attrib()
        is_docstart: bool = attrib()

        @classmethod
        def from_line(cls, line: str) -> "CoNLLIngester._CoNLLToken":
            splits = line.split()
            text = splits[0]
            ne_tag = splits[-1]

            if len(splits) == 5:
                # Assume has lemmas like 2002 German data
                lemmas = tuple(splits[1].split("|"))
                pos_tag = splits[2]
                chunk_tag = splits[3]
            else:
                lemmas = None
                # Other tags will be POS if available, then chunk if available
                pos_tag = splits[1] if len(splits) > 2 else None
                chunk_tag = splits[2] if len(splits) > 3 else None

            is_docstart = text == DOCSTART
            return cls(text, pos_tag, lemmas, chunk_tag, ne_tag, is_docstart)
