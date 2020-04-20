import time
from abc import ABCMeta, abstractmethod
from typing import Iterable, Sequence

from nerpy.document import Document, Mention
from nerpy.encoding import MentionEncoder
from nerpy.features import ExtractedFeatures, SentenceFeatureExtractor


class MentionAnnotator(metaclass=ABCMeta):
    @abstractmethod
    def mentions(self, doc: Document) -> Sequence[Mention]:
        raise NotImplementedError

    def add_mentions(self, doc: Document) -> Document:
        return doc.copy_with_mentions(self.mentions(doc))


class Trainable(metaclass=ABCMeta):
    @abstractmethod
    def train(self, docs: Iterable[Document], *args, **kwargs) -> None:
        raise NotImplementedError


class SequenceMentionAnnotator(MentionAnnotator, metaclass=ABCMeta):
    @property
    @abstractmethod
    def mention_encoder(self) -> MentionEncoder:
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_extractor(self) -> SentenceFeatureExtractor:
        raise NotImplementedError

    def extract_features(self, docs: Iterable[Document]) -> ExtractedFeatures:
        # Avoid repeated lookups of these properties
        feature_extractor = self.feature_extractor
        mention_encoder = self.mention_encoder

        mention_count = 0
        token_count = 0
        document_count = 0
        sentence_count = 0
        start_time = time.perf_counter()
        features = []
        labels = []
        for doc in docs:
            for sentence, mentions in doc.sentences_with_mentions():
                sent_x = feature_extractor.extract(sentence, doc)
                sent_y = mention_encoder.encode_mentions(sentence, mentions)
                assert len(sent_x) == len(sent_y)
                features.append(sent_x)
                labels.append(sent_y)

                mention_count += len(mentions)
                token_count += len(sent_x)
                sentence_count += 1

            document_count += 1

        print(
            f"Extracted features for {document_count} documents, {sentence_count} sentences, "
            f"{token_count} tokens, {mention_count} mentions in {time.perf_counter() - start_time} "
            "seconds"
        )

        return ExtractedFeatures(feature_extractor, features, labels)
