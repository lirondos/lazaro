"""A CRFSuite-based mention annotator."""
import time
from pathlib import Path
from typing import IO, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from attr import attrib, attrs
from attr.validators import instance_of
from pycrfsuite import Tagger, Trainer  # pylint: disable=no-name-in-module

from nerpy.annotator import SequenceMentionAnnotator, Trainable
from nerpy.document import Document, Mention, MentionType
from nerpy.encoding import MentionEncoder
from nerpy.features import ExtractedFeatures, SentenceFeatureExtractor

# TODO: Figure out how to serialize models with their strategies and feature extractors
# TODO: Refactor to reduce redundancy around feature extraction and multiple training methods


# Due to the Tagger object, cannot be frozen
@attrs
class CRFSuiteAnnotator(SequenceMentionAnnotator, Trainable):
    _mention_type: MentionType = attrib(validator=instance_of(MentionType))
    _feature_extractor: SentenceFeatureExtractor = attrib(
        validator=instance_of(SentenceFeatureExtractor)
    )
    # Mypy raised a false positive about a concrete class being needed
    _mention_encoder: MentionEncoder = attrib(
        validator=instance_of(MentionEncoder)  # type: ignore
    )
    _tagger: Tagger = attrib(validator=instance_of(Tagger))

    @classmethod
    def from_model(
        cls,
        mention_type: MentionType,
        feature_extractor: SentenceFeatureExtractor,
        mention_encoder: MentionEncoder,
        model_path: Union[str, Path],
    ) -> "CRFSuiteAnnotator":
        tagger = Tagger()
        tagger.open(model_path)
        return cls(mention_type, feature_extractor, mention_encoder, tagger)

    @classmethod
    def for_training(
        cls,
        mention_type: MentionType,
        feature_extractor: Optional[SentenceFeatureExtractor],
        mention_encoder: MentionEncoder,
    ) -> "CRFSuiteAnnotator":
        tagger = Tagger()
        return cls(mention_type, feature_extractor, mention_encoder, tagger)

    def mentions(self, doc: Document) -> Sequence[Mention]:
        mentions: List[Mention] = []
        for sentence in doc.sentences:
            sent_x = self._feature_extractor.extract(sentence, doc)
            pred_y = self._tagger.tag(sent_x)
            mentions.extend(self._mention_encoder.decode_mentions(sentence, pred_y))

        return mentions

    @property
    def mention_encoder(self) -> MentionEncoder:
        return self._mention_encoder

    @property
    def feature_extractor(self) -> SentenceFeatureExtractor:
        return self._feature_extractor

    # Training method specifies values for the kwargs, so it will not exactly match the interface
    # noinspection PyMethodOverriding
    def train(  # type: ignore
        self,
        docs: Iterable[Document],
        *,
        model_path: Union[str, Path],
        algorithm: str,
        train_params: Optional[Mapping] = None,
        verbose: bool = False,
        log_file: Optional[IO[str]] = None,
    ) -> None:
        if train_params is None:
            train_params = {}
        trainer = Trainer(algorithm=algorithm, params=train_params, verbose=verbose)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        mention_count = 0
        token_count = 0
        document_count = 0
        sentence_count = 0
        print("Extracting features", file=log_file)
        start_time = time.perf_counter()
        for doc in docs:
            for sentence, mentions in doc.sentences_with_mentions():
                sent_x = self._feature_extractor.extract(sentence, doc)
                sent_y = self._mention_encoder.encode_mentions(sentence, mentions)
                assert len(sent_x) == len(sent_y)
                trainer.append(sent_x, sent_y)

                mention_count += len(mentions)
                token_count += len(sent_x)
                sentence_count += 1

            document_count += 1

        print(
            "Feature extraction took {} seconds".format(time.perf_counter() - start_time),
            file=log_file,
        )
        print(
            f"Extracted features for {document_count} documents, {sentence_count} sentences, "
            f"{token_count} tokens, {mention_count} mentions",
            file=log_file,
        )
        print("Training", file=log_file)
        start_time = time.perf_counter()
        trainer.train(model_path)
        print(
            "Training took {} seconds".format(time.perf_counter() - start_time),
            file=log_file,
        )
        self._tagger.open(model_path)

    def train_featurized(
        self,
        training_data: ExtractedFeatures,
        model_path: Union[str, Path],
        *,
        algorithm: str,
        train_params: Optional[Mapping] = None,
        verbose: bool = False,
        log_file: Optional[IO[str]] = None,
    ) -> None:
        assert (
            training_data.extractor == self._feature_extractor
        ), "Training data feature extractor differs from instance feature extractor"

        if train_params is None:
            train_params = {}
        trainer = Trainer(algorithm=algorithm, params=train_params, verbose=verbose)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        for sent_x, sent_y in zip(training_data.features, training_data.labels):
            trainer.append(sent_x, sent_y)

        start_time = time.perf_counter()
        trainer.train(model_path)
        print(
            "Training took {} seconds".format(time.perf_counter() - start_time),
            file=log_file,
        )
        self._tagger.open(model_path)


def train_crfsuite(
    mention_encoder: MentionEncoder,
    feature_extractor: SentenceFeatureExtractor,
    mention_type: MentionType,
    model_path: Union[str, Path],
    train_docs: Iterable[Document],
    train_params: Dict,
    *,
    verbose: bool = False,
) -> CRFSuiteAnnotator:
    algorithm = train_params.pop("algorithm")
    annotator = CRFSuiteAnnotator.for_training(
        mention_type, feature_extractor, mention_encoder
    )
    annotator.train(
        train_docs,
        model_path=model_path,
        algorithm=algorithm,
        train_params=train_params,
        verbose=verbose,
    )
    return annotator
