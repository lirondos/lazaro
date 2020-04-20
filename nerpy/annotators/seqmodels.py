"""A CRFSuite-based mention annotator."""
import pickle
import time
from pathlib import Path
from typing import IO, Dict, Iterable, List, Optional, Sequence, Union

from attr import attrib, attrs
from attr.validators import instance_of

from nerpy.annotator import SequenceMentionAnnotator
from nerpy.document import Document, Mention, MentionType
from nerpy.encoding import MentionEncoder
from nerpy.features import SentenceFeatureExtractor, SequenceFeatures, SequenceLabels
from sequencemodels import ViterbiStructuredPerceptron


# Due to the model object, cannot be frozen
@attrs
class SequenceModelsAnnotator(SequenceMentionAnnotator):
    _mention_type: MentionType = attrib(validator=instance_of(MentionType))
    _feature_extractor: SentenceFeatureExtractor = attrib(
        validator=instance_of(SentenceFeatureExtractor)
    )
    # Mypy raised a false positive about a concrete class being needed
    _mention_encoder: MentionEncoder = attrib(
        validator=instance_of(MentionEncoder)  # type: ignore
    )
    _model: ViterbiStructuredPerceptron = attrib(
        validator=instance_of(ViterbiStructuredPerceptron)
    )

    # TODO: Make serialization model work sanely across all annotators
    @classmethod
    def from_model(
        cls,
        mention_type: MentionType,
        feature_extractor: SentenceFeatureExtractor,
        mention_encoder: MentionEncoder,
        model_path: Union[str, Path],
    ) -> "SequenceModelsAnnotator":
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        return cls(mention_type, feature_extractor, mention_encoder, model)

    @classmethod
    def for_training(
        cls,
        mention_type: MentionType,
        feature_extractor: Optional[SentenceFeatureExtractor],
        mention_encoder: MentionEncoder,
    ) -> "SequenceModelsAnnotator":
        model = ViterbiStructuredPerceptron()
        return cls(mention_type, feature_extractor, mention_encoder, model)

    def mentions(self, doc: Document) -> Sequence[Mention]:
        mentions: List[Mention] = []
        for sentence in doc.sentences:
            sent_x = self._feature_extractor.extract(sentence, doc)
            pred_y = self._model.predict(sent_x)
            mentions.extend(self._mention_encoder.decode_mentions(sentence, pred_y))

        return mentions

    @property
    def mention_encoder(self) -> MentionEncoder:
        return self._mention_encoder

    @property
    def feature_extractor(self) -> SentenceFeatureExtractor:
        return self._feature_extractor

    def train(
        self,
        docs: Iterable[Document],
        *,
        epochs: int,
        averaged: bool = True,
        verbose: bool = False,
        log_file: Optional[IO[str]] = None,
    ) -> None:
        mention_count = 0
        token_count = 0
        document_count = 0
        sentence_count = 0
        print("Extracting features", file=log_file)
        start_time = time.perf_counter()

        features: List[SequenceFeatures] = []
        labels: List[SequenceLabels] = []

        for doc in docs:
            for sentence, mentions in doc.sentences_with_mentions():
                sent_x = self._feature_extractor.extract(sentence, doc)
                sent_y = self._mention_encoder.encode_mentions(sentence, mentions)
                assert len(sent_x) == len(sent_y)
                features.append(sent_x)
                labels.append(sent_y)

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
        self._model.train(
            features, labels, epochs=epochs, averaged=averaged, verbose=verbose
        )
        print(
            "Training took {} seconds".format(time.perf_counter() - start_time),
            file=log_file,
        )


def train_seqmodels(
    mention_encoder: MentionEncoder,
    feature_extractor: SentenceFeatureExtractor,
    mention_type: MentionType,
    model_path: Union[str, Path],
    train_docs: Iterable[Document],
    train_params: Dict,
    *,
    verbose: bool = False,
) -> SequenceModelsAnnotator:
    epochs = train_params["max_iterations"]
    averaged = bool(train_params.get("averaged", True))

    annotator = SequenceModelsAnnotator.for_training(
        mention_type, feature_extractor, mention_encoder
    )
    annotator.train(
        train_docs, epochs=epochs, averaged=averaged, verbose=verbose,
    )
    with open(model_path, "wb") as model_file:
        pickle.dump(annotator._model, model_file, protocol=pickle.HIGHEST_PROTOCOL)
    return annotator
