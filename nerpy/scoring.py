from collections import defaultdict
from decimal import ROUND_HALF_UP, Context, Decimal
from typing import DefaultDict, Dict, Iterable, Mapping, Optional, TextIO

from attr import attrib, attrs

from nerpy.document import Document, EntityType, Token

TokenCounter = DefaultDict[EntityType, DefaultDict[str, "ScoringCounter"]]


@attrs(frozen=True)
class Score:
    precision: float = attrib()
    recall: float = attrib()
    fscore: float = attrib()


@attrs(frozen=True)
class ScoringResult:
    score: Score = attrib()
    type_scores: Mapping[EntityType, Score] = attrib()

    def __str__(self) -> str:
        lines = ["***** Entity Type Scores *****"]
        for entity in sorted(self.type_scores):
            scores = self.type_scores[entity]
            lines.append(f"Entity Type: {entity}")
            lines.append(f"Precision: {_as_decimal(scores.precision):0.2f}")
            lines.append(f"Recall: {_as_decimal(scores.recall):0.2f}")
            lines.append(f"F1-score: {_as_decimal(scores.fscore):0.2f}")
            lines.append("")

        lines.append("***** Overall Score *****")
        lines.append(f"Total Precision: {_as_decimal(self.score.precision):0.2f}")
        lines.append(f"Total Recall:  {_as_decimal(self.score.recall):0.2f}")
        lines.append(f"Total F1-Score: {_as_decimal(self.score.fscore):0.2f}")

        return "\n".join(lines)

    # TODO: Deprecate this and refactor callers to print on their own
    def print(self, file: Optional[TextIO] = None) -> None:
        print(str(self), file=file)


def _as_decimal(num: float) -> Decimal:
    # Always round .5 up, not towards even numbers as is the default
    rounder = Context(rounding=ROUND_HALF_UP, prec=4)
    return rounder.create_decimal_from_float(num * 100)


@attrs(auto_attribs=True)
class ScoringCounter:
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0


def score_prf(
    gold_docs: Iterable[Document],
    system_docs: Iterable[Document],
    *,
    check_docids: bool = False,
) -> ScoringResult:

    # Counters for total precision and recall
    precision_true_positives = 0
    recall_true_positives = 0
    total_system_mentions = 0
    total_gold_mentions = 0

    # Counters for per entity precision and recall
    entity_counts: DefaultDict[EntityType, ScoringCounter] = defaultdict(ScoringCounter)

    # TODO: Correctly handle the case where number of system and gold docs do not match
    for gold_doc, system_doc in zip(gold_docs, system_docs):
        if check_docids and system_doc.id != gold_doc.id:
            raise ValueError(
                "Gold and system document IDs do not match: "
                + str(gold_doc.id)
                + ","
                + str(system_doc.id)
            )

        system_mentions = system_doc.mentions
        gold_mentions = gold_doc.mentions

        # Update total and per entity precision counts
        for mention in system_mentions:
            if mention in gold_mentions:
                precision_true_positives += 1
                entity_counts[mention.entity_type].true_positives += 1
            else:
                entity_counts[mention.entity_type].false_positives += 1

        # Update total and per entity recall counts
        for mention in gold_mentions:
            if mention in system_mentions:
                recall_true_positives += 1
            else:
                entity_counts[mention.entity_type].false_negatives += 1

        total_system_mentions += len(system_mentions)
        total_gold_mentions += len(gold_mentions)

    # Calculate total precision, recall and fscore
    total_precision = (
        precision_true_positives / total_system_mentions if total_system_mentions else 0.0
    )
    total_recall = (
        recall_true_positives / total_gold_mentions if total_gold_mentions else 0.0
    )
    total_fscore = (
        (2 * total_precision * total_recall) / (total_precision + total_recall)
        if (total_precision + total_recall)
        else 0.0
    )

    # Calculate per entity precision, recall and fscore
    type_scores: Dict[EntityType, Score] = {}
    for entity_type in entity_counts:
        tp = entity_counts[entity_type].true_positives
        fp = entity_counts[entity_type].false_positives
        fn = entity_counts[entity_type].false_negatives

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = (2 * p * r) / (p + r) if (p + r) else 0.0

        score = Score(p, r, f)
        type_scores[entity_type] = score

    # Return scoring result
    total_score = Score(total_precision, total_recall, total_fscore)
    scoring_result = ScoringResult(total_score, type_scores)
    return scoring_result


class ScoringCounts:
    def count(
        self,
        system_docs: Iterable[Document],
        gold_docs: Iterable[Document],
        *,
        check_docids: bool = False,
    ) -> TokenCounter:
        # Counter for each entity type
        scoring_counts: TokenCounter = defaultdict(lambda: defaultdict(ScoringCounter))

        # TODO: Correctly handle the case where number of system and gold docs do not match
        for gold_doc, system_doc in zip(gold_docs, system_docs):
            if check_docids and system_doc.id != gold_doc.id:
                raise ValueError(
                    "Gold and system document IDs do not match: "
                    + str(gold_doc.id)
                    + ","
                    + str(system_doc.id)
                )

            system_mentions = system_doc.mentions
            gold_mentions = gold_doc.mentions

            for mention in system_mentions:
                entity_type = mention.entity_type
                tokens = mention.tokens(system_doc)
                token_text = ScoringCounts._extract_token_text(tokens)

                if mention not in gold_mentions:
                    # False positive
                    scoring_counts[entity_type][token_text].false_positives += 1
                else:
                    # True positive
                    scoring_counts[entity_type][token_text].true_positives += 1

            for mention in gold_mentions:
                entity_type = mention.entity_type
                tokens = mention.tokens(system_doc)
                token_text = ScoringCounts._extract_token_text(tokens)

                if mention not in system_mentions:
                    # False negative
                    scoring_counts[entity_type][token_text].false_negatives += 1

        return scoring_counts

    @staticmethod
    def _extract_token_text(tokens: Iterable[Token]) -> str:
        token_text = " ".join(token.text for token in tokens)
        return token_text
