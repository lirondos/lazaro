import unittest
from typing import Sequence, Dict, List, NamedTuple, Tuple, Counter
from span_evaluation import Span, PRF1, ScoringEntity, ScoringCounts, labels_to_span



class TestSpanEvaluation(unittest.TestCase):
    def test_labels_to_span(self):
        self.assertEqual(
            [
                Span(1,3,"ENG"),
                Span(4, 5, "ENG")
            ],
            labels_to_span(["O", "B-ENG", "I-ENG", "O", "B-ENG"]),
        )
        self.assertEqual(
            [
                Span(0,1,"ENG")
            ],
            labels_to_span(["B-ENG", "O"]),
        )
        self.assertEqual(
            [
                Span(0,1,"ENG")
            ],
            labels_to_span(["B-ENG"]),
        )
        self.assertEqual(
            [
                Span(0,2,"ENG")
            ],
            labels_to_span(["B-ENG", "I-ENG"]),
        )
        self.assertEqual(
            [
                Span(0,1,"ENG"),
                Span(1,2,"ENG")
            ],
            labels_to_span(["B-ENG", "B-ENG", "O"]),
        )
        self.assertEqual(
            [
                Span(0,1,"ENG"),
                Span(1,2,"OTHER")
            ],
            labels_to_span(["B-ENG", "B-OTHER", "O"]),
        )
        self.assertEqual(
            [
                Span(0,1,"ENG"),
                Span(1,2,"OTHER")
            ],
            labels_to_span(["B-ENG", "I-OTHER", "O"]),
        )

if __name__ == "__main__":
    unittest.main()