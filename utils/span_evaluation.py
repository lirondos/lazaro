from typing import Sequence, Dict, List, NamedTuple, Tuple, Counter, Set, Optional
import argparse
from collections import defaultdict
from decimal import ROUND_HALF_UP, Context
import tabulate

parser = argparse.ArgumentParser(description="A script for span level evaluation of seq labeling files in BIO format")
parser.add_argument('--predicted', type=str, help='Path to file with predicted data')
parser.add_argument('--goldstandard', type=str, help='Path to file with goldstandard data')
parser.add_argument('--untyped', action='store_true', help='Ignore categories, just consider span range (default False)')

class PRF1(NamedTuple):
    precision: float
    recall: float
    f1: float
    
class Span(NamedTuple): 
    start_token: int
    end_token: int
    #category: str


def compare(
    goldstandard: Sequence[Span],
    predicted: Sequence[Span],
    untyped: Optional[bool] = False,
) -> Dict[str, PRF1]:
    tp = []
    fp = []
    fn = []
    counts = defaultdict(lambda: defaultdict(int))
    for predicted_labels, goldstandard_labels in list(zip(predicted, goldstandard)):
        predicted_spans = labels_to_span_dict(predicted_labels)
        goldstandard_spans = labels_to_span_dict(goldstandard_labels)

        if untyped:
            predicted_spans = {"UNTYPED": set().union(*predicted_spans.values())}
            goldstandard_spans = {"UNTYPED": set().union(*goldstandard_spans.values())}
            
        for cat, spans_predicted in predicted_spans.items():
            spans_goldstandard = goldstandard_spans[cat]
            true_positives = spans_predicted.intersection(spans_goldstandard)
            counts[cat]["tp"] += len(true_positives)
            false_positives = spans_predicted.difference(spans_goldstandard)
            counts[cat]["fp"] += len(false_positives)
            false_negatives = spans_goldstandard.difference(spans_predicted)
            counts[cat]["fn"] += len(false_negatives)
    prf1 = dict()
    for key, value in counts.items():
        precision = get_precision(counts[key]["tp"], counts[key]["fp"])
        recall = get_recall(counts[key]["tp"], counts[key]["fn"])
        f1 = get_f1(precision, recall)
        prf1[key] = PRF1(precision, recall, f1)
    #get_prf1_all(counts, prf1)
    return prf1, counts

def get_precision(tp, fp):
    if tp + fp == 0:
        return 0
    return tp/(tp+fp)

def get_recall(tp, fn):
    if tp + fn == 0:
        return 0
    return tp/(tp+fn)

def get_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2*precision*recall/(precision+recall)

def print_results(prf1):
    # Always round .5 up, not towards even numbers as is the default
    rounder = Context(rounding=ROUND_HALF_UP, prec=4)
    print("Tag\tPrec\tRec\tF1")
    for ent_type, score in sorted(prf1.items()):
        if ent_type == "":
            ent_type = "ALL"
        metrics = [str(float(rounder.create_decimal_from_float(num * 100))) for num in score]
        #print("{:30s} {:30s}".format(ent_type, "\t".join(metrics)))
        print(ent_type + "\t" + "\t".join(metrics))
        
def print_counts(d):
    print()
    print("{:<8} {:<10} {:<10} {:<10}".format('CAT','TP','FP', 'FN'))
    for k, v in d.items():
        print("{:<8} {:<10} {:<10} {:<10}".format(k, v["tp"], v["fp"], v["fn"]))
    print()  


        
def labels_to_span_dict(labels: Sequence[str]) -> Dict[str, Set[Span]]:
    spans = defaultdict(set)
    tag_interruptus = False
    span_type = None
    initial = None
    for i, label in enumerate(labels):
        if (label == "O"):  # The current label is O
            if tag_interruptus:  # If we were in the middle of a span, we create it and append it
                spans[span_type].add(Span(initial, i))
            tag_interruptus = False  # We are no longer in the middle of a tag
        else:  # The current label is B I 
            label_type = label.split("-")[1]  # We get the type (PER, MISC, etc)
            if tag_interruptus:  # if we were in the middle of a tag
                if label_type != span_type or label.startswith("B"):  # and the types dont match or current tag is B
                    spans[span_type].add(Span(initial, i))  # we close the previous span and append it
                    span_type = label_type  # we are now in the middle of a new span
                    initial = i
            else:  # we were not in the middle of a span
                initial = i  # initial position will be the current position
                tag_interruptus = True  # we are now in the middle of a span
                span_type = label_type
    if tag_interruptus:  # this covers entities at the end of the sentence (we left a tag_interruptus at the end of the list)
        spans[span_type].add(Span(initial, len(labels)))
    return spans
        
def labels_to_span(labels: Sequence[str]) -> List[Span]:
    spans = []
    tag_interruptus = False
    span_type = None
    initial = None
    for i, label in enumerate(labels):
        if (label == "O"):  # The current label is O
            if tag_interruptus:  # If we were in the middle of a span, we create it and append it
                spans.append(Span(initial, i, span_type))
            tag_interruptus = False  # We are no longer in the middle of a tag
        else:  # The current label is B I 
            label_type = label.split("-")[1]  # We get the type (PER, MISC, etc)
            if tag_interruptus:  # if we were in the middle of a tag
                if label_type != span_type or label.startswith("B"):  # and the types dont match or current tag is B
                    spans.append(Span(initial, i, span_type))  # we close the previous span and append it
                    span_type = label_type  # we are now in the middle of a new span
                    initial = i
            else:  # we were not in the middle of a span
                initial = i  # initial position will be the current position
                tag_interruptus = True  # we are now in the middle of a span
                span_type = label_type
    if tag_interruptus:  # this covers entities at the end of the sentence (we left a tag_interruptus at the end of the list)
        spans.append(Span(initial, len(labels), span_type))
    return spans
    
def load_conll(path_to_file) -> Sequence[Sequence[str]]:
    sentences = []
    labels = []
    with open(path_to_file, "r", encoding="utf-8") as f:
        label = []
        sentence = []
        for line in f:
            if not line.strip():
                sentences.append(sentence)
                labels.append(label)
                sentence = []
                label = []
            else:
                try:
                    mytoken, mylabel = line.split(" ")
                    label.append(mylabel.rstrip())
                    sentence.append(mytoken.rstrip())
                except ValueError:
                    print("Skipping line with wrong CoNLL format")
    sentences.append(sentence) # we add the last sentence seq/token seq
    labels.append(label)
    return labels, sentences # sentences can be using for debugging purposes

if __name__ == "__main__":

    args = parser.parse_args()
    predicted_labels, _ = load_conll(args.predicted) 
    goldstandard_labels, _ = load_conll(args.goldstandard)
    prf1, counts = compare(goldstandard_labels, predicted_labels, args.untyped)
    print_counts(counts)
    print_results(prf1)

    

