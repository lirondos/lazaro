from typing import Mapping, Sequence, Dict, Optional, List, NamedTuple, Tuple, Counter, Iterable
import pymagnitude
import csv
#from utils import FeatureExtractor, ScoringCounts, ScoringEntity

from nltk import ConfusionMatrix
from spacy.tokens import Span, Doc, Token
from spacy.language import Language
from utils import PUNC_REPEAT_RE, DIGIT_RE, UPPERCASE_RE, LOWERCASE_RE
import pycrfsuite
from collections import defaultdict
import sys
from decimal import ROUND_HALF_UP, Context

import spacy
from utils import PRF1
import json
import random
import copy
import pickle
import os
import string
from utils2 import WindowedTokenFeatureExtractor, CRFsuiteEntityRecognizer, BILOUEncoder, BIOEncoder, IOEncoder, ScoringCounts, ScoringEntity
from utils2 import BiasFeature, TokenFeature, UppercaseFeature, TitlecaseFeature, TrigramFeature, QuotationFeature, WordEnding, POStagFeature, WordVectorFeature, WordShapeFeature

NLP = spacy.load('es_core_news_md', disable=["ner"])

def ingest_json_document(doc_json: Mapping, nlp: Language) -> Doc:
    #json_as_dict = json.loads(doc_json)
    if not doc_json["annotation_approver"] and not doc_json["labels"]:
        raise ValueError("Instance is not annotated!")
    else:
        doc = nlp(doc_json["text"])
        #print(doc)
        #print(doc_json["id"])
        spans = list()
        for label in doc_json["labels"]:
            #if label[2] != "OTHER":
            start_char =  label[0]
            end_char = label[1]
            tag = label[2]
            token_start = get_starting_token(start_char, doc)
            token_end = get_ending_token(end_char, doc)
            if token_start is None or token_end is None:
                raise ValueError("Token alignment impossible!")
            spans.append(Span(doc, token_start, token_end, tag))
        doc.ents = spans
        return doc
    #except ValueError:
        #print(("Instance is not annotated!"))


def get_starting_token(start_char, doc):
    for token in doc:
        if start_char <= token.idx:
            return token.i
    return None

def get_ending_token(end_char, doc):
    for token in doc:
        if end_char <= token.idx:
            return token.i
    return None

def span_prf1_type_map(
    reference_docs: Sequence[Doc],
    test_docs: Sequence[Doc],
    type_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, PRF1]:
    """
    if type_map is not None:
        remapping(reference_docs, type_map) # ugly code, but otherwise the
        remapping(test_docs, type_map)
    """
    counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(reference_docs)):
        ents_from_ref = {ent for ent in reference_docs[i].ents}
        ents_from_test = {ent for ent in test_docs[i].ents}
        if type_map is not None:
            ents_from_ref = remapping(ents_from_ref, type_map)  # ugly code, but otherwise the
            ents_from_test = remapping(ents_from_test, type_map)
        for ent_test in ents_from_test:
            if is_ent_in_list(ent_test, ents_from_ref):
                counts[ent_test.label_]["tp"] += 1
            else:
                counts[ent_test.label_]["fp"] += 1
        for ent_ref in ents_from_ref:
            if not is_ent_in_list(ent_ref, ents_from_test):
                counts[ent_ref.label_]["fn"] += 1
    prf1 = dict()
    for key, value in counts.items():
        precision = get_precision(counts[key]["tp"], counts[key]["fp"])
        recall = get_recall(counts[key]["tp"], counts[key]["fn"])
        f1 = get_f1(precision, recall)
        prf1[key] = PRF1(precision, recall, f1)
    get_prf1_all(counts, prf1)
    #print(counts)
    return prf1

def remapping(ents, type_map):
    new_ents = set()
    for ent in ents:
        if ent.label_ in type_map.keys():
            new_ents.add(Span(ent.doc, ent.start, ent.end, type_map[ent.label_]))
        else:
            new_ents.add(ent)
    return new_ents


def is_ent_in_list(ent_ref, list):
    for elem in list:
        if same_ents(ent_ref, elem):
            return True
    return False

def get_prf1_all(counts, prf1):
    tp_all = 0
    fp_all = 0
    fn_all = 0
    for ent, values in counts.items():
        tp_all += counts[ent]["tp"]
        fp_all += counts[ent]["fp"]
        fn_all += counts[ent]["fn"]
    precision_all = get_precision(tp_all, fp_all)
    recall_all = get_recall(tp_all, fn_all)
    prf1[""] = PRF1(precision_all, recall_all, get_f1(precision_all, recall_all))

def same_ents(ent1, ent2):
    return ent1.label_ == ent2.label_ and ent1.start == ent2.start and ent1.end == ent2.end

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



def span_scoring_counts(
    reference_docs: Sequence[Doc], test_docs: Sequence[Doc], typed: bool = True
) -> ScoringCounts:
    tp = []
    fp = []
    fn = []
    for i in range(len(reference_docs)):
        doc_reference = reference_docs[i]
        doc_test = test_docs[i]
        #if not typed:
        #remove_labels(doc_reference, doc_test)
        for ent_test in doc_test.ents:
            if is_ent_in_list(ent_test, doc_reference.ents):
                tp.append(ScoringEntity(tuple(ent_test.text.split()), ent_test.label_))
            else:
                fp.append(ScoringEntity(tuple(ent_test.text.split()), ent_test.label_))
        for ent_ref in doc_reference.ents:
            if not is_ent_in_list(ent_ref, doc_test.ents):
                fn.append(ScoringEntity(tuple(ent_ref.text.split()), ent_ref.label_))
    return ScoringCounts(Counter(tp), Counter(fp), Counter(fn))



def print_results(prf1):
    # Always round .5 up, not towards even numbers as is the default
    rounder = Context(rounding=ROUND_HALF_UP, prec=4)
    print("{:30s} {:30s}".format("Tag", "Prec\tRec\tF1"), file=sys.stderr)
    for ent_type, score in sorted(prf1.items()):
        if ent_type == "":
            ent_type = "ALL"
        metrics = [str(float(rounder.create_decimal_from_float(num * 100))) for num in score]
        print("{:30s} {:30s}".format(ent_type, "\t".join(metrics)), file=sys.stderr)


def load_data(path):
    mylist = list()
    with open(path, encoding="utf8") as f:
        lines = f.readlines()
    for line in lines:
        json_as_dict = json.loads(line)
        try:
            doc = ingest_json_document(json_as_dict, NLP)
            mylist.append(doc)
        except ValueError:
            pass
    return mylist

def main(path_to_train, path_to_test) -> None:
    training = load_data(path_to_train)
    test = load_data(path_to_test) 
    predicted = copy.deepcopy(test)
    for doc in predicted:
        doc.ents = [] 
        """
        with open('data.pickle', 'wb') as handle:
            pickle.dump(docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        I chose to pickle in order to: 
        - avoid doing everything from the begining every time I run the code
        - make sure that the experiements are run on the same test/train split (so that metrics are comparable)
    
        with open('data.pickle', 'rb') as handle:
            docs = pickle.load(handle)
        """

    features = [BiasFeature(),                    
                    TokenFeature(),
                    UppercaseFeature(),
                    TitlecaseFeature(),
                    TrigramFeature(),
                    QuotationFeature(),
                    WordEnding(),
                    POStagFeature(),
                    WordShapeFeature(),
                    WordVectorFeature()
                    ]

    crf = CRFsuiteEntityRecognizer(WindowedTokenFeatureExtractor(features,2,), BIOEncoder())
    crf.train(training, "lbfgs", {"max_iterations":  40, 'c2': 0.01, 'c1': 0.01}, "tmp.model")
    predicted = [crf(doc) for doc in predicted]
    #prf1 = span_prf1_type_map(test, predicted, {"ENG":"LOANWORD", "OTHER":"LOANWORD"})
    prf1 = span_prf1_type_map(test, predicted)
    print_results(prf1)
    print(span_scoring_counts(test, predicted))



if __name__ == "__main__":
    path_to_train = "data/training.jsonl"
    path_to_test = "data/test.jsonl"

    main(path_to_train, path_to_test)
