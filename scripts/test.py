from typing import Mapping, Sequence, Dict, Optional, List, NamedTuple, Tuple, Counter, Iterable
#import pymagnitude
import csv
#from utils import FeatureExtractor, ScoringCounts, ScoringEntity

from nltk import ConfusionMatrix
from spacy.tokens import Span, Doc, Token
from spacy.language import Language
from spacy.tokenizer import Tokenizer
import pycrfsuite
from collections import defaultdict
import sys
import codecs
from decimal import ROUND_HALF_UP, Context
from statistics import mean
import spacy
import json
import random
import copy
import argparse
import pickle
import os
import string
from tabulate import tabulate
import os
from datetime import datetime, timezone
import time
from spacy.lang.tokenizer_exceptions import URL_PATTERN
import re
import sys
sys.path.append("/home/ealvarezmellado/lazaro/utils/")
#sys.path.append("/home/ealvarezmellado/lazaro/utils/")
from utils2 import BiasFeature, TokenFeature, UppercaseFeature, TitlecaseFeature, TrigramFeature, QuotationFeature, WordEnding, POStagFeature, WordVectorFeature, WordShapeFeature, WordVectorFeatureSpacy, BigramFeature, IsInDict, GraphotacticFeature, LemmaFeature, DigitFeature, PunctuationFeature, WordVectorFeatureNerpy, WordProbability, WordVectorFeatureNorm, SentencePositionFeature, BrownClusterFeature, HigherEnglishProbability, QuatrigramFeature, AllCapsFeature, PerplexityFeature, URLFeature, EmailFeature, TwitterFeature
from utils2 import WindowedTokenFeatureExtractor, CRFsuiteEntityRecognizer, BILOUEncoder, BIOEncoder, IOEncoder, ScoringCounts, ScoringEntity, BMESEncoder, BIOESEncoder
from constants import ANGLICISM_INDEX, TO_BE_TWEETED_PATTERN, AUTOMATICALLY_ANNOTATED_FOLDER, TO_BE_PREDICTED_FOLDER, CORPUS
from utils import PUNC_REPEAT_RE, DIGIT_RE, UPPERCASE_RE, LOWERCASE_RE
from utils import PRF1




KFOLD = 10
NLP = spacy.load('es_core_news_md', disable=["ner"])
TODAY = datetime.now(timezone.utc).strftime('%d%m%Y')


parser = argparse.ArgumentParser()


parser.add_argument('--test', type=str, help='Path to list file listing training files')
parser.add_argument('--predicted', type=str, help='Path where predicted docs will be stored')
parser.add_argument('--model', type=str, help='Path where CRF model is stored')
parser.add_argument('--encoder', type=str, default='BIO', help = 'Encoding to be apply (BIO, IO, BILOU; default BIO)')
parser.add_argument('--window', type=int, default=2, help = 'Window size to be considered (default 2)')
parser.add_argument('--verbose', action='store_true', help='Prints list of false positives, true positives and false negatives (default False)')
parser.add_argument('--stats', action='store_true', default=False, help='Print corpus numbers (number of tokens, anglicisms, headlines, etc)  (default False)')
parser.add_argument('--include_other', action='store_true', default=True, help='Whether to include OTHER tag  (default False)')
parser.add_argument('--collapse_tags', action='store_true', default=False, help='Whether to collapse ENGLISH and OTHER tags into a single LOANWORD tag  (default True)')
parser.add_argument('--embeddings', type=str, default="w2v_SBWC", help = 'Embeddings to be used: w2v_SBWC, glove_SBWC, fasttext_SBWC, fasttext_SUC, fasttext_wiki, spacy (default is spacy)')
parser.add_argument('--scaling', type=float, default=0.5, help = 'Scaling for word emebeddings (default 1.0)')
parser.add_argument('--has_goldstandard', action='store_true', default=False, help='Whether the test set has gold annotation and we want to evaluate')





ENCODER_DICT = {"BIO": BIOEncoder(), "IO": IOEncoder(), "BILOU": BILOUEncoder(), "BMES": BMESEncoder(), "BIOES": BIOESEncoder()}
TAG_COLLAPSE = {"ENG":"BORROWING", "OTHER":"BORROWING"}



def custom_tokenizer(nlp):
    # contains the regex to match all sorts of urls:
    prefix_re = re.compile(spacy.util.compile_prefix_regex(Language.Defaults.prefixes).pattern.replace("#", "!"))
    infix_re = spacy.util.compile_infix_regex(Language.Defaults.infixes)
    suffix_re = spacy.util.compile_suffix_regex(Language.Defaults.suffixes)

    #special_cases = {":)": [{"ORTH": ":)"}]}
    #prefix_re = re.compile(r'''^[[("']''')
    #suffix_re = re.compile(r'''[])"']$''')
    #infix_re = re.compile(r'''[-~]''')
    #simple_url_re = re.compile(r'''^#''')

    hashtag_pattern = r'''|^(#[\w_-]+)$'''
    url_and_hashtag = URL_PATTERN + hashtag_pattern
    url_and_hashtag_re = re.compile(url_and_hashtag)


    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=url_and_hashtag_re.match)

def ingest_json_document(doc_json: Mapping, nlp: Language, include_other: bool, is_predict = False) -> Doc:
    if is_predict:
        doc = nlp(doc_json["title"] + "\n" + doc_json["text"]) if "title" in doc_json else nlp(doc_json["text"])
        doc.user_data["date"] = doc_json["date"] if "date" in doc_json else ""
        doc.user_data["url"] = doc_json["url"] if "url" in doc_json else ""
        doc.user_data["newspaper"] = doc_json["source"] if "source" in doc_json else ""
        doc.user_data["categoria"] = doc_json["category"] if "category" in doc_json else ""
        doc.ents = []
        return doc
    else:
        if not doc_json["annotation_approver"] and not doc_json["labels"]:
            raise ValueError("Instance is not annotated!")
        else:
            doc = nlp(doc_json["text"])
            spans = list()
            #print(doc_json)
            for label in doc_json["labels"]:
                #print(doc_json["text"])
                if include_other or label[2] != "OTHER":
                    if doc_json["annotation_approver"] != "lazaro":
                        start_char =  label[0]
                        end_char = label[1]
                        tag = label[2]
                        token_start = get_starting_token(start_char, doc)
                        token_end = get_ending_token(end_char, doc)
                    else:
                        token_start =  label[0]
                        token_end = label[1]
                        tag = label[2]
                    if token_start is None or token_end is None:
                        raise ValueError("Token alignment impossible!")
                    spans.append(Span(doc, token_start, token_end, tag))
            doc.ents = spans
        return doc

def get_starting_token(start_char, doc):
    for token in doc:
        if start_char <= token.idx:
            return token.i
    return None

def get_ending_token(end_char, doc):
    for token in doc:
        if end_char <= token.idx:
            return token.i
    return doc[-1].i + 1
    
def evaluate(predicted, test):
    if args.verbose: print("Evaluating...")
    tag_collapse = None
    if args.collapse_tags:
        tag_collapse = TAG_COLLAPSE
    prf1, scores = span_prf1_type_map(test, predicted, tag_collapse)
    return prf1, scores

def span_prf1_type_map(
    reference_docs: Sequence[Doc],
    test_docs: Sequence[Doc],
    type_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, PRF1]:
    tp = []
    fp = []
    fn = []
    counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(reference_docs)):
        ents_from_ref = {ent for ent in reference_docs[i].ents}
        ents_from_test = {ent for ent in test_docs[i].ents}
        print(ents_from_ref)
        print(ents_from_test)
        if type_map is not None:
            ents_from_ref = remapping(ents_from_ref, type_map)  # ugly code, but otherwise the
            ents_from_test = remapping(ents_from_test, type_map)
        for ent_test in ents_from_test:
            if is_ent_in_list(ent_test, ents_from_ref):
                counts[ent_test.label_]["tp"] += 1
                tp.append(ScoringEntity(tuple(ent_test.text.split()), ent_test.label_))
            else:
                counts[ent_test.label_]["fp"] += 1
                fp.append(ScoringEntity(tuple(ent_test.text.split()), ent_test.label_))
        for ent_ref in ents_from_ref:
            if not is_ent_in_list(ent_ref, ents_from_test):
                counts[ent_ref.label_]["fn"] += 1
                fn.append(ScoringEntity(tuple(ent_ref.text.split()), ent_ref.label_))
    prf1 = dict()
    for key, value in counts.items():
        precision = get_precision(counts[key]["tp"], counts[key]["fp"])
        recall = get_recall(counts[key]["tp"], counts[key]["fn"])
        f1 = get_f1(precision, recall)
        prf1[key] = PRF1(precision, recall, f1)
    get_prf1_all(counts, prf1)
    return prf1, ScoringCounts(Counter(tp), Counter(fp), Counter(fn))

def get_ents(
    docs: Sequence[Doc],
    type_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, PRF1]:
    all_ents = list()
    for i in range(len(docs)):
        ents = {ent for ent in docs[i].ents}
        if type_map is not None:
            ents = remapping(ents, type_map)  # ugly code, but otherwise the
        for ent in ents:
            all_ents.append(ScoringEntity(tuple(ent.text.split()), ent.label_))
    return all_ents


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

def print_results(prf1):
    # Always round .5 up, not towards even numbers as is the default
    rounder = Context(rounding=ROUND_HALF_UP, prec=4)
    #print("{:30s} {:30s}".format("Tag", "Prec\tRec\tF1"))
    print("Tag\tPrec\tRec\tF1")
    for ent_type, score in sorted(prf1.items()):
        if ent_type == "":
            ent_type = "ALL"
        metrics = [str(float(rounder.create_decimal_from_float(num * 100))) for num in score]
        #print("{:30s} {:30s}".format(ent_type, "\t".join(metrics)))
        print(ent_type + "\t" + "\t".join(metrics))


def load_data(path, include_other, is_predict = False):
    mylist = list()
    with open(path.rstrip(), encoding="utf8") as f:
        lines = f.readlines()
    for line in lines:
        try:
            json_as_dict = json.loads(line.rstrip())
            doc = ingest_json_document(json_as_dict, NLP, include_other, is_predict)
            mylist.append(doc)
        except ValueError as err:
            print(line)
    return mylist

def print_statistics(training, test) -> None:
    headlines_training = 0
    tokens_training = 0
    headlines_with_loanwords_training = 0
    total_loanwords_in_training = 0
    tag_count_training = defaultdict(int)
    for doc in training:
        headlines_training += 1
        tokens_training += len(doc)
        headlines_with_loanwords_training = headlines_with_loanwords_training + int(bool(doc.ents))
        total_loanwords_in_training += len(doc.ents)
        for ent in doc.ents:
            tag_count_training[ent.label_] += 1

    headlines_test = 0
    tokens_test = 0
    headlines_with_loanwords_test = 0
    total_loanwords_in_test = 0
    tag_count_test = defaultdict(int)
    for doc in test:
        headlines_test += 1
        tokens_test += len(doc)
        headlines_with_loanwords_test = headlines_with_loanwords_test + int(bool(doc.ents))
        for ent in doc.ents:
            tag_count_test[ent.label_] += 1
        total_loanwords_in_test = total_loanwords_in_test + len(doc.ents)

    table = [
        ["Number of headlines", headlines_training, headlines_test],
        ["Number of tokens", tokens_training, tokens_test],
        ["Number of headlines with loanwords", headlines_with_loanwords_training, headlines_with_loanwords_test],
        ["Number of loanwords", total_loanwords_in_training, total_loanwords_in_test],
        ["Number of anglicisms", tag_count_training["ENG"], tag_count_test["ENG"]],
        ["Number of OTHER", tag_count_training["OTHER"], tag_count_test["OTHER"]]
    ]

    print(tabulate(table, headers=["", "Training", "Test"]))

def predict(path_to_model, window_size, test_set) -> None:
    features = [    #WordVectorFeatureNerpy(args.embeddings, args.scaling),
                    BiasFeature(),
                    TokenFeature(),
                    UppercaseFeature(),
                    TitlecaseFeature(),
                    TrigramFeature(),
                    QuotationFeature(),
                    WordEnding(),
                    POStagFeature(),
                    WordShapeFeature(),
                    URLFeature(),
                    EmailFeature(),
                    TwitterFeature()
                    ]
                    
    crf = CRFsuiteEntityRecognizer(WindowedTokenFeatureExtractor(features,window_size,), ENCODER_DICT[args.encoder])
    crf.tagger = pycrfsuite.Tagger()
    crf.tagger.open(path_to_model)


    if args.verbose: print("Predicting...")
    
    test_set = copy.deepcopy(test_set)
    for doc in test_set:
        doc.ents = []
    predicted = [crf(doc) for doc in test_set]
    return predicted

    
def write_predictions(predicted_docs) -> None:
    for mydoc in predicted_docs:
        labels = list()
        labels_tokens = list()
        myspans = list()
        for ent in mydoc.ents:
            if ent.end == len(mydoc):
                labels.append([mydoc[ent.start].idx, mydoc[-1].idx, ent.label_])
            else:
                labels.append([mydoc[ent.start].idx, mydoc[ent.end].idx, ent.label_])
            labels_tokens.append([ent.start, ent.end, ent.label_])
            myspans.append(ent.text)
        mydict = {"text": mydoc.text, "date": mydoc.user_data["date"], "annotation_approver": "lazaro", "newspaper": mydoc.user_data["newspaper"], "categoria": mydoc.user_data["categoria"], "url": mydoc.user_data["url"], "labels": labels, "labels_tokens": labels_tokens, "spans": myspans}
        with open(args.predicted, 'a', encoding = "utf-8") as f:
            f.write(json.dumps(mydict)+'\n')



if __name__ == "__main__":

    args = parser.parse_args()

    NLP.tokenizer = custom_tokenizer(NLP)

    if args.verbose: print(args)
    if args.verbose: print("Loading test data...")

    test = list()
    with open(args.test, "r", encoding="utf-8") as f:
        for line in f:
            test.extend(load_data(line, args.include_other, is_predict=not args.has_goldstandard))
    predicted_docs = predict(args.model, args.window, test)
    if args.has_goldstandard:
        prf1, predictions = evaluate(predicted_docs, test)
        #print(predictions)
        print_results(prf1)
    write_predictions(predicted_docs)