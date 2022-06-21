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
print(sys.path)
#sys.path.append("/home/ealvarezmellado/lazaro/utils/")
from utils2 import BiasFeature, TokenFeature, UppercaseFeature, TitlecaseFeature, TrigramFeature, QuotationFeature, WordEnding, POStagFeature, WordVectorFeature, WordShapeFeature, WordVectorFeatureSpacy, BigramFeature, IsInDict, GraphotacticFeature, LemmaFeature, DigitFeature, PunctuationFeature, WordVectorFeatureNerpy, WordProbability, WordVectorFeatureNorm, SentencePositionFeature, BrownClusterFeature, HigherEnglishProbability, QuatrigramFeature, AllCapsFeature, PerplexityFeature, URLFeature, EmailFeature, TwitterFeature
from utils2 import WindowedTokenFeatureExtractor, CRFsuiteEntityRecognizer, BILOUEncoder, BIOEncoder, IOEncoder, ScoringCounts, ScoringEntity, BMESEncoder, BIOESEncoder
from constants import ANGLICISM_INDEX, TO_BE_TWEETED_PATTERN, AUTOMATICALLY_ANNOTATED_FOLDER, TO_BE_PREDICTED_FOLDER, CORPUS
from utils import PUNC_REPEAT_RE, DIGIT_RE, UPPERCASE_RE, LOWERCASE_RE
from utils import PRF1
from io import open
from conllu import parse_incr


KFOLD = 10
NLP = spacy.load('es_core_news_md', disable=["ner"])
TODAY = datetime.now(timezone.utc).strftime('%d%m%Y')


parser = argparse.ArgumentParser()


parser.add_argument('--training', type=str, help='Path to list file listing training files', default=CORPUS)
parser.add_argument('--model_path', type=str, help='Path where the model will be stored', default="model.tmp")
parser.add_argument('--max_iterations', type=int, default=None, help='Max interations (default None)')
parser.add_argument('--c1',type=float, default=0.05, help='L1 regularization coefficient (default 0.01)')
parser.add_argument('--c2',type=float, default=0.01, help='L2 regularization coefficient (default 0.01)')
parser.add_argument('--delta',type=float, default=1e-3, help='delta')
parser.add_argument('--encoder', type=str, default='BIO', help = 'Encoding to be apply (BIO, IO, BILOU; default BIO)')
parser.add_argument('--window', type=int, default=2, help = 'Window size to be considered (default 2)')
parser.add_argument('--embeddings', type=str, default="w2v_SBWC", help = 'Embeddings to be used: w2v_SBWC, glove_SBWC, fasttext_SBWC, fasttext_SUC, fasttext_wiki, spacy (default is spacy)')
parser.add_argument('--scaling', type=float, default=0.5, help = 'Scaling for word emebeddings (default 1.0)')
parser.add_argument('--verbose', action='store_true', help='Prints list of false positives, true positives and false negatives (default False)')
parser.add_argument('--stats', action='store_true', default=False, help='Print corpus numbers (number of tokens, anglicisms, headlines, etc)  (default False)')
parser.add_argument('--include_other', action='store_true', default=True, help='Whether to include OTHER tag  (default False)')
parser.add_argument('--collapse_tags', action='store_true', default=False, help='Whether to collapse ENGLISH and OTHER tags into a single LOANWORD tag  (default True)')




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
        doc = nlp(doc_json["title"] + "\n" + doc_json["text"])
        doc.user_data["date"] = doc_json["date"]
        doc.user_data["url"] = doc_json["url"]
        doc.user_data["newspaper"] = doc_json["newspaper"]
        doc.user_data["categoria"] = doc_json["categoria"]
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

def load_data_conll(path, is_test = False):
    mylist = list()
    with open(path, encoding="utf8") as f:
        lines = f.readlines()
    sentence = list()
    tags = list()
    for line in lines:
        if line.strip():
            if is_test:
                sentence.append(line.split()[0])
            else:
                el = line.split()
                if len(el) == 2:
                    sentence.append(el[0])
                    tags.append(el[1])
                else:
                    continue
        else:
            doc = NLP(" ".join(sentence))
            if is_test:
                mylist.append(doc)
            else:
                mylist.append((doc, tags))
                tags = list()
            sentence = list()
    return mylist


def train(train_set, max_iterations, c1, c2, encoder, window_size) -> None:
    features = [    WordVectorFeatureNerpy(args.embeddings, args.scaling),
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

    crf = CRFsuiteEntityRecognizer(WindowedTokenFeatureExtractor(features,window_size,), ENCODER_DICT[encoder])
    if args.verbose: print("Training...")
    crf.train(train_set, "lbfgs", {"max_iterations":  max_iterations, 'c1': c1, 'c2': c2, 'delta': args.delta}, args.model_path)




if __name__ == "__main__":

    args = parser.parse_args()

    NLP.tokenizer = custom_tokenizer(NLP)

    if args.verbose: print(args)
    if args.verbose: print("Loading data...")
    training = list()
    with open(args.training, "r", encoding="utf-8") as f:
        for line in f:
            training.extend(load_data_conll(line))
    train(training, args.max_iterations, args.c1, args.c2, args.encoder, args.window)
