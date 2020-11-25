

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
import pandas as pd
import sys
sys.path.append("/home/ealvarezmellado/lazaro/utils/")
from utils2 import BiasFeature, TokenFeature, UppercaseFeature, TitlecaseFeature, TrigramFeature, QuotationFeature, WordEnding, POStagFeature, WordVectorFeature, WordShapeFeature, WordVectorFeatureSpacy, BigramFeature, IsInDict, GraphotacticFeature, LemmaFeature, DigitFeature, PunctuationFeature, WordVectorFeatureNerpy, WordProbability, WordVectorFeatureNorm, SentencePositionFeature, BrownClusterFeature, HigherEnglishProbability, QuatrigramFeature, AllCapsFeature, PerplexityFeature, URLFeature, EmailFeature, TwitterFeature
from utils2 import WindowedTokenFeatureExtractor, CRFsuiteEntityRecognizer, BILOUEncoder, BIOEncoder, IOEncoder, ScoringCounts, ScoringEntity, BMESEncoder, BIOESEncoder
from constants import ANGLICISM_INDEX, TO_BE_TWEETED_PATTERN, AUTOMATICALLY_ANNOTATED_FOLDER, TO_BE_PREDICTED_FOLDER, CORPUS
from utils import PUNC_REPEAT_RE, DIGIT_RE, UPPERCASE_RE, LOWERCASE_RE
from utils import PRF1
from secret import MY_HOST, MY_USERNAME, MY_PASS, MY_DB
import mysql.connector


KFOLD = 10
NLP = spacy.load('es_core_news_md', disable=["ner"])
TODAY = datetime.now(timezone.utc).strftime('%d%m%Y')


parser = argparse.ArgumentParser()


parser.add_argument('--train_folder', type=str, help='Path to file with training data', default=CORPUS)
parser.add_argument('--val_folder', type=str, help='Path to file with validation data', default=None)
parser.add_argument('--cross_validation', type=bool, help='Path to file with validation data', default=False)
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
parser.add_argument('--expanded_features', action='store_true', default=False, help = 'Include expanded features (default False)')



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
    with open(path, encoding="utf8") as f:
        lines = f.readlines()
    for line in lines:
        try:
            json_as_dict = json.loads(line)
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

def train_predict(train_set, test_set, max_iterations, c1, c2, encoder, window_size) -> None:
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
    crf.train(train_set, "lbfgs", {"max_iterations":  max_iterations, 'c1': c1, 'c2': c2, 'delta': args.delta}, "tmp.model")

    if args.verbose: print("Predicting...")
    if args.val_folder or args.cross_validation:
        test_set = copy.deepcopy(test_set)
        for doc in test_set:
            doc.ents = []
    predicted = [crf(doc) for doc in test_set]
    return predicted
    
def write_to_db(mydb, ent, label, context, newspaper, url, date, categoria,start, end):
    
    mycursor = mydb.cursor()
    date_object = datetime.strptime(date, '%A, %d %B %Y').date()
    date_str = date_object.strftime('%Y-%m-%d')
    sql = "INSERT INTO t_anglicisms (borrowing,lang,context,newspaper,url,date,section,start_token,end_token, new_date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    val = (ent, label, context, newspaper, url, date, categoria,start, end, date_str)
    mycursor.execute(sql, val)

    mydb.commit()
    
def connect_to_db():
    mydb = mysql.connector.connect(host=MY_HOST,user=MY_USERNAME,password=MY_PASS,database=MY_DB)
    return mydb
    
def write_predictions(predicted_docs):
    anglicism_pd = pd.read_csv(ANGLICISM_INDEX, error_bad_lines=False)
    try:
        mydb = connect_to_db()
    except Exception as e:
        print(e)
    with open(TO_BE_TWEETED_PATTERN + TODAY +'.csv', 'a', encoding = "utf-8", newline='') as tobetweeted, open(ANGLICISM_INDEX, 'a', encoding="utf-8", newline='') as anglicism_index:
        anglicism_index_writer = csv.writer(anglicism_index, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        tobetweeted_writer = csv.writer(tobetweeted, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        tobetweeted_writer.writerow(["borrowing", "lang", "context", "newspaper", "url", "date", "categoria"])
        anglicisms_of_the_day = defaultdict(int)
        for mydoc in predicted_docs:
            myents = list()
            myspans = list()
            for ent in mydoc.ents:
                myents.append([ent.start, ent.end, ent.label_])
                myspans.append(ent.text)
                if ent.start < 15:
                    context = mydoc[0:ent.end + 15].text
                else:
                    context = mydoc[ent.start - 15:ent.end + 15].text
                context = context.replace("\n", ". ")
                anglicism_index_writer.writerow([ent.text.lower(), ent.label_, context, mydoc.user_data["newspaper"], mydoc.user_data["url"], mydoc.user_data["date"], mydoc.user_data["categoria"],ent.start, ent.end])
                try:
                    write_to_db(mydb, ent.text, ent.label_, context, mydoc.user_data["newspaper"], mydoc.user_data["url"], mydoc.user_data["date"], mydoc.user_data["categoria"],ent.start, ent.end)
                except Exception as e:
                    print(e)
                seriesObj = anglicism_pd.apply(lambda x: True if x['borrowing'] == ent.text.lower() else False, axis=1)
                times_appeared_prev = len(seriesObj[seriesObj == True].index)
                if times_appeared_prev == 1:
                    context = context.replace("\'", "")
                    context = context.replace("\"", "")
                    context = context.replace("“", "")
                    context = context.replace("”", "")
                    context = context.replace("‘", "")
                    context = context.replace("’", "")
                    tobetweeted_writer.writerow([ent.text, ent.label_, context, mydoc.user_data["newspaper"], mydoc.user_data["url"],mydoc.user_data["date"], mydoc.user_data["categoria"]])
                elif times_appeared_prev == 0 and anglicisms_of_the_day[ent.text.lower()] == 1:
                    tobetweeted_writer.writerow([ent.text, ent.label_, context, mydoc.user_data["newspaper"], mydoc.user_data["url"], mydoc.user_data["date"], mydoc.user_data["categoria"]])
                anglicisms_of_the_day[ent.text.lower()] += 1
            mydict = {"text": mydoc.text, "date": mydoc.user_data["date"], "annotation_approver": "lazaro", "newspaper": mydoc.user_data["newspaper"], "categoria": mydoc.user_data["categoria"], "url": mydoc.user_data["url"], "labels": myents, "spans": myspans}
            with open(AUTOMATICALLY_ANNOTATED_FOLDER + TODAY + '.jsonl', 'a', encoding = "utf-8") as f:
                f.write(json.dumps(mydict)+'\n')



if __name__ == "__main__":

    args = parser.parse_args()

    NLP.tokenizer = custom_tokenizer(NLP)

    if args.verbose: print(args)
    if args.verbose: print("Loading data...")
    training = list()
    test = list()
    for file in os.listdir(args.train_folder):
        if os.path.isfile(os.path.join(args.train_folder, file)):
            training.extend(load_data(os.path.join(args.train_folder, file), args.include_other))
    if not args.val_folder and not args.cross_validation:# we are  in test
        path_to_test = TO_BE_PREDICTED_FOLDER + TODAY + "/"
        for file in os.listdir(path_to_test):
            if os.path.isfile(os.path.join(path_to_test, file)):
                test.extend(load_data(os.path.join(path_to_test, file), args.include_other, is_predict = True))
        predicted_docs = train_predict(training, test, args.max_iterations, args.c1, args.c2, args.encoder, args.window)
        write_predictions(predicted_docs)
    elif args.val_folder: # there is validation folder
        for file in os.listdir(args.val_folder):
            if os.path.isfile(os.path.join(args.val_folder, file)):
                test.extend(load_data(os.path.join(args.val_folder, file), args.include_other))
        predicted_docs = train_predict(training, test, args.max_iterations, args.c1, args.c2, args.encoder, args.window)
        prf1, predictions = evaluate(predicted_docs, test)
        print(predictions)
        print_results(prf1)
    elif args.cross_validation: # cross validation
        for i in range(KFOLD): # 10 k fold
            instances_no = len(training) // KFOLD
            test_cv = training[instances_no*i: instances_no*i+ instances_no]
            train_cv = training[0:instances_no*i] + training[instances_no*i+ instances_no:]
            predicted_docs = train_predict(train_cv, test_cv, args.max_iterations, args.c1, args.c2, args.encoder,
                                           args.window)
            prf1, predictions = evaluate(predicted_docs, test_cv)
            print(test_cv)
            print(predictions)
            print_results(prf1)
            print("####################")