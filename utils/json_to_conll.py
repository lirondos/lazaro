import sys
import json
#from crf import load_data, ENCODER_DICT
from utils2 import BMESEncoder, BILOUEncoder, BIOEncoder, BIOESEncoder, IOEncoder
from utils_review import custom_tokenizer
import argparse
import spacy
from typing import Mapping, Sequence, Dict, Optional, List, NamedTuple, Tuple, Counter, Iterable
from spacy.tokens import Span, Doc, Token
from spacy.language import Language
from spacy.tokenizer import Tokenizer

ENCODER_DICT = {"BIO": BIOEncoder(), "IO": IOEncoder(), "BILOU": BILOUEncoder(), "BMES": BMESEncoder(), "BIOES": BIOESEncoder()}

parser = argparse.ArgumentParser()

parser.add_argument('--json_file', type=str, help='Path to file with jsonl data')
parser.add_argument('--conll_file', type=str, help='Path where the CONLL will be stored')
parser.add_argument('--encoding', type=str, default='BIO', help='Encoding to use (BIO default)')
parser.add_argument('--include_others', type=bool, default=True, help='Whether to include OTHER tag')

args = parser.parse_args()
NLP = spacy.load('es_core_news_md', disable=["ner"])
NLP.tokenizer = custom_tokenizer(NLP)

def load_data(path, include_other):
    mylist = list()
    with open(path, encoding="utf8") as f:
        lines = f.readlines()
    for line in lines:
        try:
            json_as_dict = json.loads(line)
            doc = ingest_json_document(json_as_dict, NLP, include_other)
            for sent in doc.sents:
                tokens = list(sent)
                encoded_labels = encoder.encode(tokens)
                with open(path_to_conll, "a", encoding="utf-8") as f:
                    for token, label in list(zip(tokens, encoded_labels)):
                        f.write(token.text + " " + label + "\n")
                    f.write("\n")
        except:
            print("Error in: " + path)

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

def ingest_json_document(doc_json: Mapping, nlp: Language, include_other: bool) -> Doc:
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

if __name__ == "__main__":
    path_to_jsonl = args.json_file
    path_to_conll = args.conll_file
    encoding = args.encoding
    include_others = args.include_others

    encoder = ENCODER_DICT[encoding]

    load_data(path_to_jsonl, include_others)
