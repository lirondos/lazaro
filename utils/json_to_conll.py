import sys
import json
from baseline import load_data, ENCODER_DICT
from utils2 import BMESEncoder, BILOUEncoder, BIOEncoder, BIOESEncoder
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--json_file', type=str, help='Path to file with jsonl data')
parser.add_argument('--conll_file', type=str, help='Path where the CONLL will be stored')
parser.add_argument('--encoding', type=str, default='BIO', help='Encoding to use (BIO default)')
parser.add_argument('--include_others', type=bool, default=True, help='Whether to include OTHER tag')


args = parser.parse_args()
if __name__ == "__main__":
    path_to_jsonl = args.json_file
    path_to_conll = args.conll_file
    encoding = args.encoding
    include_others = args.include_others

    encoder = ENCODER_DICT[encoding]

    docs = load_data(path_to_jsonl, include_others)
    for doc in docs:
        for sent in doc.sents:
            tokens = list(sent)
            encoded_labels = encoder.encode(tokens)
            with open(path_to_conll, "a", encoding = "utf-8") as f:
                for token, label in list(zip(tokens, encoded_labels)):
                    f.write(token.text + " " + label + "\n")
                f.write("\n")