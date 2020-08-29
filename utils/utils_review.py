import spacy
from spacy.tokens import Span, Doc, Token
from spacy.tokens import Span, Doc, Token
from spacy.language import Language
from spacy.tokenizer import Tokenizer
import re
from spacy.lang.tokenizer_exceptions import URL_PATTERN
import json



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


NLP = spacy.load('es_core_news_md', disable=["ner"])
NLP.tokenizer = custom_tokenizer(NLP)


def get_index(text, word): 
    doc = NLP(text)
    indices = list()
    for i, token in enumerate(doc):
        if word == token.text:
            indices.append(i)
    return indices
    
def get_token(text, index): 
    doc = NLP(text)
    return doc[index]

import json
def validate_json_file(path):
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                json_as_dict = json.loads(line)
                if len(json_as_dict["spans"]) != len(json_as_dict["labels"]):
                    print("Spans and labels mismatch in line:")
                    print(line)
            except ValueError as err:
                print(err)
                print(line)
