#!/usr/bin/python3
# Author: Suzanna Sia

import string
table = str.maketrans("", "", string.punctuation)

def tokenize_text(text, tokenizer=None):
    # lower case, remove punct and tokenize.
    # used for word overlap in bm25 and submodopt

    if tokenizer is None:
        # do not remove stopwords 
        text = text.lower()
        text = text.translate(table) 
        text = text.split()
        return text
    else:
        raise Exception("havent implemented another tokenizer")
