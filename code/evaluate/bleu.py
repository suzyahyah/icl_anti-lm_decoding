#!/usr/bin/python3
# Author: Suzanna Sia


import numpy as np
import json
import os
import pandas as pd 
from langdetect import detect
import string
import pathlib

import warnings
warnings.filterwarnings("ignore")

from sacrebleu.metrics import BLEU

table = str.maketrans({key: " {0}".format(key) for key in string.punctuation})
table2 = str.maketrans("", "", string.punctuation)

rd = lambda x: np.around(x, 3)
sstr = lambda x: x.lower().translate(table).translate(table2).strip()


def evaluate_sacrebleu(hyp_list, ref_list, bleu_c, bleu_s):

    rref = [ref_list]
    hyp_list_ = []
    for h in hyp_list:
        if type(h) == str:
            hyp_list_.append(h)
        else:
            hyp_list_.append("blank")
    hyp_list = hyp_list_

    hyp_list = [h.replace("</s>", "") for h in hyp_list]
    hyp_list = [h.replace("<pad>", "") for h in hyp_list]
    bleu_corpus = bleu_c.corpus_score(hyp_list, rref)
    scores = score_helper(bleu_corpus)
    scores['tokenizer'] = bleu_c.tokenizer.__class__.__name__.replace("Tokenizer", "")

    all_dp = []
    for i in range(len(hyp_list)):
        bleu_sentence = bleu_s.sentence_score(hyp_list[i], [rref[0][i]])
        dp = score_helper(bleu_sentence, name='sb')
        # bad try except statement 
        try:
            dp['lang'] = detect(hyp_list[i])
        except:
            dp['lang'] = "NA"
        all_dp.append(dp)

    all_dp_df = pd.DataFrame(all_dp)
    return scores, all_dp_df

def score_helper(bleu_object, name='bleu'):
    scores = {}
    scores[name] = rd(bleu_object.score)
    scores[f'{name}1'] = rd(bleu_object.precisions[0])
    scores[f'{name}2'] = rd(bleu_object.precisions[1])
    scores[f'{name}3'] = rd(bleu_object.precisions[2])
    scores[f'{name}4'] = rd(bleu_object.precisions[3])
    scores['brevity_penalty'] = rd(bleu_object.bp)
    return scores

def guess_tokenizer(hyp_fn):
    if ("-ja." in hyp_fn) or ("-jpn." in hyp_fn):
        tokenizer = "ja-mecab"
    if "-zh" in hyp_fn:
        tokenizer = "zh"

    else:
        tokenizer = None

    return tokenizer

def main(hyp_fn, ref_fn):
    ref_df = pd.read_csv(ref_fn, sep="\t")
    hyp_df = pd.read_csv(hyp_fn, sep="\t")
    merge_df = ref_df.merge(hyp_df, on="id")

    tokenizer = guess_tokenizer(hyp_fn)

    bleu_c = BLEU(tokenize=tokenizer, lowercase=True)
    bleu_s = BLEU(tokenize=tokenizer, effective_order=True, lowercase=True)

    all_doc_scores = []
    all_doc_dp_df = []

    if "doc_id" in merge_df.columns:

        for k, grp in merge_df.groupby('doc_id'):
            grp['gen_text'] = grp['gen_text'].apply(lambda x: "<blank>" if type(x) != str else x)
            hyp_list = grp['gen_text'].values.tolist()
            ref_list = grp['target'].values.tolist()
            per_doc_scores, per_doc_dp_df = evaluate_sacrebleu(hyp_list, 
                                                               ref_list,
                                                               bleu_c, 
                                                               bleu_s)

            per_doc_scores['doc_id'] = str(grp['doc_id'].values[0])
            per_doc_dp_df['id'] = grp['id'].values

            all_doc_scores.append(per_doc_scores)
            all_doc_dp_df.append(per_doc_dp_df)
    else:
        if len(merge_df) == len(ref_df):
            hyp_list = merge_df['gen_text'].values.tolist()
            ref_list = merge_df['target'].values.tolist()
        else:
            print("WARNING: len(merge) != len(ref)")
            hyp_list = hyp_df['gen_text'].values.tolist()
            ref_list = ref_df['target'].values.tolist()

        per_doc_scores, per_doc_dp_df = evaluate_sacrebleu(hyp_list, 
                                                           ref_list, 
                                                           bleu_c, 
                                                           bleu_s)

        per_doc_dp_df['id'] = hyp_df['id']

        all_doc_scores.append(per_doc_scores)
        all_doc_dp_df.append(per_doc_dp_df)


    fn = (hyp_fn).replace("generated", "results").replace(".csv.hyp", ".json")
    pathlib.Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)

    with open(fn, 'w') as f:
        json.dump(all_doc_scores, f)

    print("Average doc score:", pd.DataFrame(all_doc_scores)['bleu'].mean())
    print("Doc level corpus scores saved to:", fn)

    # this remains the same

    fn = (hyp_fn).replace("generated", "results").replace(".txt.hyp", ".csv")
    pd.concat(all_doc_dp_df).to_csv(fn, index=False)
    print("Sentence scores saved to:", fn)


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--hyp_fn', type=str)
    argparser.add_argument('--ref_fn', type=str, default="")
    argparser.add_argument('--tokenizer', default=None, type=str)
    args = argparser.parse_args()

    main(args.hyp_fn, args.ref_fn)

