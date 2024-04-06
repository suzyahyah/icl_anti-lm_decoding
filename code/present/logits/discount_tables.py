#!/usr/bin/python3
# Author: Suzanna Sia

from code.utils import io_utils
from omegaconf import OmegaConf
import os
import itertools
import json
import argparse
import pandas as pd
import numpy as np
import pathlib

mkpath = lambda x: pathlib.Path(os.path.dirname(x)).mkdir(parents=True, exist_ok=True) 

rd = lambda x: np.around(x, 1)

# Exp settings that we are looping over
settings = {"NPROMPTS": [0],
            "CONTRASTS":['x'],
            "SAMPLING": ['default'],
            "ALPHAS": [0.3],
            #"frequency": ['once',  "", 'decay', 'jsd'],
            "MODELS":  ['gptn2.7B', 'xglm2.9B', 'xglm7.5B', 'bloom3b', 'bloom7b1', 'llama7b', 'llama7b-chat'],
    #        "MODELS":  ['llama7b', 'llama7b-chat'],
            #"METRICS": ['bleu', 'empty']}
            "METRICS": ['bleu']}

# Other Possible flags: 
# python code/present/logits/main_tables.py --data.direction en-es
# python code/present/logits/main_tables.py --format_cf configs/format_instr_L2.yaml

def keep_row_condition(results):
    # keep condition
    condition1 = (results['alpha']==0.1) & (results['mode']=='pmi')
    condition2 = (results['alpha']==0.3) & (results['mode']=="anti_lm") & (results['model'].isin(['gptn2.7B', 'xglm2.9B', 'xglm7.5B', 'bloom3b', 'bloom7b1']))
    condition3 = (results['name'] == "default")
    condition4 = (results['alpha']==0.1) & (results['contrast']=="ux")
    condition5 = (results['alpha']==0.3) & (results['mode']=='anti_lm') & (results['model'].isin(['llama7b', 'llama7b-chat']))
    condition6 = (results['alpha']==0.3) & (results['mode']=="anti_lm_relu") & (results['model'].isin(['gptn2.7B', 'xglm2.9B', 'xglm7.5B', 'bloom3b', 'bloom7b1', "llama7b", "llama7b-chat"]))
    results = results[condition1 | condition2 | condition3 | condition4 | condition5 | condition6]
    return results

def get_default_argparser():
    # expose this function
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', default=0, type=int)
    argparser.add_argument('--format_cf', default='configs/format/instr_L1L2.yaml') 
    argparser.add_argument('--prompt_select_cf',
                            default='configs/prompt_select/random.yaml')
    argparser.add_argument('--data_cf', default='configs/data/default.yaml')
    argparser.add_argument('--generator_cf', default='configs/generator/default.yaml')
    argparser.add_argument('--logitsp_cf', default='configs/logits_processor/default.yaml')
    argparser.add_argument('--model_cf', default='configs/model/default.yaml')
    argparser.add_argument('--file_paths_cfg', default="configs/file_paths/logits.yaml")

    return argparser

def main():
    results = []
    #this corresponds to configs/logits_processor/{mode}.yaml
    MODES = ["default", "anti_lm", "anti_lm_relu", "anti_lm_logistic", "anti_lm_gompertz"]

    for mode in MODES:
        argparser = get_default_argparser()
        args, uk_args = argparser.parse_known_args()
        args.logitsp_cf = f"configs/logits_processor/{mode}.yaml"

        args = io_utils.merge_omegaconf_w_argparse(args, uk_args, verbose=False)

        cfp = OmegaConf.load(args.file_paths_cfg)

        combinations = list(itertools.product(*settings.values()))
        keys = list(settings.keys())

        # this is equivalennt to big nested for loop
        for exp_variables in combinations:
            # very brittle to ordering
            nprompt, contrast, sampling_type, alpha, model, metric = exp_variables

            args.sample_prompts.nprompts = nprompt
            args.logitsp.contrast_logits_type = contrast
            args.generator.name = sampling_type
            args.logitsp.alpha = alpha
            args.model.model_size = model

            gen_fn = cfp['gen_fn'].format(**args)
            res_fn = cfp['res_fn'].format(**args)
            res = {}
            res['mode'] = mode

            if os.path.exists(res_fn): # and os.path.exists(gen_fn):
                if metric == "bleu":
                    with open(res_fn, 'r') as f:
                        data = json.load(f)
                    res[metric] = rd(data[0][metric])

                elif metric == "empty":
                    
                    data = pd.read_csv(gen_fn, sep="\t")
                    data['gen_text'].fillna("", inplace=True)
                    data['gen_text'] = data['gen_text'].apply(lambda x: x.replace("<|endoftext|>", "").strip())
                    data['gen_text'] = data['gen_text'].apply(lambda x: x.replace("<", "").strip())
                    empty_gen_str = len(data[data['gen_text']==""])
                    res[metric] = np.around (100 * (empty_gen_str ) / len(data), 1)
                    
            else:
                #if "relu" in res_fn and "AntiLM_x" in res_fn:
                #    print(res_fn)
                continue

            if mode == "default":
                res['contrast'] = "NA"
                res['alpha'] = 0
                res['name'] = "default"

            else:
                res['contrast'] = contrast
                res['alpha'] = alpha
                #res['name'] = f"{mode}_{contrast}_{args.logitsp.frequency}"
                res['name'] = f"{mode}_{contrast}"

            res['sampling'] = sampling_type
            res['nprompt'] = nprompt
            res['model'] = model
            res['fn'] = res_fn
            results.append(res)

    results = pd.DataFrame(results)
    results = results.drop_duplicates()
    #results = keep_row_condition(results)

    # Printing Latex Table
    SAMPLING = settings['SAMPLING']
    EXPS = ['default', 'anti_lm_x', 'anti_lm_relu_x', 'anti_lm_logistic_x', 'anti_lm_gompertz_x']
    table_columns = list(itertools.product(SAMPLING, EXPS))

    gb_df = pd.pivot_table(results, index='model', columns=['sampling', 'name'], values=metric)
    gb_df = gb_df.reindex(settings['MODELS']) #, level=0)
    for col in table_columns:
        if col not in gb_df:
            gb_df[col] = "NA"

    gb_df = gb_df[table_columns]

    #print(gb_df[('default')].max(axis=1) - gb_df[('default','default')])
    #print(gb_df[('beamsearch')].max(axis=1) - gb_df[('beamsearch','default')])

    latex_save_fn = f'results/tables/logits/decay_table.{args.data.direction}.tex'
    mkpath(latex_save_fn)

    gb_df.to_latex(latex_save_fn)

    print(gb_df)
    print("saved to:", latex_save_fn)
    #metric_dfs.append(results)


if __name__ == "__main__":
    main()
