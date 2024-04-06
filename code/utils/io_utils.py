#!/usr/bin/python3
# Author: Suzanna Sia

### Standard imports
import os
import pathlib
import pandas as pd
from omegaconf import OmegaConf
from code.evaluate import bleu as eval_bleu
#from code.evaluate import comet as eval_comet
# something wrong wiht my comet installation not going to bother

mkpath = lambda x: pathlib.Path(os.path.dirname(x)).mkdir(parents=True, exist_ok=True)


def save_test_prompts_used(args, cfg, prompt_ds):

    ds2 = prompt_ds.ds2
    args.data.source, args.data.target = args.data.direction.split('-')

    test_source_fn = cfg['test_source_fn'].format(**args)
    test_target_fn = cfg['test_target_fn'].format(**args)

    mkpath(test_source_fn)
    mkpath(test_target_fn)


    with open(test_source_fn, 'w') as f:
        f.write("\n".join(ds2.df['source']))
    with open(test_target_fn, 'w') as f:
        f.write("\n".join(ds2.df['target']))

    parallel_csv = cfg['test_parallel_fn'].format(**args)
    mkpath(parallel_csv)
    print("parallel csv:", parallel_csv)
    ds2.df.to_csv(parallel_csv, index=False, sep="\t", header=True)  
    # get vals for everything
    # save the vals in used_prompts_fn
    
    # save used prompts
    used_prompts_fn = cfg['used_prompts_fn'].format(**args)
    mkpath(used_prompts_fn)

    with open(used_prompts_fn, 'w') as f:
        f.write(prompt_ds[0]['prompt'])


def save_and_eval(cfp, args, all_gen_text):

    hyp_fn = cfp['gen_fn'].format(**(args))

    if args.data.ntest != -1:
        hyp_fn = hyp_fn + ".temp"

    if args.data.dev:
        hyp_fn = hyp_fn + ".dev"

    mkpath(hyp_fn) 
    pd.DataFrame(all_gen_text).to_csv(hyp_fn, index=False, sep="\t")

    eval_hyp(cfp, args)

def eval_hyp(cfp, args):
    # Construct path locations from settings
    hyp_fn = cfp['gen_fn'].format(**(args))
    if args.data.ntest != -1:
        hyp_fn = hyp_fn + ".temp"
    if args.data.dev:
        hyp_fn = hyp_fn + ".dev"
    ref_fn = cfp['test_parallel_fn'].format(**args)
    eval_bleu.main(hyp_fn, ref_fn)

    #metric_to_method = {
    #    "bleu": eval_bleu.main,
        #"comet": eval_comet.main,
    #}

    # Evaluate
    #for metric in args.metrics:
    #    method = metric_to_method.get(metric)
    #    if not method:
    #        raise ValueError(f"Metric {metric} not supported.")
    #    method(hyp_fn, ref_fn)


def merge_omegaconf_w_argparse(args, uk_args, verbose=True):
    uk_args_ = []

    for i, arg in enumerate(uk_args):
        if arg.startswith('--') and arg != "nproc_per_node":
            if not uk_args[i+1].startswith('--'):
                uk_args_.append(f"{arg.replace('--','')}={uk_args[i+1]}")
            else:
                raise ValueException("omegaconf argparse cannot handle store_value=True")

    # gather all configs from unknown args
    uk_args_ = OmegaConf.from_dotlist(uk_args_)

    config_files = []
    args = vars(args)

    # load the config files
    for key in args.keys():
        if key.endswith('_cf'):
            config_files.append(OmegaConf.load(args[key]))

    # merge all configs from config files and unknown args
    known_args = OmegaConf.merge(*config_files)
    merge_args = OmegaConf.merge(known_args, uk_args_)

    for k in args:
       merge_args[k] = args[k]

    merge_args.format.direction = merge_args.data.direction 
    merge_args.format.domain = merge_args.data.testset

    return merge_args


def get_default_argparser():
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', default=0)
    argparser.add_argument('--metric', default="bleu")

    argparser.add_argument('--data_cf', default='configs/data/default.yaml')
    argparser.add_argument('--prompt_select_cf',
                            default='configs/prompt_select/random.yaml')
    argparser.add_argument('--format_cf', default='configs/format/instr_L1L2.yaml')
    argparser.add_argument('--training_cf', default='configs/training/default.yaml')
    argparser.add_argument('--model_cf', default='configs/model/default.yaml')
    argparser.add_argument('--logitsp_cf', default='configs/logits_processor/default.yaml')
    argparser.add_argument('--generator_cf', default='configs/generator/default.yaml')
    argparser.add_argument('--file_paths_cfg', default="")

    return argparser
