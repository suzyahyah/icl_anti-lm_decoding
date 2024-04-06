#!/usr/bin/python3
# Author: Suzanna Sia
# pylint: disable=C0303,C0103

### Local/Custom imports
from code.utils import utils, build, io_utils
from code.datasets.data_utils import CollateFn

### Standard imports
import os
import torch
import sys
import json
import argparse
from omegaconf import OmegaConf

### Third Party imports
from torch.utils.data import DataLoader
from transformers.utils import WEIGHTS_NAME


def run(args, cfp):

    args.format = utils.set_lang_delim_tokens(args.format,
                                              args.data.direction,
                                              args.model.model_size)

    ds_promptbank, ds_test = build.build_datasets(args.data)

    model, tokenizer = build.build_model_tok(args, cfp, args.model, args.format)
    model.training = False

    prompt_ds = build.build_prompt_dataset(args, args.format, 
                                           model, tokenizer, ds_promptbank, ds_test)


    logitsp = build.build_logits_processor(args.logitsp, model, tokenizer, prompt_ds)

    collate_fn = CollateFn(tokenizer)
    dataloader = DataLoader(prompt_ds,
                            collate_fn=collate_fn,
                            batch_size=args.generator.batch_size)

    build.do_analysis(args, cfp, model, tokenizer, dataloader, prompt_ds, collate_fn)

    all_gen_text = utils.gen_text(args.hf_generator_configs,
                                  args.format,
                                  model,
                                  tokenizer,
                                  dataloader,
                                  args=args,
                                  logitsp=logitsp)

    io_utils.save_test_prompts_used(args, cfp, prompt_ds) 
    io_utils.save_and_eval(cfp, args, all_gen_text)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--seed', default=0, type=int)
    argparser.add_argument('--do_analysis', default="")

    argparser.add_argument('--format_cf', default='configs/format/instr_L1L2.yaml')
    argparser.add_argument('--prompt_select_cf',
                            default='configs/prompt_select/random.yaml')

    argparser.add_argument('--data_cf', default='configs/data/default.yaml')
    argparser.add_argument('--generator_cf', default='configs/generator/default.yaml')
    argparser.add_argument('--logitsp_cf', default='configs/logits_processor/default.yaml')
    argparser.add_argument('--model_cf', default='configs/model/default.yaml')
    argparser.add_argument('--file_paths_cfg', default="")

    args, uk_args = argparser.parse_known_args()

    args = io_utils.merge_omegaconf_w_argparse(args, uk_args)
    cfp = OmegaConf.load(args.file_paths_cfg)
    print("save to:", cfp['gen_fn'].format(**args))
    run(args, cfp)
