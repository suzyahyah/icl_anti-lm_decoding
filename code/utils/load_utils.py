#!/usr/bin/python3

import torch
import os
import re
import pathlib
import pandas as pd
import pickle
import re
import json
from omegaconf import OmegaConf
from code.datasets.data_utils import get_fn_dataset
from transformers.utils import WEIGHTS_NAME

def get_model_class(model_size, hack, args_model):
    if "trl" in args_model:
        # transformer reinforcement learning library
        print("Loading from Transformer RL Library")
        from trl import AutoModelForCausalLMWithValueHead
        clsName = AutoModelForCausalLMWithValueHead
        return clsName

    if "gptn" in model_size:
        from transformers import GPTNeoForCausalLM
        clsName = GPTNeoForCausalLM
        if hack:
            from code.model_hacks import gptneo_model_hack
            clsName = gptneo_model_hack.GPTNeoForCausalLMHack

    if "bloom" in model_size:
        from transformers import AutoModelForCausalLM
        clsName = AutoModelForCausalLM
        if hack:
            from code.model_hacks import bloom_model_hack
            clsName = bloom_model_hack.BloomForCausalLMHack

    if "xglm" in model_size:

        from transformers import  XGLMForCausalLM
        clsName = XGLMForCausalLM
        if hack:
            raise Exception("xglm hack not implemented")

    if "llama" in model_size:
        from transformers import LlamaForCausalLM
        clsName = LlamaForCausalLM

    print("Loading model class:", str(clsName.__class__))
    return clsName


def load_if_hack(clsName, og_fol, args_model):
    if args_model.hack:
        model = clsName.from_pretrained(og_fol, args_model)
        if save_fol != "":
            full_model_dict = model.state_dict()
            best_model_path = os.path.join(save_fol, WEIGHTS_NAME)
            masks_dict = torch.load(best_model_path)
            full_model_dict.update(masks_dict)
            model.load_state_dict(full_model_dict)
    else:
        model = clsName.from_pretrained(og_fol, torch_dtype=torch.bfloat16)
    return model

def get_models(model_size="gptn2.7B", save_fol="", args_model=None, hack=False, cuda=True):
    print(f"loading models from..{save_fol}")
    load_default = False
    clsName = get_model_class(model_size, args_model.layer_mask, args_model)

    if "gptn" in model_size:
        from transformers import GPT2Tokenizer
        size = model_size.replace("gptn", "")  
        og_fol = f"EleutherAI/gpt-neo-{size}"
        tokenizer = GPT2Tokenizer.from_pretrained(og_fol)
        # only for gptneo, not for xglm
        tokenizer.pad_token = tokenizer.eos_token
        model = load_if_hack(clsName, og_fol, args_model)

    elif "bloom" in model_size:
        from transformers import AutoTokenizer

        digit = re.search(r'\d', model_size).start()
        og_fol = f"bigscience/{model_size[:digit]}-{model_size[digit:]}"
        tokenizer = AutoTokenizer.from_pretrained(og_fol)
        model  = load_if_hack(clsName, og_fol, args_model) 

    elif "xglm" in model_size:
        from transformers import XGLMTokenizer
        size = model_size.replace("xglm","")
        og_fol = f"facebook/xglm-{size}"
        tokenizer = XGLMTokenizer.from_pretrained(og_fol)
        tokenizer.bos_token_id = tokenizer.eos_token_id
        model  = load_if_hack(clsName, og_fol, args_model) 
        # because they initialised with the eos token instead

    elif "t5" in model_size:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        size = model_size.replace("t5","")
        save_fol = f"t5-{size}"
        tokenizer = T5Tokenizer.from_pretrained(save_fol)
        if not hack:
            model = T5ForConditionalGeneration.from_pretrained(save_fol)
    elif "opt" in model_size:
        from transformers import GPT2Tokenizer, OPTForCausalLM
        size = model_size.replace("opt", "")
        save_fol = f"facebook/opt-{size}"
        tokenizer = GPT2Tokenizer.from_pretrained(save_fol)
        if not hack:
            model = OPTForCausalLM.from_pretrained(save_fol)
    elif "llama" in model_size:
        from transformers import AutoTokenizer
        # need to do several steps before we get here.
        og_fol = f"/exp/ssia/projects/llama/{model_size}"
        tokenizer = AutoTokenizer.from_pretrained(og_fol)
        tokenizer.pad_token = tokenizer.eos_token
        model = load_if_hack(clsName, og_fol, args_model)

    else:
        raise Exception("model not specified:", model_size)
        
    tokenizer.padding_side = "left"
    if cuda:
        model = model.cuda()
    print("loaded models..")
    return model, tokenizer


