#!/usr/bin/python3
# Author: Suzanna Sia
# pylint: disable=C0303,C0103

### Standard imports

### Custom imports
from code.datasets.data_utils import get_fn_dataset
from code.datasets.prompt_dataset import PromptsDataset      

from code.utils import load_utils

from code.decoder_hacks.anti_lm import AntiLMLogitsProcessor
from code.decoder_hacks.pmi import PMILogitsProcessor
from code.datasets.data_utils import CollateFn

from code.explore import logits
from code.explore import prompts_stats


def build_logits_processor(args_logits_p, model, tokenizer, prompt_ds=None):
    logits_processor_list = []

    if "AntiLM" in args_logits_p.name:
        # this includes both L1ReduceLogits and L1ReduceConcatLogits
        processor = AntiLMLogitsProcessor(args_logits_p, model, tokenizer, prompt_ds)

    elif "PMI" in args_logits_p.name:
        processor = PMILogitsProcessor(args_logits_p, model, tokenizer, prompt_ds=prompt_ds)

    elif args_logits_p.name == "DefaultLogits":
        return None
    else:
        # other options. subtract logits of L1, L2 delimiters

        raise Exception("logits processor not recognised:", args_logits_p.name)
    logits_processor_list.append(processor)
    return logits_processor_list



def build_model_tok(args, cfp, args_model, args_format):

    model, tokenizer = load_utils.get_models(args_model.model_size, 
                                             save_fol=args_model.save_fol,
                                             args_model=args_model,
                                             cuda=args_model.cuda,
                                             hack=args_model.layer_mask)

    if "prefix" in args_format.L2_delim.type:
        prefix_tokens = [f'[[{i}]]' for i in range(args_format.L2_delim.n_toks)] 
        tokenizer.add_special_tokens({"additional_special_tokens": prefix_tokens})
        model.resize_token_embeddings(len(tokenizer))
        print("Resizing and adding special tokens")

    if args_model.get("mask_row") is not None:
        if not "causal_mask" in args_model:
            print("Masking rows..", end=" ")
            rows = [args_model.get("mask_row")]

            if args_model.get("mask_from"):
                if "Bloom" in str(model.__class__):
                    nl = model.config.num_hidden_layers
                else:
                    nl = model.config.num_layers
                rows = list(range(args_model.get("mask_row"), nl))

            for row in rows:
                print(row, end=" ")
                if "Bloom" in str(model.__class__):
                    model.transformer.h[row].self_attention.head_mask.gate_values.fill_(0)

                elif "GPT" in str(model.__class__):
                    model.transformer.h[row].attn.attention.head_mask.gate_values.fill_(0)

                else:
                    raise Exception("not implemented for model")

    if "hf_trainer_cf" in args:
        if args.hf_trainer_cf.sharded_ddp == "":
            model = model.cuda()

    return model, tokenizer




def build_data_collator(args, tokenizer):
    if args.data_collator['type'] == "autoregressive_translation":
        data_collator = AutoRegressiveTranslationDC(tokenizer=tokenizer, mlm=False)
    elif args.data_collator['type'] == "autoregressive_lm":
        data_collator = AutoRegressiveLM_DC(tokenizer=tokenizer, mlm=False)

    else:
        raise Exception("not accounted for:", args.data_collator['type'])


    data_collator.my_collate_fn = CollateFn(tokenizer, cuda=False)
    return data_collator


def build_early_stop_callback(args):
    patience = args.early_stopping_cf.patience
    stopping_th = args.early_stopping_cf.threshold

    earlystop_cb = EarlyStoppingCallback(early_stopping_patience=patience,
                                         early_stopping_threshold=stopping_th) 
    return earlystop_cb


def build_datasets(args_data):
    test_name = "test"
    if args_data.dev:
        test_name = "train"


    ds_promptbank = get_fn_dataset(args_data.trainset, 
                                   'train', 
                                   args_data.direction, 
                                   data_path=args_data.train_data_fn)

    ds_test = get_fn_dataset(args_data.testset, 
                             test_name,
                             args_data.direction, 
                             data_path=args_data.test_data_fn)

    
    return ds_promptbank, ds_test

def build_prompt_dataset(args, format_cf, model, tokenizer, ds_promptbank, ds_test):
    # Different classes correspond to different ways of sampling

    args_d = args.data
    args_p = args.sample_prompts

    if args.sample_prompts.sampling_method == "random":
        prompts_ds_class = PromptsDataset
    else:
        raise Exception("Method not available:", args.sample_prompts.sampling_method)


    prompt_ds = prompts_ds_class(format_cf, ds_promptbank, ds_test, 
                                 seed=args.seed,
                                 **args_p,
                                 ntest=args_d.ntest,
                                 model=model,
                                 tokenizer=tokenizer)
    return prompt_ds

def do_analysis(args, cfp, model, tokenizer, dataloader, prompt_ds, collate_fn):
    if args.do_analysis == "":
        return

    # has not been tested in a while
    elif args.do_analysis == "logits":
        logits.get_logits(args, cfp, model, tokenizer, dataloader)

    elif args.do_analysis == "prompt_stats":
        prompts_stats.main(args, cfp, model, tokenizer, collate_fn, prompt_ds)
    else:
        raise Exception("Unrecognised analysis:", args.do_analysis)
