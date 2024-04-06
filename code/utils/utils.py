#!/usr/bin/python3
# Author: Suzanna Sia

# Standard imports
import pandas as pd

import torch
from tqdm import tqdm


@torch.no_grad()
def gen_text(dc_cfg, format_cf, model, tokenizer, dataloader, logitsp, args=None):

    all_gen_text = []
    all_ids = []

    for j, batch in enumerate(tqdm(dataloader)):
        all_ids.append(batch['ids'])
        # the max position embeddings for gptneo is 2048
        max_cutoff = 2000

        if batch['input_ids'].shape[1] > max_cutoff:
            print(f"max cutoff at batch {j}")
            batch['input_ids'] = batch['input_ids'][:, -max_cutoff:]
            batch['input_mask'] = batch['input_mask'][:, -max_cutoff:]

        max_len = batch['input_ids'].shape[1] + batch['query_ids'].shape[1]*2 # longer for cn
        max_len = min(2045, max_len)

        sep_ = format_cf['sep']
        l1_delim = format_cf['L1_delim']['value']
        l2_delim = format_cf['L2_delim']['value']

        trainable_prefix = tokenizer.additional_special_tokens
        bad_words_ids = None
        if len(trainable_prefix) > 0:
            bad_words_ids = tokenizer.batch_encode_plus(trainable_prefix)['input_ids']
            if "XGLM" in type(model).__name__:
                bad_words_ids = [id[1:] for id in bad_words_ids]

        outputs = model.generate(batch['input_ids'],
                                 bad_words_ids=bad_words_ids,
                                 use_cache=True,
                                 attention_mask=batch['input_mask'],
                                 logits_processor=logitsp,
                                 max_length=max_len,
                                 pad_token_id=tokenizer.pad_token_id,
                                 return_dict_in_generate=True,
                                 output_scores=True,
                                 **dc_cfg)

        gen_ids = outputs.sequences

        # later versions of huggingface dont need to find start_ix
        start_ix = batch['input_ids'].shape[1]
        gen_text = tokenizer.batch_decode(gen_ids[:, start_ix:])
        gen_text = [t[:t.find(sep_)+1] if sep_ in t else t for t in gen_text]

        if l1_delim.strip() != "":
            gen_text = [t.split(l1_delim)[0].strip().replace(sep_, "") for t in gen_text]

        if type(l2_delim) != str:
            l2_delim = l2_delim[0]

        if l2_delim.strip() != "":
            gen_text = [t.split(l2_delim)[0].strip().replace(sep_, "") for t in gen_text]

        # we always need to remove newline carriage otherwise the generated text file will
        # screw up badly. 
        gen_text = [t.replace("\n", "") for t in gen_text]
        gen_text = [t.replace("</s>", "") for t in gen_text]
        gen_text = [t.replace("<pad>", "") for t in gen_text]

        if j == 0:
            print(tokenizer.decode(batch['input_ids'][0]))
            print("\n====")
            print(gen_text)

        for i in range(len(gen_text)):
            all_gen_text.append({"id": batch['ids'][i], "gen_text": gen_text[i]})

    return all_gen_text 




def get_lang_from_langcodes(lang, lang_dict):
    if len(lang) > 2:
        key = "FLORES101-code"
    else:
        key = "MM100-code"
    lang = lang_dict[lang_dict[key]==lang]['language'].values[0]
    return lang


def set_lang_delim_tokens(decode_configs, direction, model_size): #, prefix):
    # either set as "English" "French" or "[0]" for special prefix

    lang_dict = pd.read_csv("assets/flores_map.csv", sep="\t")
    L1, L2 = direction.split('-')

    L1 = get_lang_from_langcodes(L1, lang_dict)
    L2 = get_lang_from_langcodes(L2, lang_dict)

    decode_configs['header_og'] = decode_configs['header']

    decode_configs['header'] = decode_configs['header'].replace("<L1>", L1)
    decode_configs['header'] = decode_configs['header'].replace("<L2>", L2)

    decode_configs['L1_delim']['value'] = decode_configs.L1_delim.value.replace("<L1>", L1)

    # we only modify L2
    if decode_configs.L2_delim.type == "string":
        decode_configs['L2_delim']['value'] = decode_configs.L2_delim.value.replace("<L2>", L2)

    elif decode_configs.L2_delim.type == "prefix":
        prefix_tokens = [f'[[{i}]]' for i in range(decode_configs.L2_delim.n_toks)]
        prefix = "".join(prefix_tokens).strip()
        decode_configs['L2_delim']['value'] = decode_configs.L2_delim.value.replace("<L2>", prefix)

    elif decode_configs.L2_delim.type == "prefix_surround":
        # surround the word French with special separators
        value = decode_configs.L2_delim.value.replace("<L2>", L2)
        if decode_configs.L2_delim.n_toks == 2:
            value = prefix_tokens[0] + value + prefix_tokens[1]
        elif decode_configs.L2_delim.n_toks == 1:
            value = prefix_tokens[0] + value
        else:
            raise Exception("not defined")
        decode_configs['L2_delim']['value'] = value
    else:
        raise Exception("not recognised delim type")

    if "xglm" in model_size:
        decode_configs['sep'] = "</s>"

    print(f"L1_delim:", decode_configs[f'L1_delim']) 
    print(f"L2_delim:", decode_configs[f'L2_delim']) 
    return decode_configs



