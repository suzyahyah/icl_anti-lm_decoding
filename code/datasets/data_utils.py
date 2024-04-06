#!/usr/bin/python3
# Author: Suzanna Sia
from code.datasets.bitext_dataset import *
import langid

def read_lines_strip(fn):
    if "EPARL" in fn:
        encoding = "ISO-8859-1"
    else:
        encoding = 'utf-8'

    print("reading:", fn)
    with open(fn, 'r', encoding=encoding) as f: 
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    return lines

def filter_long_sentences(par_lines, max_words=30, min_words=3):

    new_lines = []
    included = []
    for i, line in enumerate(par_lines):
        if len(line[1].split())>max_words or len(line[1].split())<min_words:
            continue
        else:
            new_lines.append(line)
            included.append(i)

    print(f"reduced lines from {len(par_lines)} tp {len(new_lines)}")
    return new_lines, included

def lang_check_(line, target_lang):

    lang, score = langid.classify(line)
    if lang != target_lang:
        return False
    else:
        return True


def process_and_save_df(df, save_fn, target_lang):

    df = df.dropna()
    for x in ['source', 'target']:
        df[x] = df[x].apply(lambda x: x.replace('"', '').strip())

    df['keep'] = df['target'].apply(lambda x: len(x.split()) > 3)
    df = df[df['keep']]
    df = df.drop('keep', axis=1)

    # this is the thing that takes super long 
    df['lang'] = df['target'].apply(lambda x: lang_check_(x, target_lang))
    df = df[df['lang']]

    # Process long sentences so that we don't accidentally run into OOM
    # filter by source sentence (english)
    df['short'] = df['source'].apply(lambda x: len(x.split()) < 25)
    df['long'] = df['source'].apply(lambda x: len(x.split()) >= 25)
    drop_columns = ['lang', 'short', 'long']
    df_short = df[df['short']].drop(drop_columns, axis=1)
    df_long = df[df['long']].drop(drop_columns, axis=1)

    df_short.to_csv(save_fn, sep="\t", index=False)
    df_long.to_csv(save_fn + ".long", sep="\t", index=False)

    print(f"Saved to {save_fn}: {len(df_short)}")
    print(f"Saved to {save_fn + '.long'} {len(df_long)}")

    return df

def add_line_id_to_doc_id(df):
    # temp hack to set line id
    doc_ids = df['doc_id']
    # set line_id
    prev_line = doc_ids.iloc[0]
    new_doc_lines = []
    line_ids = [0]
    k = 1

    for line in doc_ids.iloc[1:]:
        if prev_line == line:
            line_ids.append(k)
            k += 1
            continue
        else:
            new_doc_lines.append(line)
            prev_line = line
            line_ids.append(0)
            k = 1

    df['line_id'] = line_ids
    df['id'] = df['doc_id'].astype(str) + "_" + df['line_id'].astype(str)
    return df



def process_and_save_text(fn, mode, target_lang="fr"):
    with open(fn, 'r') as f:
        data = f.readlines()

    data_short = [d.strip() for d in data if len(d.split()) > 3 and len(d.split())<25]
    data_long = [d.strip() for d in data if len(d.split()) > 3 and len(d.split())>=25]

    data_short = [d for d in data_short if lang_check_(d, target_lang)] #, filter=True)
    data_long = [d for d in data_long if lang_check_(d, target_lang)] #, filter=True)

    save_fn = fn.replace(".raw", "")

    with open(save_fn, 'w') as f:
        f.write("\n".join(data_short))

    with open(save_fn + ".long", "w") as f:
        f.write("\n".join(data_long))

    print(f"Saved to {save_fn}", len(data_short))
    print(f"Saved to {save_fn+'.long'}", len(data_long))


def remove_broke_lines(fn, out_fn):
    all_lines = []
    with open(fn, 'r') as f:
        data = f.readlines()

    for i, line in enumerate(data):
        line = line.strip().split("\t")
        if len(line[1].strip().split())*3 < len(line[2].strip().split()):
            continue
        if len(line[1].strip().split()) > len(line[2].strip().split())*3 :
            continue
        all_lines.append(data[i].strip().replace('"',""))

    with open(out_fn, 'w') as f:
        f.write("\n".join(all_lines))
    print(f"written to:{out_fn}, {len(all_lines)}")

def get_fn_dataset(dataname, mode, direction="en-fr", data_path="data"):
    if dataname == "FLORES":

        # flores has no train set, so we sample prompts from the dev set. 
        if mode == "train": 
            mode = "dev"
        if mode == "test":
            mode = "devtest"

        dataset = FLORESdataset(mode, direction)
        print(f"loading flores dataset: {mode} {direction}")
        return  dataset

    else:
        raise Exception(dataname, "not recognised")

class BitextCollateFn:
    def __init__(self, tokenizer, cuda=True):
        self.tokenizer = tokenizer
        self.use_cuda = cuda

    def __call__(self, batch):
        sources = [b['source'] for b in batch]
        encoded =  self.tokenizer.batch_encode_plus(sources, padding=True, return_tensors='pt')

        batch = {}
        batch['input_ids'] = encoded['input_ids'].cuda()
        batch['attention_mask'] = encoded['attention_mask'].cuda()

        return batch


class CollateFn:
    def __init__(self, tokenizer, cuda=True):
        self.tokenizer = tokenizer
        #self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_cuda = cuda

    def get_encoding_length(self, sequence):
        encoded = self.tokenizer.batch_encode_plus(sequence, padding=True, return_tensors='pt')
        enc_ids = encoded['input_ids']
        enc_mask = encoded['attention_mask']
        if self.use_cuda:
            enc_ids = enc_ids.cuda()
            enc_mask = enc_mask.cuda()
        return enc_ids, enc_mask

    def __call__(self, batch):

        inputs = [b['input'] for b in batch]
        instructions = [b['instructions'] for b in batch]
        inputs2 = [b['input2'] for b in batch]
        prompts = [b['prompt'] for b in batch]
        targets = [b['target'] for b in batch]
        queries = [b['query'] for b in batch]
        queries_raw = [b['query_raw'] for b in batch]

        ids = [b['id'] for b in batch]

        input_ids, input_mask = self.get_encoding_length(inputs)
        input2_ids, input2_mask = self.get_encoding_length(inputs2)
        prompt_ids, prompt_mask = self.get_encoding_length(prompts)
        instructions_ids, instructions_mask = self.get_encoding_length(instructions)
        target_ids, target_mask = self.get_encoding_length(targets)
        query_ids, query_mask = self.get_encoding_length(queries)
        query_raw_ids, query_raw_mask = self.get_encoding_length(queries_raw)

        input_len = [len(self.tokenizer.encode(b['input'])) for b in batch]
        input2_len = [len(self.tokenizer.encode(b['input2'])) for b in batch]
        instructions_len = [len(self.tokenizer.encode(b['instructions'])) for b in batch]
        prompt_len = [len(self.tokenizer.encode(b['prompt'])) for b in batch]
        target_len = [len(self.tokenizer.encode(b['target'])) for b in batch]
        query_len = [len(self.tokenizer.encode(b['query'])) for b in batch]


        return {"ids":ids, 
                "inputs":inputs,
                "input_ids": input_ids, 
                "input_mask": input_mask,
                "input_len": input_len,

                "inputs2":inputs2,
                "input2_ids": input2_ids, 
                "input2_mask": input2_mask,
                "input2_len": input2_len,

                "instructions": instructions,
                "instructions_ids": instructions_ids,
                "instructions_len": instructions_len,
                "instructions_mask": instructions_mask,
 
                "prompts": prompts,
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "prompt_len": prompt_len,

                "targets": targets,
                "target_ids": target_ids,
                "target_mask": target_mask,
                "target_len": target_len,
               
                "queries": queries,
                "query_ids": query_ids,
                "query_mask": query_mask,
                "query_len": query_len,
                "queries_raw": queries_raw,
                "query_raw_ids": query_raw_ids}

