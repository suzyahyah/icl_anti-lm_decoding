#!/usr/bin/python3
# Author: Suzanna Sia

import string
import random
from torch.utils.data import Dataset

class PromptsDataset(Dataset):
    def __init__(self, decode_configs, 
                ds1, ds2, nprompts=5, seed=0, ntest=-1, shuffle_mode="",
                tokenizer=None, sample_on_new=False, sampling_method="random",
                filter_length=None, **kwargs):
        # take in two datasets, the first will be used to prompt, the second will be used to test
        super().__init__()
        self.nprompts = nprompts
        self.seed = seed
        self.printed = False
        self.set_data_keys(False)
        self.tokenizer = tokenizer 

        self.calc_token_count(ds1)
        self.calc_token_count(ds2)
        self.ds1 = ds1
        self.ds2 = ds2

        self.q = decode_configs['L1_delim']['value']
        self.a = decode_configs['L2_delim']['value']
        self.eos = decode_configs['eos']
        self.sep = decode_configs['sep']
        self.header = decode_configs['header']
        self.header_og = decode_configs['header_og']

        self.shuffle_mode = shuffle_mode 
        self.sample_on_new = sample_on_new # sample new prefix for every item
        self.sampling_method = sampling_method
        self.filter_length = filter_length
        self.prefix = self.get_prefix(self.ds1, seed=self.seed)
        self.ntest = ntest

    def calc_token_count(self, ds):
        ds.df['src_wc'] = ds.df['source'].apply(lambda x: len(x.split()))
        ds.df['target_wc'] = ds.df['target'].apply(lambda x: len(x.split()))
        ds.df['src_target_wc'] = ds.df['src_wc'] + ds.df['target_wc'] 

    def set_data_keys(self, align=True):
        # deprecated
        if align:
            self.skey = "source_align"
            self.tkey = "target_align"
        else:
            self.skey = "source"
            self.tkey = "target"

    def get_source(self, ds):
        seed = self.seed
        vals = ds.df.sample(n=self.nprompts, random_state=seed)[self.skey]
        vals = vals.values
        return " ".join(vals)

    def get_target(self, ds):
        seed = self.seed
        vals = ds.df.sample(n=20, random_state=seed)[self.tkey]
        vals = vals.values
        return ". ".join(vals)

    def targetlang_prefix(self, ds, prefix):
        # variants to try
        # 1. with intermediate instruction
        # 2. with QA in front of target
        # 3. with intermediate instructions
        #
        targets = ds.df.sample(20, random_state=self.seed+1)['target'].values
        targets = (self.sep).join([self.a+s.strip() for s in targets])
        prefix = targets + prefix

        return prefix
    

    def get_vals(self, i, query=""):
        ds = self.ds1
        if self.filter_length is not None:
            # sanity check that length filter goes here. And check the overall wordcount
            if self.sample_on_new:
                vals = ds.df.sample(n=self.nprompts)
            else:
                vals = ds.df.sample(n=self.nprompts, random_state=self.seed)
            vals = vals[[self.skey, self.tkey]]
        else:
            if self.sample_on_new:
                vals = ds.df.sample(n=self.nprompts) #, random_state=self.seed)
            else:
                vals = ds.df.sample(n=self.nprompts, random_state=self.seed)
            vals = vals[[self.skey, self.tkey]]
        return vals

    def get_prefix(self, ds, seed=None, compress=False, source_only=False, target_only=False, i=0):
        # we only need to get the prefix once.
        nprompts = self.nprompts
        if compress:
            seed = seed + 1
            nprompts = 15

        if source_only:
            vals = ds.df.sample(n=self.nprompts, random_state=seed)[[self.skey, self.skey]]
        elif target_only:
            vals = ds.df.sample(n=self.nprompts, random_state=seed)[[self.tkey, self.tkey]]
        else:
            # every class has their own customised get_vals()
            vals = self.get_vals(i, query="")

        vals = self._replacerepeat(ds, vals)
        vals = vals.values
        vals = self._shufflevariants(vals)
        vals = [(v[0].strip(), v[1].strip()) for v in vals]

        prefix = f"{self.sep}".join([f"{self.q}{v[0]}{self.a}{v[1]} " for v in vals])
        if prefix == "":
            prefix = f"{self.header}{prefix}"
        else:
            prefix = f"{self.header}{self.sep}{prefix}"
        return prefix

    def _replacerepeat(self, ds, vals):

        table = str.maketrans({key: " {0}".format(key) for key in string.punctuation})
        table2 = str.maketrans("", "", string.punctuation) 
        sstr = lambda x: x.lower().translate(table).translate(table2).strip()

        for i in range(len(vals)):
            if sstr(vals.iloc[i][self.skey]) == sstr(vals.iloc[i][self.tkey]):
                # print("REPLACEMENT:", vals.iloc[i])
                nval = ds.df.sample(n=1, random_state=self.seed+1)[[self.skey, self.tkey]]
                vals.iloc[i][self.skey] = nval[self.skey].values[0]
                vals.iloc[i][self.tkey] = nval[self.tkey].values[0]

        return vals

    def _shufflevariants(self, vals):

        if "word" in self.shuffle_mode:
            for i, v in enumerate(vals):
                newt = v[1].replace(".","").split()
                random.shuffle(newt)
                vals[i][1] = " ".join(newt)+"."

        if "sentence" in self.shuffle_mode:
            targets = [v[1] for v in vals]
            random.shuffle(targets)
            for i, v in enumerate(vals):
                vals[i][1] = targets[i]

        if "shift_targets_left" in self.shuffle_mode:
            targets = [v[1] for v in vals]
            targets = targets[1:] + [targets[0]]
            for i, v  in enumerate(vals):
                vals[i][1] = targets[i]

        if "shift_sources_left" in self.shuffle_mode:
            sources = [v[0] for v in vals]
            sources = sources[1:] + [sources[0]]
            for i, v  in enumerate(vals):
                vals[i][0] = sources[i]

        if "order" in self.shuffle_mode:
            vals2 = [":::".join(v) for v in vals]
            random.shuffle(vals2)
            vals = [v.split(":::") for v in vals2]


        if "all" in self.shuffle_mode:
            # shuffle all targetshuffletaset(decode_configs, ds1, ds2,
            new_vals = self.ds1.df.sample(n=self.nprompts, random_state=(self.seed+1))[[self.skey, self.tkey]].values
            targets = [v[1] for v in new_vals]
            for i, v in enumerate(vals):
                vals[i][1] = targets[i]

        return vals

    def __getitem__(self, i):

        # if sample prefix, we keep sampling a new one everytime theres a new item.
        if self.sample_on_new:
            self.prefix = self.get_prefix(self.ds1, seed=self.seed, i=i)


        source = self.ds2[i][self.skey]
        if source.strip()[-1] not in ['!', '.', '?']:
            source = source + "."
        
        if type(self.a) == str:
            query = f"{self.q}{source}{self.a}" 
        else:
            query = f"{self.q}{source}{self.a[0]}" 

        # inference time input
        total_input = f"{self.prefix}{self.sep}{query}"

        # training time input
        total_input2 = f"{total_input}{self.ds2[i]['target']}{self.eos}"

        if not self.printed:
            print(total_input)
            self.printed = True

        # typically built with input_ids
        item = {"id": self.ds2[i]['id'],
                "instructions": self.header,
                "input": total_input,
                "input2": total_input2,
                "prompt": self.prefix,
                "query": query,
                "query_raw": source,
                "target": self.ds2[i]['target'] + self.eos}


        return item 

    def __len__(self):
        if self.ntest==-1:
            return len(self.ds2)
        else:
            return min(len(self.ds2), self.ntest)

