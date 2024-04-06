#!/usr/bin/python3
# Author: Suzanna Sia

import pandas as pd
import os
from code.datasets import data_utils
from torch.utils.data import Dataset

class BitextDataset(Dataset):

    def swap_col_names(self, df, direction):

        if direction.split('-')[0] == "en":
            pass
        elif direction.split('-')[1] == "en":
            df[['target', 'source']] = df[['source', 'target']]
        else:
            raise Exception("invalid direction")
        return df

    def __getitem__(self, i):
        item = self.df.iloc[i]
        return item

    def update_df_with_generated(self, target_fn):
        data = pd.read_csv(target_fn, sep="\t")
        self.df['target_gen'] = data['gen_text']

    def construct_from_bitext(self, fn1, fn2):
        #FLORES only

        with open(fn1, 'r') as f:
            L1_data = f.readlines()

        with open(fn2, 'r') as f:
            L2_data = f.readlines()
        
        L1_data = [l1.strip() for l1 in L1_data]
        L2_data = [l2.strip() for l2 in L2_data]

        return L1_data, L2_data


    def filter_length(self, min_len=5, max_len=10): 
        self.df['short'] = self.df['source'].apply(lambda x: len(x.split()) > min_len and len(x.split()) < max_len)

        min_len, max_len = 15, 20
        self.df['med'] = self.df['source'].apply(lambda x: len(x.split()) > min_len and len(x.split()) < max_len)

        min_len, max_len = 25, 30
        self.df['long'] = self.df['source'].apply(lambda x: len(x.split()) > min_len and len(x.split()) < max_len)
        
        print("short:", f"{len(self.df[self.df['short']])}/{len(self.df)}")
        print("med:", f"{len(self.df[self.df['med']])}/{len(self.df)}")
        print("long:", f"{len(self.df[self.df['long']])}/{len(self.df)}")


    def __len__(self):
        return len(self.df)


class FLORESdataset(BitextDataset):
    def __init__(self, mode="", direction=""):
        super().__init__()
        self.direction = direction
        L1, L2 = direction.split('-')

        # lang conversion
        flores_map = pd.read_csv('assets/flores_map.csv', header=0, sep="\t")
        if len(L1) > 2:
            # already converted
            pass
        else:
            L1 = flores_map[flores_map['MM100-code'] == L1]['FLORES101-code'].values[0]
            L2 = flores_map[flores_map['MM100-code'] == L2]['FLORES101-code'].values[0]

        # we just work with dev set for flores
        fn1 = f"data/FLORES/flores101_dataset/{mode}/{L1}.{mode}"
        fn2 = f"data/FLORES/flores101_dataset/{mode}/{L2}.{mode}"

        L1_data, L2_data = self.construct_from_bitext(fn1, fn2)
        
        df = pd.DataFrame()
        df['source'] = L1_data
        df['target'] = L2_data

        df['id'] = list(range(len(L1_data)))
        self.df = df
        self.filter_length()
        print("FLORES len lines:", len(self.df))


class TEMPdataset(BitextDataset):
    def __init__(self, fn):
        super().__init__()
        self.df = pd.read_csv(fn, names=['id', 'source', 'target'], sep=",")
        self.df = self.df.dropna()
        print("len lines:", len(self.df))



