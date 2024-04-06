#!/usr/bin/python3
# Author: Suzanna Sia

from transformers import LogitsProcessor
import torch.nn.functional as F
import torch
import math
from scipy.spatial.distance import jensenshannon as jsd
import numpy as np

### Local/Custom imports
dt = lambda x: x.detach().cpu().double().numpy()

class AntiLMLogitsProcessor(LogitsProcessor):
    def __init__(self, args_logits_p, model, tokenizer, prompt_ds):
        super().__init__()

        self.name = args_logits_p.name
        self.alpha = args_logits_p.alpha # confusing. this refers to both alpha and gamma
        self.frequency = args_logits_p.frequency #{decay, once, always}
        self.contrast_logits_type = args_logits_p.contrast_logits_type
        self.contrast_scores = {}

        self.tokenizer = tokenizer
        self.model = model
        self.prompt_ds = prompt_ds
        self.prev_test_sentence = ""
        self.new_scores = None
        self.step = 0

        self.process_logits()

        print("Loaded Logits processor:", self.name, self.alpha)

    def compute_jsd(self, s1, s2):

        out1  = self.model(self.tokenizer.encode(s1, return_tensors="pt").cuda())
        out2  = self.model(self.tokenizer.encode(s2, return_tensors="pt").cuda())

        P = F.softmax(out1['logits'][:,-1,:])
        Q = F.softmax(out2['logits'][:,-1,:])

        M = (0.5 * (P + Q))

        p_loss = (P * (torch.log2(P) - torch.log2(M))).sum() 
        q_loss = (Q * (torch.log2(Q) - torch.log2(M))).sum()

        base_jsd = 0.5 * (p_loss + q_loss)
        return base_jsd



    def process_logits(self):
        if "u" in self.contrast_logits_type:
            header = self.prompt_ds.header
            input_ids = self.tokenizer.encode(header, return_tensors='pt')
            contrast_scores = self.model(input_ids.cuda())
            self.contrast_scores['u'] = contrast_scores.logits[:, -1, :]

        elif self.contrast_logits_type == "x":
            print("contrast logits source language (x); do nothing")


        else:
            raise Exception("contrast logits type not recognised:", self.contrast_logits_type)


    def compute_scores(self, scores, batch_len):
        if self.frequency == "once":
            pass

        elif self.frequency == "always":
            scores -= self.alpha * self.contrast_scores[self.contrast_logits_type]

        elif self.frequency == "decay":
            # keep the negative sign
            positive_alpha = self.alpha > 0
            alpha = self.alpha ** self.step

            if not positive_alpha:
                # check if positive then make it negative
                # otherwise we will be negating a negative value
                if alpha > 0:
                    alpha = -alpha

            scores -= alpha * self.contrast_scores[self.contrast_logits_type]
            # what do scores and self.contrast_scores look like actually

        elif self.frequency == "gompertz":
            # return -A * np.exp(-B * np.exp(-C * t)) + A
            B = 20
            C = 1 

            alpha = - self.alpha * np.exp(-B * np.exp(-C * self.step)) + self.alpha
            scores -= alpha * self.contrast_scores[self.contrast_logits_type]
            
        elif self.frequency == "logistic":
            # return -A / (1 + np.exp(-k*(t_values - t0))) + A

            alpha = -self.alpha / (1 + np.exp(-1 * (self.step - 10))) + self.alpha
            scores -= alpha * self.contrast_scores[self.contrast_logits_type]

        elif self.frequency == "relu":
            alpha = max(0, 0.3 - (self.step / 10) * 0.3)
            scores -= alpha * self.contrast_scores[self.contrast_logits_type]
            
        else:
            raise Exception("frequency option not recognised:", self.frequency)

        return scores

    def update_contrast_scores(self, decoded):

        if "u" in self.contrast_logits_type:
            pass


        if "x" in self.contrast_logits_type:
            L1_sentences = []
            check_n = self.prompt_ds.nprompts + 1
            q_delim = self.prompt_ds.q
            a_delim = self.prompt_ds.a
            instr = self.prompt_ds.header
            self.test_sentences = []

            for j in range(len(decoded)):

                test_sentence = decoded[j].split(q_delim)[check_n].split(a_delim)[0].strip()
                L1_sentences.append(test_sentence)
            outputs = self.tokenizer.batch_encode_plus(L1_sentences, 
                                                       return_tensors='pt',
                                                       padding=True)

            contrast_scores = self.model(input_ids=outputs['input_ids'].cuda(),
                                    attention_mask=outputs['attention_mask'].cuda())

            self.contrast_scores['x'] = contrast_scores.logits[:, -1, :]


    def __call__(self, input_ids, scores):  
        decoded = self.tokenizer.batch_decode(input_ids)

        q_delim = self.prompt_ds.q
        a_delim = self.prompt_ds.a

        check_n = self.prompt_ds.nprompts + 1
        sanity_test_sentence = decoded[0].split(q_delim)[check_n].split(a_delim)[0]
        batch_lens = [len(self.tokenizer(sent.split(q_delim)[check_n].split(a_delim)[0])['input_ids']) for sent in decoded]

        self.step += 1

        if sanity_test_sentence != self.prev_test_sentence:
            # new sentence, update contrast_scores
            self.step = 1 # reset decay step
            self.update_contrast_scores(decoded)
            self.prev_test_sentence = sanity_test_sentence

        return self.compute_scores(scores, batch_lens)

