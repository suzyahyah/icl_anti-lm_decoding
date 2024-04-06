#!/usr/bin/python3
# Author: Suzanna Sia

from transformers import LogitsWarper

class PMILogitsProcessor(LogitsWarper):
    def __init__(self, args_logits_p, model, tokenizer, prompt_ds):
        super().__init__()

        self.name = args_logits_p.name
        self.alpha = args_logits_p.alpha
        self.contrast_logits_type = args_logits_p.contrast_logits_type

        self.model = model
        self.tokenizer = tokenizer
        self.prompt_ds = prompt_ds

        self.contrast_scores = 0

        
    def __call__(self, input_ids, scores):
        # not ncluding separator tokens

        decoded = self.tokenizer.batch_decode(input_ids)
        q_delim = self.prompt_ds.q
        a_delim = self.prompt_ds.a
        check_n = self.prompt_ds.nprompts + 1

        sentences = []
        for decode in decoded:
            if self.contrast_logits_type == "u":
                generated = decode.split(q_delim)[check_n].split(a_delim)[1]
                sentence = self.prompt_ds.prefix + generated

            elif self.contrast_logits_type == "x":
                sentence = decode.split(q_delim)[check_n]
                sentence.replace(a_delim, "")

            else:
                raise Exception("Not recognised:", self.contrast_logits_type)

            sentences.append(sentence)

        encoded = self.tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True)
        contrast_scores = self.model(input_ids=encoded['input_ids'].cuda(),
                                     attention_mask=encoded['attention_mask'].cuda())

        self.contrast_scores = contrast_scores.logits[:, -1, :]

        return scores - self.alpha * self.contrast_scores

