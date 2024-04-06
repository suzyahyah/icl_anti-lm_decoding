## Anti-LM Decoding for Zeroshot Machine Translation
This repository contains the code for the NAACL 2024 paper: [pdf](https://arxiv.org/pdf/2311.08324.pdf)

### Project Structure
All code should be run from the project directory home page. e.g,
`$HOME/projects/icl_antilm_decoding`

### Getting Data
To get started, follow these steps to prepare the required data.

`bash bin/prepare-flores.sh`

### Running Experiments:

#### Single experiment run

To run a single experiment, modify and execute the following script with your desired
parameters:

`bash bin/submit_logits.sh $seed $model $direction $logitsp_cf $nprompts $alpha $contrast $gen_cf`

Example:

`bash bin/submit_docmt.sh 0 gptn125m en-fr configs/logits_processor/anti_lm.yaml 0 0.3 x configs/generator/default.yaml`

P.S the experiments are run with a roll-your-own config system. To understand how it works, see
[explanation here](https://suzyahyah.github.io/code/2023/10/01/omegaconf-argparse.html)


#### Batch experiment run (COE grid)

Copy and modify this file for your own grid

`bash bin/batch_submit_logits.sh` 

#### Config files

Different logits decoder settings are controlled by the configuration files located in `configs/logits_processor/{anti_lm, pmi}.yaml` files and `configs/generator/{beamsearch, default}.yaml`.

#### Running different languages

To run experiments with different languages, refer to the MM100-code in  `assets/flores_map.csv`. Both language directions e.g., `en-fr` or `fr-en` are supported.

#### Running different models
The code was run on the following models:

* `bloom{3b,7b1}`
* `gptn{125m,2.7B}`
* `xglm{2.9B,7.5B}`
* `llama{7b,7b-chat}`

Capitalisation matters for the `b` and the `B` in the model names.

#### Printing Results

Results are printed using the file path specified in `configs/file_paths/logits.yaml`. Use the
following command to print results:

For example,

`for dir in fr de pt; do p code/present/logits/main_tables.py --data.direction en-${dir}; done`

## Contributing

#### Experimenting with Decoding methods

If the method exists in default huggingface generator configs, this can be modified in `configs/generator/{}.yaml > hf_generator_configs`. These arguments go directly into `model.generate()` in `code/utils/utils.py`. Remember to modify `generator.name` if you make a new config file because filenames (experiment versions) depend on `generator.name`, see `configs/file_paths/logits.yaml` 
