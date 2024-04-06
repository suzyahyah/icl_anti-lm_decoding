#!/usr/bin/env bash
# Author: Suzanna Sia

logs_e=$(pwd)/logs_e
logs_o=$(pwd)/logs_o

### loop settings
DIRS=(en-fr en-pt en-de)
#DIRS=(fr-en pt-en de-en)
#DIRS=(en-pt en-de)
MODELS=(bloom3b xglm2.9B xglm7.5B bloom7b1 llama7b llama7b-chat)
#MODELS=(gptn125M)

#MODELS=(bloomz7b1 bloom7b1 llama7b llama7b-chat)
#MODELS=(llama7b-chat llama7b)
NPROMPTS=(0)
#ALPHA=(0.1 0.3 0.5 0.8)
ALPHA=(0.3)
GPU=rtx
#GPU=rtx
INSTRS=(instr_L1L2)
MODES=(anti_lm_gompertz anti_lm_logistic)
CONTRASTS=(x)
SEEDS=(0)  # there is no point running different seeds in 0 shot.
GENS=(default) # beam_search) # {beam_search, default}
QSUB=1 # {-1, 0, 1}

for seed in ${SEEDS[@]}; do
for direction in ${DIRS[@]}; do
for model in ${MODELS[@]}; do
for nprompts in ${NPROMPTS[@]}; do
for instr in ${INSTRS[@]}; do
for alpha in ${ALPHA[@]}; do
for mode in ${MODES[@]}; do
for contrast in ${CONTRASTS[@]}; do
for gen in ${GENS[@]}; do

    if [[ "$model" == "bloom7b1" || "$model" == "xglm7.5B" ]]; then
        GPU=v100
    else
        GPU=rtx
    fi
    #GPU=v100

#    if [[ "$mode" == "anti_lm" ]]; then
#        alpha=0.3
#    elif [[ "$mode" == "default" ]]; then
#        alpha=0
#    else
#        alpha=0.1
#    fi

    logitsp_cf=configs/logits_processor/${mode}.yaml
    gen_cf=configs/generator/${gen}.yaml

    [ ! -f "$logitsp_cf" ] && echo "config file $logitsp_cf does not exist" && exit 1

    #args="$seed $model $direction $logitsp_cf $nprompts $alpha $contrast $gen_cf ${instr}_${direction}"
    args="$seed $model $direction $logitsp_cf $nprompts $alpha $contrast $gen_cf ${instr}"

    echo $args
    if [[ $QSUB == 0 ]]; then
        bash bin/submit_logits.sh $args

    elif [[ $QSUB == 1 ]]; then
        qsubname=${model}_${cf_}${direction}
        settings="-l mem_free=50G,gpu=1,h_rt=30:00:00 -q gpu.q@@$GPU"
        qsub -N $qsubname $settings -e $logs_e -o $logs_o bin/submit_logits.sh $args
    else
        echo "Do nothing"
    fi
done
done
done
done
done
done
done
done
done
