#!/bin/bash
# create necessary folders
mkdir ckpt
mkdir graph
mkdir prediction
mkdir statistic

# install necessary package
python3.9 -m pip install tqdm
python3.9 -m pip install spacy==3.0.5
python3.9 -m spacy download zh_core_web_md
python3.9 -m pip install transformers==4.5.0

python3.9 train.py --eval 0 --num_epoch 0 --model_name roberta_large --context ${1} --test ${2} --prediction ${3}
# generate prediction of bonus part
cd ./bonus
# slot_tagging
python3.9 ./train_slot.py --num_epoch 0
# intent_classification
python3.9 ./train_intent.py --num_epoch 0
