#!/bin/bash
python3.9 -m pip install gdown
mkdir ./ckpt
mkdir ./ckpt/roberta_large

gdown --id  --output ./ckpt/robert_large
# install bonus checkpoint
gdown --id "188SJ02vikJUmvpthCUm7aJybd3IbvISK" --output ./bonus/ckpt/intent/model.ckpt
gdown --id "1Z6F1n2x4CL3F87wVikBW069UXz2aSfcp" --output ./bonus/ckpt/slot/model.ckpt
