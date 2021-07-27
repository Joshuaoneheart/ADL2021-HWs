# ADL 2020 Hw1 

## Goal
Implement LSTM-based method to achieve intent classification and slot tagging.

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
```shell
python train_intent.py
```
