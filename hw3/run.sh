#!/bin/bash
python3.8 train.py --phase "test" --out_file ${2} --test_file ${1} --device cuda:1
