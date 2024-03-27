#!/bin/bash
python networks/train_model.py --dir "normal"
python networks/train_model.py --dir "outside_parameters"
python networks/train_model.py --dir "outside_model"
python networks/train_model.py --dir "all_models_small" --load_ext_coef True
python networks/train_model.py --dir "all_models_large" --load_ext_coef True