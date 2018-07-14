#!/usr/bin/env bash

python3 features_extra.py
python3 features_counter.py
python3 features_flat_base.py
python3 features_flat_spalen.py
python3 features_salestart.py
python3 features_start_square.py
python3 features_status_month3.py
python3 features_status_month2.py
python3 features_status_month1.py
python3 features_price.py
python3 features_values_month3.py
python3 features_values_month2.py
python3 features_values_month1.py

python3 features_merge_month3.py
python3 train_model_month3.py

python3 features_merge_month2.py
python3 train_model_month2.py

python3 features_merge_month1.py
python3 train_model_month1.py

python3 merge_sub.py
