#!/usr/bin/env bash

PY="python3"

$PY features_extra.py
$PY features_counter.py
$PY features_flat_base.py
$PY features_flat_spalen.py
$PY features_salestart.py
$PY features_start_square.py
$PY features_status.py
$PY features_price.py

$PY features_merge.py

$PY train_model.py
