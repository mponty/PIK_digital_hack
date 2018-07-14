import pandas as pd
import numpy as np


def name_to_col_num(df, col_names):
    col_indexes = []
    for col_name in col_names:
        if col_name not in list(df.columns):
            continue
        new_ind = list(df.columns).index(col_name)
        col_indexes.append(new_ind)

    return col_indexes


def smart_col_num(df):
    cat_ff = []

    for col_ind, col_name in enumerate(list(df.columns)):
        str_check = str(df[col_name].values[0])

        try:
            float(str_check)
        except ValueError:
            cat_ff.append(col_ind)

    return cat_ff
