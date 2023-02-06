# %%
import pathlib
import typing
import numpy as np
import pandas as pd
import torch

import data
import ohe
import model


df = data.get_df(
    cleaned_data_path = pathlib.Path("data/ORD_USPTO/cleaned_data.pkl"),
    rxn_classes_path = pathlib.Path("data/ORD_USPTO/classified_rxn.smi"),
)

mask = data.get_classified_rxn_data_mask(df)

#unpickle
rxn_diff_fp = np.load("data/ORD_USPTO/USPTO_rxn_diff_fp.pkl.npy", allow_pickle=True)
product_fp = np.load("data/ORD_USPTO/USPTO_product_fp.pkl.npy", allow_pickle=True)
# Run all cells in the "Read in data" section to get data_df

assert df.shape[0] == mask.shape[0]
assert df.shape[0] == rxn_diff_fp.shape[0]
assert df.shape[0] == product_fp.shape[0]

if True:
    df = df[mask]
    rxn_diff_fp = rxn_diff_fp[mask]
    product_fp = product_fp[mask]
    print("Applied mask")

assert df.shape[0] == rxn_diff_fp.shape[0]
assert df.shape[0] == product_fp.shape[0]

rng = np.random.default_rng(12345)


indexes = np.arange(df.shape[0])
rng.shuffle(indexes)

train_test_split = 0.8
train_val_split = 0.6

test = indexes[int(df.shape[0] * train_test_split):]
train_val = indexes[:int(df.shape[0] * train_test_split)]
train = train_val[:int(train_val.shape[0] * train_val_split)]
val = train_val[int(train_val.shape[0] * train_val_split):]

assert rxn_diff_fp.shape == product_fp.shape

train_product_fp = torch.Tensor(product_fp[train])
train_rxn_diff_fp = torch.Tensor(rxn_diff_fp[train])
val_product_fp = torch.Tensor(product_fp[val])
val_rxn_diff_fp = torch.Tensor(rxn_diff_fp[val])


def apply_train_ohe_fit(df, train, val, to_tensor:bool = True):
    enc = ohe.GetDummies()
    _ = enc.fit(df.iloc[train])
    _ohe = enc.transform(df)
    _tr, _val = _ohe.iloc[train].values, _ohe.iloc[val].values
    if to_tensor:
        _tr, _val = torch.Tensor(_tr), torch.Tensor(_val)
    return _tr, _val, enc


train_catalyst, val_catalyst, cat_enc = apply_train_ohe_fit(df[['catalyst_0']], train, val)
train_solvent_0, val_solvent_0, sol0_enc = apply_train_ohe_fit(df[['solvent_0']], train, val)
train_solvent_1, val_solvent_1, sol1_enc = apply_train_ohe_fit(df[['solvent_1']], train, val)
train_reagents_0, val_reagents_0, reag0_enc = apply_train_ohe_fit(df[['reagents_0']], train, val)
train_reagents_1, val_reagents_1, reag1_enc = apply_train_ohe_fit(df[['reagents_1']], train, val)
train_temperature, val_temperature, temp_enc = apply_train_ohe_fit(df[['temperature_0']].fillna(-1), train, val)

del df
# %%


m = model.ColeyModel(
    product_fp_dim=train_product_fp.shape[-1],
    rxn_diff_fp_dim=train_rxn_diff_fp.shape[-1],
    cat_dim=train_catalyst.shape[-1],
    sol1_dim=train_solvent_0.shape[-1],
    sol2_dim=train_solvent_1.shape[-1],
    reag1_dim=train_reagents_0.shape[-1],
    reag2_dim=train_reagents_1.shape[-1],
    temp_dim=train_temperature.shape[-1],
)


# %%

output = m.forward(
    product_fp=train_product_fp,
    rxn_diff_fp=train_rxn_diff_fp,
    cat=train_catalyst,
    sol1=train_solvent_0,
    sol2=train_solvent_1,
    reag1=train_reagents_0,
    reag2=train_reagents_1,
)

train_catalyst_pred, train_solvent_0_pred, train_solvent_1_pred, train_reagents_0_pred, train_reagents_1_pred, train_temperature_pred = output



