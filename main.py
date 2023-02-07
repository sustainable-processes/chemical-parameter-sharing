# %%
import pathlib

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import src.reactions.get
import src.learn.ohe
import src.learn.model
import src.learn.fit
import src.learn.metrics


df = src.reactions.get.get_reaction_df(
    cleaned_data_path = pathlib.Path("data/ORD_USPTO/cleaned_data.pkl"),
    rxn_classes_path = pathlib.Path("data/ORD_USPTO/classified_rxn.smi"),
    verbose=True,
)

mask = src.reactions.get.get_classified_rxn_data_mask(df)

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
    print("  - Applied mask")

assert df.shape[0] == rxn_diff_fp.shape[0]
assert df.shape[0] == product_fp.shape[0]

rng = np.random.default_rng(12345)


indexes = np.arange(df.shape[0])
rng.shuffle(indexes)

train_test_split = 0.9
train_val_split = 0.8

test_idx = indexes[int(df.shape[0] * train_test_split):]
train_val_idx = indexes[:int(df.shape[0] * train_test_split)]
train_idx = train_val_idx[:int(train_val_idx.shape[0] * train_val_split)]
val_idx = train_val_idx[int(train_val_idx.shape[0] * train_val_split):]

assert rxn_diff_fp.shape == product_fp.shape

train_product_fp = torch.Tensor(product_fp[train_idx])
train_rxn_diff_fp = torch.Tensor(rxn_diff_fp[train_idx])
val_product_fp = torch.Tensor(product_fp[val_idx])
val_rxn_diff_fp = torch.Tensor(rxn_diff_fp[val_idx])

train_catalyst, val_catalyst, cat_enc = src.learn.ohe.apply_train_ohe_fit(df[['catalyst_0']], train_idx, val_idx)
train_solvent_0, val_solvent_0, sol0_enc = src.learn.ohe.apply_train_ohe_fit(df[['solvent_0']], train_idx, val_idx)
train_solvent_1, val_solvent_1, sol1_enc = src.learn.ohe.apply_train_ohe_fit(df[['solvent_1']], train_idx, val_idx)
train_reagents_0, val_reagents_0, reag0_enc = src.learn.ohe.apply_train_ohe_fit(df[['reagents_0']], train_idx, val_idx)
train_reagents_1, val_reagents_1, reag1_enc = src.learn.ohe.apply_train_ohe_fit(df[['reagents_1']], train_idx, val_idx)
train_temperature, val_temperature, temp_enc = src.learn.ohe.apply_train_ohe_fit(df[['temperature_0']].fillna(-1), train_idx, val_idx)

# del df

print("Loaded data")

# %%


cut_off = 5_000
train_data = {
    "product_fp": train_product_fp[:cut_off],
    "rxn_diff_fp": train_rxn_diff_fp[:cut_off],
    "catalyst": train_catalyst[:cut_off],
    "solvent_1": train_solvent_0[:cut_off],
    "solvent_2": train_solvent_1[:cut_off],
    "reagents_1": train_reagents_0[:cut_off],
    "reagents_2": train_reagents_1[:cut_off],
    "temperature": train_temperature[:cut_off],
}

cut_off = None
val_data = {
    "product_fp": val_product_fp[:cut_off],
    "rxn_diff_fp": val_rxn_diff_fp[:cut_off],
    "catalyst": val_catalyst[:cut_off],
    "solvent_1": val_solvent_0[:cut_off],
    "solvent_2": val_solvent_1[:cut_off],
    "reagents_1": val_reagents_0[:cut_off],
    "reagents_2": val_reagents_1[:cut_off],
    "temperature": val_temperature[:cut_off],
}

m = src.learn.model.ColeyModel(
    product_fp_dim=train_data['product_fp'].shape[-1],
    rxn_diff_fp_dim=train_data['rxn_diff_fp'].shape[-1],
    cat_dim=train_data['catalyst'].shape[-1],
    sol1_dim=train_data['solvent_1'].shape[-1],
    sol2_dim=train_data['solvent_2'].shape[-1],
    reag1_dim=train_data['reagents_1'].shape[-1],
    reag2_dim=train_data['reagents_2'].shape[-1],
    temp_dim=train_data['temperature'].shape[-1],
)

pred = m.forward_dict(data=train_data)
print("true", (pd.Series(train_data['catalyst'].argmax(dim=1)).value_counts() / train_data['catalyst'].shape[0]).iloc[:5], sep="\n")
print("pred", (pd.Series(pred['catalyst'].argmax(dim=1)).value_counts() / train_data['catalyst'].shape[0]).iloc[:5], sep="\n")

train_acc = src.learn.metrics.get_topk_acc(
    pred=pred['catalyst'], 
    true=train_data['catalyst'],
    k=[1,5],
)
val_acc = src.learn.metrics.get_topk_acc(
    pred=m.forward_dict(data=val_data)['catalyst'], 
    true=val_data['catalyst'],
    k=[1,5],
)
print(f"untrained catalyst top 1 acc: {train_acc[1]=} {val_acc[1]=}")
print(f"untrained catalyst top 5 acc: {train_acc[5]=} {val_acc[5]=}")

targets=[
    "catalyst",
    "solvent_1",
    "solvent_2",
    "reagents_1",
    "reagents_2",
    "temperature",
]

losses, acc_metrics = src.learn.fit.train_loop(
    model=m, 
    train_data=train_data, 
    epochs=10,
    batch_size=0.05,
    loss_fn=torch.nn.CrossEntropyLoss(), 
    optimizer=torch.optim.Adam(m.parameters(), lr=1e-2),
    targets=targets,
    val_data=val_data,
)

pred = m.forward_dict(data=train_data)
print("true", (pd.Series(train_data['catalyst'].argmax(dim=1)).value_counts() / train_data['catalyst'].shape[0]).iloc[:5], sep="\n")
print("pred", (pd.Series(pred['catalyst'].argmax(dim=1)).value_counts() / train_data['catalyst'].shape[0]).iloc[:5], sep="\n")

train_acc = src.learn.metrics.get_topk_acc(
    pred=pred['catalyst'], 
    true=train_data['catalyst'],
    k=[1,5],
)
val_acc = src.learn.metrics.get_topk_acc(
    pred=m.forward_dict(data=val_data)['catalyst'], 
    true=val_data['catalyst'],
    k=[1,5],
)
print(f"trained catalyst top 1 acc: {train_acc[1]=} {val_acc[1]=}")
print(f"trained catalyst top 5 acc: {train_acc[5]=} {val_acc[5]=}")

# %%
plt.plot(losses['sum']["train"], label="sum train"); plt.legend()
plt.plot(losses['sum']["val"], label="sum val"); plt.legend()
# %%
if "catalyst" in targets:
    plt.plot(losses['catalyst']["train"], label="catalyst train"); #plt.legend()
    plt.plot(losses['catalyst']["val"], label="catalyst val"); plt.legend()
    plt.show()
if "solvent_1" in targets:
    plt.plot(losses['solvent_1']["train"], label="solvent_1 train"); #plt.legend()
    plt.plot(losses['solvent_1']["val"], label="solvent_1 val"); plt.legend()
    plt.show()
if "solvent_2" in targets:
    plt.plot(losses['solvent_2']["train"], label="solvent_2 train"); #plt.legend()
    plt.plot(losses['solvent_2']["val"], label="solvent_2 val"); plt.legend()
    plt.show()
if "reagents_1" in targets:
    plt.plot(losses['reagents_1']["train"], label="reagents_1 train"); #plt.legend()
    plt.plot(losses['reagents_1']["val"], label="reagents_1 val"); plt.legend()
    plt.show()
if "reagents_2" in targets:
    plt.plot(losses['reagents_2']["train"], label="reagents_2 train"); #plt.legend()
    plt.plot(losses['reagents_2']["val"], label="reagents_2 val"); plt.legend()
    plt.show()
if "temperature" in targets:
    plt.plot(losses['temperature']["train"], label="temperature train"); #plt.legend()
    plt.plot(losses['temperature']["val"], label="temperature val"); plt.legend()
    plt.show()

# %%
