# %%
import pathlib

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import src.reactions.get
import src.reactions.filters
import src.learn.ohe
import src.learn.model
import src.learn.fit
import src.learn.metrics


df = src.reactions.get.get_reaction_df(
    cleaned_data_path = pathlib.Path("data/ORD_USPTO/cleaned_data.pkl"),
    rxn_classes_path = pathlib.Path("data/ORD_USPTO/classified_rxn.smi"),
    verbose=True,
)

mask = src.reactions.filters.get_classified_rxn_data_mask(df)

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

train_test_split = 0.8
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

train_catalyst, val_catalyst, cat_enc = src.learn.ohe.apply_train_ohe_fit(df[['catalyst_0']].fillna("NULL"), train_idx, val_idx)
train_solvent_0, val_solvent_0, sol0_enc = src.learn.ohe.apply_train_ohe_fit(df[['solvent_0']].fillna("NULL"), train_idx, val_idx)
train_solvent_1, val_solvent_1, sol1_enc = src.learn.ohe.apply_train_ohe_fit(df[['solvent_1']].fillna("NULL"), train_idx, val_idx)
train_reagents_0, val_reagents_0, reag0_enc = src.learn.ohe.apply_train_ohe_fit(df[['reagents_0']].fillna("NULL"), train_idx, val_idx)
train_reagents_1, val_reagents_1, reag1_enc = src.learn.ohe.apply_train_ohe_fit(df[['reagents_1']].fillna("NULL"), train_idx, val_idx)
train_temperature, val_temperature, temp_enc = src.learn.ohe.apply_train_ohe_fit(df[['temperature_0']].fillna(-1), train_idx, val_idx)

product_fp_dim = train_product_fp.shape[-1]
rxn_diff_fp_dim = train_rxn_diff_fp.shape[-1]
cat_dim = train_catalyst.shape[-1]
sol1_dim = train_solvent_0.shape[-1]
sol2_dim = train_solvent_1.shape[-1]
reag1_dim = train_reagents_0.shape[-1]
reag2_dim = train_reagents_1.shape[-1]
temp_dim = train_temperature.shape[-1]

# del df

print("Loaded data")

# %%


train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        train_product_fp,
        train_rxn_diff_fp,
        train_catalyst,
        train_solvent_0,
        train_solvent_1,
        train_reagents_0,
        train_reagents_1,
        train_temperature,
    ),
    batch_size=5000,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        val_product_fp,
        val_rxn_diff_fp,
        val_catalyst,
        val_solvent_0,
        val_solvent_1,
        val_reagents_0,
        val_reagents_1,
        val_temperature,
    ),
    batch_size=32,
    shuffle=True,
)


train_data = {
    train_product_fp,
    train_rxn_diff_fp,
    train_catalyst,
    train_solvent_0,
    train_solvent_1,
    train_reagents_0,
    train_reagents_1,
    train_temperature,

}


# %%

m = src.learn.model.ColeyModel(
    product_fp_dim=product_fp_dim,
    rxn_diff_fp_dim=rxn_diff_fp_dim,
    cat_dim=cat_dim,
    sol1_dim=sol1_dim,
    sol2_dim=sol2_dim,
    reag1_dim=reag1_dim,
    reag2_dim=reag2_dim,
    temp_dim=temp_dim,
    # use_batchnorm=True,
    # dropout_prob=0.2,
)


preds = []
for i in train_loader:
    preds.append(m.forward(*i[:-1], mode=src.learn.model.TEACHER_FORCE, training=False))
    break


# print("true", (pd.Series(train_data['catalyst'].argmax(dim=1)).value_counts() / train_data['catalyst'].shape[0]).iloc[:5], sep="\n")
# print("pred", (pd.Series(pred['catalyst'].argmax(dim=1)).value_counts() / train_data['catalyst'].shape[0]).iloc[:5], sep="\n")

# train_acc = src.learn.metrics.get_topk_acc(
#     pred=pred['catalyst'], 
#     true=train_data['catalyst'],
#     k=[1,5],
# )
# train_eval_acc = src.learn.metrics.get_topk_acc(
#     pred=m.forward_dict(data=train_data, mode=src.learn.model.HARD_SELECTION)['catalyst'], 
#     true=train_data['catalyst'],
#     k=[1,5],
# )
# val_acc = src.learn.metrics.get_topk_acc(
#     pred=m.forward_dict(data=val_data, mode=src.learn.model.HARD_SELECTION)['catalyst'], 
#     true=val_data['catalyst'],
#     k=[1,5],
# )
# print(f"untrained catalyst top 1 acc: {train_acc[1]=} {train_eval_acc[1]=} {val_acc[1]=}")
# print(f"untrained catalyst top 5 acc: {train_acc[5]=} {train_eval_acc[5]=} {val_acc[5]=}")

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
    epochs=50,
    batch_size=256,
    loss_fn=torch.nn.CrossEntropyLoss(), 
    optimizer=torch.optim.Adam(m.parameters(), lr=1e-4),
    targets=targets,
    val_data=val_data,
    train_kwargs={"mode": src.learn.model.TEACHER_FORCE},
    val_kwargs={"mode": src.learn.model.HARD_SELECTION},
    train_eval=True, 
)

# pred = m.forward_dict(data=train_data, mode=src.learn.model.TEACHER_FORCE)
# print("true", (pd.Series(train_data['catalyst'].argmax(dim=1)).value_counts() / train_data['catalyst'].shape[0]).iloc[:5], sep="\n")
# print("pred", (pd.Series(pred['catalyst'].argmax(dim=1)).value_counts() / train_data['catalyst'].shape[0]).iloc[:5], sep="\n")

# train_acc = src.learn.metrics.get_topk_acc(
#     pred=pred['catalyst'], 
#     true=train_data['catalyst'],
#     k=[1,5],
# )
# train_eval_acc = src.learn.metrics.get_topk_acc(
#     pred=m.forward_dict(data=train_data, mode=src.learn.model.HARD_SELECTION)['catalyst'], 
#     true=train_data['catalyst'],
#     k=[1,5],
# )
# val_acc = src.learn.metrics.get_topk_acc(
#     pred=m.forward_dict(data=val_data)['catalyst'], 
#     true=val_data['catalyst'],
#     k=[1,5],
# )
# print(f"trained catalyst top 1 acc: {train_acc[1]=} {train_eval_acc[1]=} {val_acc[1]=}")
# print(f"trained catalyst top 5 acc: {train_acc[5]=} {train_eval_acc[5]=} {val_acc[5]=}")

# %%
plt.plot(losses['sum']["train"], label="sum train"); plt.legend()
plt.plot(losses['sum']["val"], label="sum val"); plt.legend()
# %%

possible_targets=[
    "catalyst",
    "solvent_1",
    "solvent_2",
    "reagents_1",
    "reagents_2",
    "temperature",
]

for t in possible_targets:
    if t in targets:
        plt.plot(losses[t]["train"], label=f"{t} train"); #plt.legend()
        plt.plot(losses[t]["val"], label=f"{t} val"); plt.legend()
        if "train_val" in losses[t]:
            plt.plot(losses[t]["val"], label="{t} train eval"); plt.legend()
        plt.show()

# %%

print("TEACHER_FORCE")
pred = m.forward_dict(data=train_data, mode=src.learn.model.TEACHER_FORCE)
print("true", (pd.Series(train_data['catalyst'].argmax(dim=1)).value_counts() / train_data['catalyst'].shape[0]).iloc[:5], sep="\n")
print("pred", (pd.Series(pred['catalyst'].argmax(dim=1)).value_counts() / train_data['catalyst'].shape[0]).iloc[:5], sep="\n")

print("HARD_SELECTION")
pred = m.forward_dict(data=train_data, mode=src.learn.model.HARD_SELECTION)
print("true", (pd.Series(train_data['catalyst'].argmax(dim=1)).value_counts() / train_data['catalyst'].shape[0]).iloc[:5], sep="\n")
print("pred", (pd.Series(pred['catalyst'].argmax(dim=1)).value_counts() / train_data['catalyst'].shape[0]).iloc[:5], sep="\n")

print("SOFT_SELECTION")
pred = m.forward_dict(data=train_data, mode=src.learn.model.SOFT_SELECTION)
print("true", (pd.Series(train_data['catalyst'].argmax(dim=1)).value_counts() / train_data['catalyst'].shape[0]).iloc[:5], sep="\n")
print("pred", (pd.Series(pred['catalyst'].argmax(dim=1)).value_counts() / train_data['catalyst'].shape[0]).iloc[:5], sep="\n")

#%%
