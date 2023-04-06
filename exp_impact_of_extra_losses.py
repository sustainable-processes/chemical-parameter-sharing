# %%
import pathlib

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import param_sharing.reactions.get
import param_sharing.reactions.filters
import param_sharing.learn.ohe
import param_sharing.learn.model
import param_sharing.learn.fit


df = param_sharing.reactions.get.get_reaction_df(
    cleaned_data_path=pathlib.Path("data/ORD_USPTO/cleaned_data.pkl"),
    rxn_classes_path=pathlib.Path("data/ORD_USPTO/classified_rxn.smi"),
)

mask = param_sharing.reactions.filters.get_classified_rxn_data_mask(df)

# unpickle
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

test_idx = indexes[int(df.shape[0] * train_test_split) :]
train_val_idx = indexes[: int(df.shape[0] * train_test_split)]
train_idx = train_val_idx[: int(train_val_idx.shape[0] * train_val_split)]
val_idx = train_val_idx[int(train_val_idx.shape[0] * train_val_split) :]

assert rxn_diff_fp.shape == product_fp.shape

train_product_fp = torch.Tensor(product_fp[train_idx])
train_rxn_diff_fp = torch.Tensor(rxn_diff_fp[train_idx])
val_product_fp = torch.Tensor(product_fp[val_idx])
val_rxn_diff_fp = torch.Tensor(rxn_diff_fp[val_idx])

train_catalyst, val_catalyst, cat_enc = param_sharing.learn.ohe.apply_train_ohe_fit(
    df[["catalyst_0"]].fillna("NULL"), train_idx, val_idx, tensor_func=torch.Tensor
)
train_solvent_0, val_solvent_0, sol0_enc = param_sharing.learn.ohe.apply_train_ohe_fit(
    df[["solvent_0"]].fillna("NULL"), train_idx, val_idx, tensor_func=torch.Tensor
)
train_solvent_1, val_solvent_1, sol1_enc = param_sharing.learn.ohe.apply_train_ohe_fit(
    df[["solvent_1"]].fillna("NULL"), train_idx, val_idx, tensor_func=torch.Tensor
)
(
    train_reagents_0,
    val_reagents_0,
    reag0_enc,
) = param_sharing.learn.ohe.apply_train_ohe_fit(
    df[["reagents_0"]].fillna("NULL"), train_idx, val_idx, tensor_func=torch.Tensor
)
(
    train_reagents_1,
    val_reagents_1,
    reag1_enc,
) = param_sharing.learn.ohe.apply_train_ohe_fit(
    df[["reagents_1"]].fillna("NULL"), train_idx, val_idx, tensor_func=torch.Tensor
)
(
    train_temperature,
    val_temperature,
    temp_enc,
) = param_sharing.learn.ohe.apply_train_ohe_fit(
    df[["temperature_0"]].fillna(-1.0), train_idx, val_idx, tensor_func=torch.Tensor
)

del df

print("Loaded data")

# %%


cut_off = 100
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

cut_off = 10000
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

m = param_sharing.learn.model.ColeyModel(
    product_fp_dim=train_data["product_fp"].shape[-1],
    rxn_diff_fp_dim=train_data["rxn_diff_fp"].shape[-1],
    cat_dim=train_data["catalyst"].shape[-1],
    sol1_dim=train_data["solvent_1"].shape[-1],
    sol2_dim=train_data["solvent_2"].shape[-1],
    reag1_dim=train_data["reagents_1"].shape[-1],
    reag2_dim=train_data["reagents_2"].shape[-1],
    temp_dim=train_data["temperature"].shape[-1],
)

pred = m.forward_dict(data=train_data)
print(
    "true",
    pd.Series(train_data["catalyst"].argmax(dim=1)).value_counts()
    / train_data["catalyst"].shape[0],
    sep="\n",
)
print(
    "pred",
    pd.Series(pred["catalyst"].argmax(dim=1)).value_counts()
    / train_data["catalyst"].shape[0],
    sep="\n",
)


targets = [
    "catalyst",
    "solvent_1",
    "solvent_2",
    "reagents_1",
    "reagents_2",
    "temperature",
]

# %%

_targets = []
_targets_cat_losses = {}
_targets_cat_accs = {}
for t in targets:
    print(t)
    _targets.append(t)

    m = param_sharing.learn.model.ColeyModel(
        product_fp_dim=train_data["product_fp"].shape[-1],
        rxn_diff_fp_dim=train_data["rxn_diff_fp"].shape[-1],
        cat_dim=train_data["catalyst"].shape[-1],
        sol1_dim=train_data["solvent_1"].shape[-1],
        sol2_dim=train_data["solvent_2"].shape[-1],
        reag1_dim=train_data["reagents_1"].shape[-1],
        reag2_dim=train_data["reagents_2"].shape[-1],
        temp_dim=train_data["temperature"].shape[-1],
    )

    losses, acc_metrics = param_sharing.learn.fit.train_loop(
        model=m,
        train_data=train_data,
        epochs=5,
        batch_size=0.25,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(m.parameters(), lr=1e-4),
        targets=_targets,
        val_data=val_data,
    )
    _targets_cat_losses[t] = losses["catalyst"]
    _targets_cat_accs[t] = acc_metrics["catalyst"]

    f, ax = plt.subplots(1, 3)
    ax[0].plot(losses["sum"]["train"], label="sum train")
    ax[0].plot(losses["sum"]["val"], label="sum val")
    ax[1].plot(losses["catalyst"]["train"], label="catalyst train")
    ax[1].plot(losses["catalyst"]["val"], label="catalyst val")
    ax[2].plot(acc_metrics["catalyst"]["top5"]["train"], label="catalyst train 5acc")
    ax[2].plot(acc_metrics["catalyst"]["top5"]["val"], label="catalyst val 5acc")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()


# %%

f, ax = plt.subplots(3, 2, figsize=(20, 10))

for t in _targets_cat_losses:
    ax[0][0].plot(_targets_cat_losses[t]["train"], label=f"{t}")
ax[0][0].legend()
ax[0][0].set_title("train loss")
for t in _targets_cat_losses:
    ax[0][1].plot(_targets_cat_losses[t]["val"], label=f"{t}")
ax[0][1].legend()
ax[0][1].set_title("val loss")
for t in _targets_cat_accs:
    ax[1][0].plot(_targets_cat_accs[t]["top1"]["train"], label=f"{t}")
ax[1][0].legend()
ax[1][0].set_title("train acc top 1")
for t in _targets_cat_accs:
    ax[1][1].plot(_targets_cat_accs[t]["top1"]["val"], label=f"{t}")
ax[1][1].legend()
ax[1][1].set_title("val acc top 1")
for t in _targets_cat_accs:
    ax[2][0].plot(_targets_cat_accs[t]["top5"]["train"], label=f"{t}")
ax[2][0].legend()
ax[2][0].set_title("train acc top 5")
for t in _targets_cat_accs:
    ax[2][1].plot(_targets_cat_accs[t]["top5"]["val"], label=f"{t}")
ax[2][1].legend()
ax[2][1].set_title("val acc top 5")

# %%

pred = m.forward_dict(data=train_data)
print(
    "true",
    pd.Series(train_data["catalyst"].argmax(dim=1)).value_counts()
    / train_data["catalyst"].shape[0],
    sep="\n",
)
print(
    "pred",
    pd.Series(pred["catalyst"].argmax(dim=1)).value_counts()
    / train_data["catalyst"].shape[0],
    sep="\n",
)

# %%
