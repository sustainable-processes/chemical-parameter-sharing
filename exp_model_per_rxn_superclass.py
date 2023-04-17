"""
We want to do parameter sharing across subclasses, therefore we are going to look at the case of having a model per rxn superclass
We would expect that the set of models across subclasses to under perform the single across all subclasses. 
"""


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
import param_sharing.learn.metrics
import config.single_vs_multi


df = param_sharing.reactions.get.get_reaction_df(
    cleaned_data_path=pathlib.Path("data/ORD_USPTO/cleaned_data.pkl"),
    rxn_classes_path=pathlib.Path("data/ORD_USPTO/classified_rxn.smi"),
    verbose=True,
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
    print("Applied mask")

assert df.shape[0] == rxn_diff_fp.shape[0]
assert df.shape[0] == product_fp.shape[0]

rng = np.random.default_rng(config.single_vs_multi.seed)


indexes = np.arange(df.shape[0])
rng.shuffle(indexes)

train_test_split = config.single_vs_multi.train_test_split
train_val_split = config.single_vs_multi.train_val_split

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

train_super_class, val_super_class = df["rxn_super_class"].iloc[train_idx].reset_index(
    drop=True
), df["rxn_super_class"].iloc[val_idx].reset_index(drop=True)

print("Loaded data")

del df

# %%

targets = config.single_vs_multi.targets

untrained_predictions = {}
predictions = {}

super_classes = train_super_class.unique()

for sc in super_classes:
    print("super class:", sc)

    sc_train_mask = train_super_class == sc
    sc_val_mask = val_super_class == sc
    sc_train_index = train_super_class[sc_train_mask].index
    sc_val_index = val_super_class[sc_val_mask].index

    cut_off = config.single_vs_multi.train_cutoff
    train_data = {
        "product_fp": train_product_fp[sc_train_index][:cut_off],
        "rxn_diff_fp": train_rxn_diff_fp[sc_train_index][:cut_off],
        "catalyst": train_catalyst[sc_train_index][:cut_off],
        "solvent_1": train_solvent_0[sc_train_index][:cut_off],
        "solvent_2": train_solvent_1[sc_train_index][:cut_off],
        "reagents_1": train_reagents_0[sc_train_index][:cut_off],
        "reagents_2": train_reagents_1[sc_train_index][:cut_off],
        "temperature": train_temperature[sc_train_index][:cut_off],
    }

    cut_off = config.single_vs_multi.val_cutoff
    val_data = {
        "product_fp": val_product_fp[sc_val_index][:cut_off],
        "rxn_diff_fp": val_rxn_diff_fp[sc_val_index][:cut_off],
        "catalyst": val_catalyst[sc_val_index][:cut_off],
        "solvent_1": val_solvent_0[sc_val_index][:cut_off],
        "solvent_2": val_solvent_1[sc_val_index][:cut_off],
        "reagents_1": val_reagents_0[sc_val_index][:cut_off],
        "reagents_2": val_reagents_1[sc_val_index][:cut_off],
        "temperature": val_temperature[sc_val_index][:cut_off],
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
        (
            pd.Series(train_data["catalyst"].argmax(dim=1)).value_counts()
            / train_data["catalyst"].shape[0]
        ).iloc[:5],
        sep="\n",
    )
    print(
        "pred",
        (
            pd.Series(pred["catalyst"].argmax(dim=1)).value_counts()
            / train_data["catalyst"].shape[0]
        ).iloc[:5],
        sep="\n",
    )

    untrained_predictions[sc] = {"train": {}, "val": {}}
    for t in targets:
        untrained_predictions[sc]["train"][t] = {"true": train_data[t], "pred": pred[t]}
        untrained_predictions[sc]["val"][t] = {
            "true": val_data[t],
            "pred": m.forward_dict(data=val_data)[t],
        }

    losses, acc_metrics = param_sharing.learn.fit.train_loop(
        model=m,
        train_data=train_data,
        epochs=config.single_vs_multi.epochs,
        batch_size=config.single_vs_multi.batch_size,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(m.parameters(), lr=config.single_vs_multi.lr),
        targets=targets,
        val_data=val_data,
    )

    pred = m.forward_dict(data=train_data)
    print(
        "true",
        (
            pd.Series(train_data["catalyst"].argmax(dim=1)).value_counts()
            / train_data["catalyst"].shape[0]
        ).iloc[:5],
        sep="\n",
    )
    print(
        "pred",
        (
            pd.Series(pred["catalyst"].argmax(dim=1)).value_counts()
            / train_data["catalyst"].shape[0]
        ).iloc[:5],
        sep="\n",
    )

    predictions[sc] = {"train": {}, "val": {}}
    for t in targets:
        predictions[sc]["train"][t] = {"true": train_data[t], "pred": pred[t]}
        predictions[sc]["val"][t] = {
            "true": val_data[t],
            "pred": m.forward_dict(data=val_data)[t],
        }

    plt.plot(losses["sum"]["train"], label="sum train")
    plt.plot(losses["sum"]["val"], label="sum val")
    plt.legend()
    plt.show()

    if "catalyst" in targets:
        plt.plot(losses["catalyst"]["train"], label="catalyst train")
        # plt.legend()
        plt.plot(losses["catalyst"]["val"], label="catalyst val")
        plt.legend()
        plt.show()
    if "solvent_1" in targets:
        plt.plot(losses["solvent_1"]["train"], label="solvent_1 train")
        # plt.legend()
        plt.plot(losses["solvent_1"]["val"], label="solvent_1 val")
        plt.legend()
        plt.show()
    if "solvent_2" in targets:
        plt.plot(losses["solvent_2"]["train"], label="solvent_2 train")
        # plt.legend()
        plt.plot(losses["solvent_2"]["val"], label="solvent_2 val")
        plt.legend()
        plt.show()
    if "reagents_1" in targets:
        plt.plot(losses["reagents_1"]["train"], label="reagents_1 train")
        # plt.legend()
        plt.plot(losses["reagents_1"]["val"], label="reagents_1 val")
        plt.legend()
        plt.show()
    if "reagents_2" in targets:
        plt.plot(losses["reagents_2"]["train"], label="reagents_2 train")
        # plt.legend()
        plt.plot(losses["reagents_2"]["val"], label="reagents_2 val")
        plt.legend()
        plt.show()
    if "temperature" in targets:
        plt.plot(losses["temperature"]["train"], label="temperature train")
        # plt.legend()
        plt.plot(losses["temperature"]["val"], label="temperature val")
        plt.legend()
        plt.show()

    if super_classes[2] == sc:
        break

# %%

print("get catalyst acc")

for superclass in predictions:
    train_acc = param_sharing.learn.metrics.get_topk_acc(
        pred=predictions[superclass]["train"]["catalyst"]["pred"],
        true=predictions[superclass]["train"]["catalyst"]["true"],
        k=1,
    )
    val_acc = param_sharing.learn.metrics.get_topk_acc(
        pred=predictions[superclass]["val"]["catalyst"]["pred"],
        true=predictions[superclass]["val"]["catalyst"]["true"],
        k=1,
    )
    print(f"{superclass=}: {train_acc=}, {val_acc=}")

train_acc = param_sharing.learn.metrics.get_topk_acc(
    pred=torch.cat([predictions[p]["train"]["catalyst"]["pred"] for p in predictions]),
    true=torch.cat([predictions[p]["train"]["catalyst"]["true"] for p in predictions]),
    k=[1, 5],
)
val_acc = param_sharing.learn.metrics.get_topk_acc(
    pred=torch.cat([predictions[p]["val"]["catalyst"]["pred"] for p in predictions]),
    true=torch.cat([predictions[p]["val"]["catalyst"]["true"] for p in predictions]),
    k=[1, 5],
)
print(f"trained catalyst top 1 acc: {train_acc[1]=} {val_acc[1]=}")
print(f"trained catalyst top 5 acc: {train_acc[5]=} {val_acc[5]=}")

train_acc = param_sharing.learn.metrics.get_topk_acc(
    pred=torch.cat(
        [
            untrained_predictions[p]["train"]["catalyst"]["pred"]
            for p in untrained_predictions
        ]
    ),
    true=torch.cat(
        [
            untrained_predictions[p]["train"]["catalyst"]["true"]
            for p in untrained_predictions
        ]
    ),
    k=[1, 5],
)
val_acc = param_sharing.learn.metrics.get_topk_acc(
    pred=torch.cat(
        [
            untrained_predictions[p]["val"]["catalyst"]["pred"]
            for p in untrained_predictions
        ]
    ),
    true=torch.cat(
        [
            untrained_predictions[p]["val"]["catalyst"]["true"]
            for p in untrained_predictions
        ]
    ),
    k=[1, 5],
)
print(f"untrained catalyst top 1 acc: {train_acc[1]=} {val_acc[1]=}")
print(f"untrained catalyst top 5 acc: {train_acc[5]=} {val_acc[5]=}")

# %%
