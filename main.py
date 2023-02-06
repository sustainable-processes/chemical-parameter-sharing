# %%
import pathlib
import typing

import numpy as np
import pandas as pd
import torch
import torchmetrics
import matplotlib.pyplot as plt
from tqdm import trange

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
    print("  - Applied mask")

assert df.shape[0] == rxn_diff_fp.shape[0]
assert df.shape[0] == product_fp.shape[0]

rng = np.random.default_rng(12345)


indexes = np.arange(df.shape[0])
rng.shuffle(indexes)

train_test_split = 0.8
train_val_split = 0.8

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

def train_loop(model, train_data, *, epochs, batch_size, loss_fn, optimizer, targets, val_data=None):
    
    acc_metrics = {}
    losses = {"sum": {"train": [], "val": []}}
    for target in targets:
        losses[target] = {"train": [], "val": []}
        if target == "temperature":
            continue
        num_classes = train_data[target].shape[1]
        acc_metrics[target] = {
            "top1": {"metric": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1), "train": [], "val": [], "train_batch": []},
            "top3": {"metric": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=3), "train": [], "val": [], "train_batch": []},
            "top5": {"metric": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5), "train": [], "val": [], "train_batch": []},
        }
        

    train_size = train_data["product_fp"].shape[0]

    for e in (t := trange(epochs, desc='', leave=True)):

        output_str = ""
    
        idxes = np.arange(train_size)
        np.random.shuffle(idxes)

        prev_idx = 0
        interval = int(idxes.shape[0] * batch_size)

        # storage for use during mini-batching, reset every epoch
        epoch_losses = {"sum": {"train": [], "val": []}}
        for target in targets:
            epoch_losses[target] = {"train": [], "val": []}
        for target in targets:
            if target == "temperature":
                continue
            for top in ["top1", "top3", "top5"]:
                acc_metrics[target][top]['train_batch'] = [] 

        # run across training
        for idx in range(interval, idxes.shape[0]+1, interval):
            if batch_size < 1.0:
                batch_idxes = idxes[prev_idx:idx]
            else:
                batch_idxes = idxes
            prev_idx = idx

            output = model.forward(
                product_fp=train_data['product_fp'][batch_idxes],
                rxn_diff_fp=train_data['rxn_diff_fp'][batch_idxes],
                cat=train_data['catalyst'][batch_idxes],
                sol1=train_data['solvent_1'][batch_idxes],
                sol2=train_data['solvent_2'][batch_idxes],
                reag1=train_data['reagents_1'][batch_idxes],
                reag2=train_data['reagents_2'][batch_idxes],
                training=True,
            )
            pred = {}
            pred['catalyst'], pred['solvent_1'], pred['solvent_2'], pred['reagents_1'], pred['reagents_2'], pred['temperature'] = output

            loss = 0
            for target in targets:  # we can change targets to be loss functions in the future if the loss function changes
                target_batch_loss = loss_fn(pred[target], train_data[target][batch_idxes])    
                factor = 1e-4 if target == "temperature" else 1.0
                loss += (factor * target_batch_loss)
                epoch_losses[target]["train"].append(target_batch_loss.detach().numpy().item())
            epoch_losses["sum"]["train"].append(loss.detach().numpy().item())

            for target in targets:
                if target == "temperature":
                    continue
                for top in ["top1", "top3", "top5"]:
                    acc_metrics[target][top]['train_batch'].append(acc_metrics[target][top]['metric'](pred[target], train_data[target][batch_idxes].argmax(axis=1)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calc the average train loss
        losses["sum"]["train"].append(np.mean(epoch_losses["sum"]["train"]))
        for target in targets:
            losses[target]["train"].append(np.mean(epoch_losses[target]["train"]))

        # calc the average train acc
        for target in targets:
            if target == "temperature":
                continue
            for top in ["top1", "top3", "top5"]:
                acc_metrics[target][top]['train'].append(np.mean(acc_metrics[target][top]['train_batch']))

        output_str += f'Train loss: {losses["sum"]["train"][-1]:.3f}'

        # evaluate with validation data
        if val_data is not None:
            with torch.no_grad():
                output = model.forward(
                    product_fp=val_data['product_fp'],
                    rxn_diff_fp=val_data['rxn_diff_fp'],
                    cat=val_data['catalyst'],
                    sol1=val_data['solvent_1'],
                    sol2=val_data['solvent_2'],
                    reag1=val_data['reagents_1'],
                    reag2=val_data['reagents_2'],
                    training=False,
                )   
                pred = {}
                pred['catalyst'], pred['solvent_1'], pred['solvent_2'], pred['reagents_1'], pred['reagents_2'], pred['temperature'] = output 

                loss = 0
                for target in targets:  # we can change targets to be loss functions in the future if the loss function changes
                    target_batch_loss = loss_fn(pred[target], val_data[target])    
                    factor = 1e-4 if target == "temperature" else 1.0
                    loss += (factor * target_batch_loss)
                    losses[target]["val"].append(target_batch_loss.detach().numpy().item())
                losses["sum"]["val"].append(loss.detach().numpy().item()) 

                for target in targets:
                    if target == "temperature":
                        continue
                    for top in ["top1", "top3", "top5"]:
                        acc_metrics[target][top]['val'].append(acc_metrics[target][top]['metric'](pred[target], val_data[target].argmax(axis=1)))

                output_str += f' | Val loss: {losses["sum"]["val"][-1]:.3f} '
        t.set_description(output_str, refresh=True)
    return losses, acc_metrics


# m = model.ColeyModel(
#     product_fp_dim=train_product_fp.shape[-1],
#     rxn_diff_fp_dim=train_rxn_diff_fp.shape[-1],
#     cat_dim=train_catalyst.shape[-1],
#     sol1_dim=train_solvent_0.shape[-1],
#     sol2_dim=train_solvent_1.shape[-1],
#     reag1_dim=train_reagents_0.shape[-1],
#     reag2_dim=train_reagents_1.shape[-1],
#     temp_dim=train_temperature.shape[-1],
# )

# train_data = {
#     "product_fp": train_product_fp[:1000],
#     "rxn_diff_fp": train_rxn_diff_fp[:1000],
#     "catalyst": train_catalyst[:1000],
#     "solvent_1": train_solvent_0[:1000],
#     "solvent_2": train_solvent_1[:1000],
#     "reagents_1": train_reagents_0[:1000],
#     "reagents_2": train_reagents_1[:1000],
#     "temperature": train_temperature[:1000],
# }

# val_data = {
#     "product_fp": val_product_fp[:1000],
#     "rxn_diff_fp": val_rxn_diff_fp[:1000],
#     "catalyst": val_catalyst[:1000],
#     "solvent_1": val_solvent_0[:1000],
#     "solvent_2": val_solvent_1[:1000],
#     "reagents_1": val_reagents_0[:1000],
#     "reagents_2": val_reagents_1[:1000],
#     "temperature": val_temperature[:1000],
# }


# losses, acc_metrics = train_loop(
#     model=m, 
#     train_data=train_data, 
#     epochs=20,
#     batch_size=0.05,
#     loss_fn=torch.nn.CrossEntropyLoss(), 
#     optimizer=torch.optim.Adam(m.parameters(), lr=1e-4),
#     targets=[
#         "catalyst",
#         "solvent_1",
#         "solvent_2",
#         "reagents_1",
#         "reagents_2",
#         "temperature",
#     ],
#     val_data=val_data,
# )


# # %%
# plt.plot(losses['sum']["train"], label="sum train"); plt.legend()
# plt.plot(losses['sum']["val"], label="sum val"); plt.legend()
# # %%
# plt.plot(losses['catalyst']["train"], label="catalyst train"); #plt.legend()
# plt.plot(losses['catalyst']["val"], label="catalyst val"); #plt.legend()
# plt.plot(losses['solvent_1']["train"], label="solvent_1 train"); #plt.legend()
# plt.plot(losses['solvent_1']["val"], label="solvent_1 val"); #plt.legend()
# plt.plot(losses['solvent_2']["train"], label="solvent_2 train"); #plt.legend()
# plt.plot(losses['solvent_2']["val"], label="solvent_2 val"); #plt.legend()
# plt.plot(losses['reagents_1']["train"], label="reagents_1 train"); #plt.legend()
# plt.plot(losses['reagents_1']["val"], label="reagents_1 val"); #plt.legend()
# plt.plot(losses['reagents_2']["train"], label="reagents_2 train"); #plt.legend()
# plt.plot(losses['reagents_2']["val"], label="reagents_2 val"); #plt.legend()
# plt.plot(losses['temperature']["train"], label="temperature train"); #plt.legend()
# plt.plot(losses['temperature']["val"], label="temperature val"); plt.legend()

# %%


# experiment to test if the model is configured correctly and the impact of adding more targets into the loss function

cut_off = 100000

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


targets=[
    "catalyst",
    "solvent_1",
    "solvent_2",
    "reagents_1",
    # "reagents_2",
    # "temperature",
]

_targets = []
_targets_cat_losses = {}
_targets_cat_accs = {}
for t in targets:
    print(t)
    _targets.append(t)

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

    losses, acc_metrics = train_loop(
        model=m, 
        train_data=train_data, 
        epochs=20,
        batch_size=0.25,
        loss_fn=torch.nn.CrossEntropyLoss(), 
        optimizer=torch.optim.Adam(m.parameters(), lr=1e-4),
        targets=_targets,
        val_data=val_data,
    )
    _targets_cat_losses[t] = losses['catalyst']
    _targets_cat_accs[t] = acc_metrics['catalyst']

    f,ax = plt.subplots(1,3)
    ax[0].plot(losses['sum']["train"], label="sum train")
    ax[0].plot(losses['sum']["val"], label="sum val")
    ax[1].plot(losses['catalyst']["train"], label="catalyst train")
    ax[1].plot(losses['catalyst']["val"], label="catalyst val")
    ax[2].plot(acc_metrics['catalyst']['top5']['train'], label="catalyst train 5acc")
    ax[2].plot(acc_metrics['catalyst']['top5']['val'], label="catalyst val 5acc")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()


# %%

f,ax = plt.subplots(2,2, figsize=(20,10))

for t in _targets_cat_losses:
    ax[0][0].plot(_targets_cat_losses[t]["train"], label=f"{t} train")
ax[0][0].legend()
ax[0][0].set_title("train loss")
for t in _targets_cat_losses:
    ax[0][1].plot(_targets_cat_losses[t]["val"], label=f"{t} val")
ax[0][1].legend()
ax[0][1].set_title("val loss")
for t in _targets_cat_accs:
    ax[1][0].plot(_targets_cat_accs[t]['top5']["train"], label=f"{t} train")
ax[1][0].legend()
ax[1][0].set_title("train acc")
for t in _targets_cat_accs:
    ax[1][1].plot(_targets_cat_accs[t]['top5']["val"], label=f"{t} val")
ax[1][1].legend()
ax[1][1].set_title("val acc")

# %%
