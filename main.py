# %%
import pathlib
import typing
import numpy as np
import pandas as pd
import torch
import torchmetrics

import data
import ohe
import model

from tqdm import trange

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

def train_loop(model, train_data, *, epochs, batch_size, loss_fn, optimizer):#, report_freq, scheduler=None, early_stopper=None, train_cluster_ids_for_downstream=None, train_similarity_dist=None, x_val=None, y_val=None, val_cluster_ids_for_downstream=None, val_similarity_dist=None):
    
    # acc_metrics = {}
    # for feature in train_data:
    #     num_classes=train_data[feature].shape[1]
    #     acc_metrics[feature] = {
    #         "top1": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1),
    #         "top3": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=3),
    #         "top5": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5),
    #     }

    train_size = train_data["product_fp"].shape[0]

    train_losses = {
        "sum": [],
        "catalyst": [],
        "solvent_1": [],
        "solvent_2": [],
        "reagents_1": [],
        "reagents_2": [],
        "temperature": [],
    }
    
    for e in (t := trange(epochs, desc='', leave=True)):
    
        idxes = np.arange(train_size)
        np.random.shuffle(idxes)

        prev_idx = 0
        interval = int(idxes.shape[0] * batch_size)

        # epoch_train_acc = []
        # epoch_train_acc_top3 = []
        # epoch_train_acc_top5 = []

        epoch_train_losses = {
            "sum": [],
            "catalyst": [],
            "solvent_1": [],
            "solvent_2": [],
            "reagents_1": [],
            "reagents_2": [],
            "temperature": [],
        }

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
            train_catalyst_pred, train_solvent_1_pred, train_solvent_2_pred, train_reagents_1_pred, train_reagents_2_pred, train_temperature_pred = output

            catalyst_loss = loss_fn(train_catalyst_pred, train_data['catalyst'][batch_idxes])
            solvent_1_loss = loss_fn(train_solvent_1_pred, train_data['solvent_1'][batch_idxes])
            solvent_2_loss = loss_fn(train_solvent_2_pred, train_data['solvent_2'][batch_idxes])
            reagents_1_loss = loss_fn(train_reagents_1_pred, train_data['reagents_1'][batch_idxes])
            reagents_2_loss = loss_fn(train_reagents_2_pred, train_data['reagents_2'][batch_idxes])
            temp_loss = loss_fn(train_temperature_pred, train_data['temperature'][batch_idxes])

            # print(catalyst_loss, solvent_1_loss, solvent_2_loss, reagents_1_loss, reagents_2_loss, temp_loss)

            loss = catalyst_loss + solvent_1_loss + solvent_2_loss + reagents_1_loss + reagents_2_loss + 1e-4 * temp_loss
            # print(loss)

            epoch_train_losses["sum"].append(loss.detach().numpy().item())
            epoch_train_losses["catalyst"].append(catalyst_loss.detach().numpy().item())
            epoch_train_losses["solvent_1"].append(solvent_1_loss.detach().numpy().item())
            epoch_train_losses["solvent_2"].append(solvent_2_loss.detach().numpy().item())
            epoch_train_losses["reagents_1"].append(reagents_1_loss.detach().numpy().item())
            epoch_train_losses["reagents_2"].append(reagents_2_loss.detach().numpy().item())
            epoch_train_losses["temperature"].append(temp_loss.detach().numpy().item())

            t.set_description(str(epoch_train_losses["sum"][-1]), refresh=True)
    
            # epoch_train_acc.append(acc_metric_top1(pred_train, y_train[batch_idxes].argmax(axis=1)).item())
            # epoch_train_acc_top3.append(acc_metric_top3(pred_train, y_train[batch_idxes].argmax(axis=1)).item())
            # epoch_train_acc_top5.append(acc_metric_top5(pred_train, y_train[batch_idxes].argmax(axis=1)).item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_losses["sum"].append(np.mean(epoch_train_losses["sum"]))
        train_losses["catalyst"].append(np.mean(epoch_train_losses["catalyst"]))
        train_losses["solvent_1"].append(np.mean(epoch_train_losses["solvent_1"]))
        train_losses["solvent_2"].append(np.mean(epoch_train_losses["solvent_2"]))
        train_losses["reagents_1"].append(np.mean(epoch_train_losses["reagents_1"]))
        train_losses["reagents_2"].append(np.mean(epoch_train_losses["reagents_2"]))
        train_losses["temperature"].append(np.mean(epoch_train_losses["temperature"]))
        
        # print("___________________________")

        # train_acc_trajectory.append(np.mean(epoch_train_acc))
        # train_acc_trajectory_top3.append(np.mean(epoch_train_acc_top3))
        # train_acc_trajectory_top5.append(np.mean(epoch_train_acc_top5))
        
        # if report_freq:
        #     if (e % report_freq == 0) or (scheduler is not None) or (early_stopper is not None):
        #         verbose_str = f"Epoch: {e} Train [ Loss: {loss.detach().numpy().item():.5f} | Acc (Top 1): {train_acc_trajectory[-1]:.5f} | Acc (Top 3): {train_acc_trajectory_top3[-1]:.5f} | Acc (Top 5): {train_acc_trajectory_top5[-1]:.5f}]"
        #         report_epochs.append(e)
        #         train_loss_trajectory.append(loss.detach().numpy().item())

        #         if (x_val is not None) and (y_val is not None):
        #             with torch.no_grad():
        #                 if val_cluster_ids_for_downstream is None:
        #                     pred_val = model(x_val, training=False)
        #                 else:
        #                     pred_val = model(x_val, val_cluster_ids_for_downstream, training=False)
        #                 if val_similarity_dist == None:
        #                     loss_val = loss_fn(pred_val, y_val)
        #                 else:
        #                     loss_val = loss_fn(pred_val, y_val, val_similarity_dist)
                        

        #                 val_loss_trajectory.append(loss_val.detach().numpy().item())
        #                 val_acc = acc_metric_top1(pred_val, y_val.argmax(axis=1)).item()
        #                 val_acc_trajectory.append(val_acc)
        #                 val_acc_top5 = acc_metric_top5(pred_val, y_val.argmax(axis=1)).item()
        #                 val_acc_trajectory_top5.append(val_acc_top5)
        #                 val_acc_top3 = acc_metric_top3(pred_val, y_val.argmax(axis=1)).item()
        #                 val_acc_trajectory_top3.append(val_acc_top3)
        #                 verbose_str += f"\t Validation [ Loss: {loss_val.detach().numpy().item():.5f} | Acc (Top 1): {val_acc:.5f} | Acc (Top 3): {val_acc_top3:.5f} | Acc (Top 5): {val_acc_top5: .5f}]"
        #                 if scheduler is not None:
        #                     scheduler.step(loss_val)
        #                 if early_stopper is not None:
        #                     if early_stopper.early_stop(loss_val):             
        #                         break
        #         # print(verbose_str)
                
        #         t.set_description(verbose_str, refresh=True)

    # train_traj = {
    #     ("train","loss"): train_loss_trajectory, ("train","acc"): train_acc_trajectory,  ("train","acc_top3"): train_acc_trajectory_top3, ("train","acc_top5"): train_acc_trajectory_top5
    # }

    # if val_loss_trajectory:
    #     val_traj = {
    #         ("val","loss"): val_loss_trajectory, ("val","acc"): val_acc_trajectory, ("val","acc_top3"): val_acc_trajectory_top3, ("val","acc_top5"): val_acc_trajectory_top5
    #     }
    #     return pd.DataFrame({**train_traj, **val_traj}, index=report_epochs)
    # else:
    #     return pd.DataFrame(train_traj, index=report_epochs)
    return train_losses


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

train_data = {
    "product_fp": train_product_fp,
    "rxn_diff_fp": train_rxn_diff_fp,
    "catalyst": train_catalyst,
    "solvent_1": train_solvent_0,
    "solvent_2": train_solvent_1,
    "reagents_1": train_reagents_0,
    "reagents_2": train_reagents_1,
    "temperature": train_temperature,
}


print("STARTING LOOP")

train_losses = train_loop(
    model=m, 
    train_data=train_data, 
    epochs=20,
    batch_size=0.05,
    loss_fn=torch.nn.CrossEntropyLoss(), 
    optimizer=torch.optim.Adam(m.parameters(), lr=1e-4),
)

print("DONE LOOP")

import matplotlib.pyplot as plt
# %%
plt.plot(train_losses['sum'], label="sum"); plt.legend()
# %%
plt.plot(train_losses['catalyst'], label="catalyst"); plt.legend()
# %%
plt.plot(train_losses['solvent_1'], label="solvent_1"); plt.legend()
# %%
plt.plot(train_losses['solvent_2'], label="solvent_2"); plt.legend()
# %%
plt.plot(train_losses['reagents_1'], label="reagents_1"); plt.legend()
# %%
plt.plot(train_losses['reagents_2'], label="reagents_2"); plt.legend()
# %%
plt.plot(train_losses['temperature'], label="temperature"); plt.legend()
# %%
