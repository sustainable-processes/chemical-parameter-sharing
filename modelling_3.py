from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.rdBase import BlockLogs
import pandas as pd

import torch
import torchmetrics
from tqdm import trange
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def calc_fp(lst: list, radius=3, nBits=2048):
    # Usage: 
    # radius = 3
    # nBits = 2048
    # p0 = calc_fp(data_df['product_0'][:10000], radius=radius, nBits=nBits)
    
    ans = []
    for i in tqdm(lst):
        #convert to mole object
        try:
            block = BlockLogs()
            mol = Chem.MolFromSmiles(i)
            # We are using hashed fingerprint, becasue an unhased FP has length: 4294967295
            fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=nBits)
            array = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, array)
            ans += [array]
        except:
            ans += [np.zeros((nBits,), dtype=int)]
    return ans

def calc_fp_individual(smiles: str, radius=3, nBits=2048):
    #ans = []
    try:
        block = BlockLogs()
        mol = Chem.MolFromSmiles(smiles)
        # We are using hashed fingerprint, becasue an unhased FP has length: 4294967295
        fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=nBits)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, array)
        return array
    except:
        array = np.zeros((nBits,), dtype=int)
        return array

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.verbose = verbose

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.verbose:
                print(f"Testing my patience {self.counter}/{self.patience}: {validation_loss:.6f} > {self.min_validation_loss + self.min_delta:.6f}")
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"Patience of {self.patience} reached, early stopping")
                return True
        return False


class FullyConnectedReactionModel(torch.nn.Module):

    def __init__(
        self, 
        *, 
        input_dim, 
        hidden_dims, 
        target_dim,
        hidden_act, 
        output_act, 
        use_batchnorm, 
        dropout_prob
    ):
        super(FullyConnectedReactionModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, dim))
            layers.append(hidden_act())
            if use_batchnorm:
                layers.append(torch.nn.BatchNorm1d(dim))
            if dropout_prob > 0:
                layers.append(torch.nn.Dropout(p=dropout_prob))
            prev_dim = dim
        layers.append(torch.nn.Linear(prev_dim, target_dim))
        if output_act is torch.nn.Softmax:
            layers.append(output_act(dim=1))
        else:
            layers.append(output_act())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, training=True):

        if training:
            self.train()
        else:
            self.eval()

        return self.layers(x)



def train_loop(model, x_train, y_train, *, epochs, batch_size, loss_fn, optimizer, report_freq, scheduler=None, early_stopper=None, train_cluster_ids_for_downstream=None, train_similarity_dist=None, x_val=None, y_val=None, val_cluster_ids_for_downstream=None, val_similarity_dist=None):
    
    report_epochs = []
    train_loss_trajectory = []
    val_loss_trajectory = []
    train_acc_trajectory = []
    val_acc_trajectory = []
    train_acc_trajectory_top5 = []
    val_acc_trajectory_top5 = []
    train_acc_trajectory_top3 = []
    val_acc_trajectory_top3 = []

    if report_freq and (x_val is not None) and (y_val is not None):
        # if we are reporting the validation data should exist
        if train_cluster_ids_for_downstream is not None:
            assert val_cluster_ids_for_downstream is not None

        if train_similarity_dist is not None:
            assert val_similarity_dist is not None

    acc_metric_top1 = torchmetrics.Accuracy(task="multiclass", num_classes=y_train.shape[1], top_k=1)
    acc_metric_top3 = torchmetrics.Accuracy(task="multiclass", num_classes=y_train.shape[1], top_k=3)
    acc_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=y_train.shape[1], top_k=5)

    
    for e in (t := trange(epochs, desc='', leave=True)):
    # for e in range(epochs):
        idxes = np.arange(x_train.shape[0])
        np.random.shuffle(idxes)

        prev_idx = 0
        interval = int(idxes.shape[0] * batch_size)

        epoch_train_acc = []
        epoch_train_acc_top3 = []
        epoch_train_acc_top5 = []

        for idx in range(interval, idxes.shape[0]+1, interval):
            if batch_size < 1.0:
                batch_idxes = idxes[prev_idx:idx]
            else:
                batch_idxes = idxes
            prev_idx = idx

            if train_cluster_ids_for_downstream is None:
                pred_train = model(x_train[batch_idxes], training=True)
            else:
                pred_train = model(x_train[batch_idxes], train_cluster_ids_for_downstream[batch_idxes], training=True)
            if train_similarity_dist == None:
                loss = loss_fn(pred_train, y_train[batch_idxes])
            else:
                loss = loss_fn(pred_train, y_train[batch_idxes], train_similarity_dist[batch_idxes])
            
            epoch_train_acc.append(acc_metric_top1(pred_train, y_train[batch_idxes].argmax(axis=1)).item())
            epoch_train_acc_top3.append(acc_metric_top3(pred_train, y_train[batch_idxes].argmax(axis=1)).item())
            epoch_train_acc_top5.append(acc_metric_top5(pred_train, y_train[batch_idxes].argmax(axis=1)).item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc_trajectory.append(np.mean(epoch_train_acc))
        train_acc_trajectory_top3.append(np.mean(epoch_train_acc_top3))
        train_acc_trajectory_top5.append(np.mean(epoch_train_acc_top5))
        
        if report_freq:
            if (e % report_freq == 0) or (scheduler is not None) or (early_stopper is not None):
                verbose_str = f"Epoch: {e} Train [ Loss: {loss.detach().numpy().item():.5f} | Acc (Top 1): {train_acc_trajectory[-1]:.5f} | Acc (Top 3): {train_acc_trajectory_top3[-1]:.5f} | Acc (Top 5): {train_acc_trajectory_top5[-1]:.5f}]"
                report_epochs.append(e)
                train_loss_trajectory.append(loss.detach().numpy().item())

                if (x_val is not None) and (y_val is not None):
                    with torch.no_grad():
                        if val_cluster_ids_for_downstream is None:
                            pred_val = model(x_val, training=False)
                        else:
                            pred_val = model(x_val, val_cluster_ids_for_downstream, training=False)
                        if val_similarity_dist == None:
                            loss_val = loss_fn(pred_val, y_val)
                        else:
                            loss_val = loss_fn(pred_val, y_val, val_similarity_dist)
                        

                        val_loss_trajectory.append(loss_val.detach().numpy().item())
                        val_acc = acc_metric_top1(pred_val, y_val.argmax(axis=1)).item()
                        val_acc_trajectory.append(val_acc)
                        val_acc_top5 = acc_metric_top5(pred_val, y_val.argmax(axis=1)).item()
                        val_acc_trajectory_top5.append(val_acc_top5)
                        val_acc_top3 = acc_metric_top3(pred_val, y_val.argmax(axis=1)).item()
                        val_acc_trajectory_top3.append(val_acc_top3)
                        verbose_str += f"\t Validation [ Loss: {loss_val.detach().numpy().item():.5f} | Acc (Top 1): {val_acc:.5f} | Acc (Top 3): {val_acc_top3:.5f} | Acc (Top 5): {val_acc_top5: .5f}]"
                        if scheduler is not None:
                            scheduler.step(loss_val)
                        if early_stopper is not None:
                            if early_stopper.early_stop(loss_val):             
                                break
                # print(verbose_str)
                
                t.set_description(verbose_str, refresh=True)

    train_traj = {
        ("train","loss"): train_loss_trajectory, ("train","acc"): train_acc_trajectory,  ("train","acc_top3"): train_acc_trajectory_top3, ("train","acc_top5"): train_acc_trajectory_top5
    }

    if val_loss_trajectory:
        val_traj = {
            ("val","loss"): val_loss_trajectory, ("val","acc"): val_acc_trajectory, ("val","acc_top3"): val_acc_trajectory_top3, ("val","acc_top5"): val_acc_trajectory_top5
        }
        return pd.DataFrame({**train_traj, **val_traj}, index=report_epochs)
    else:
        return pd.DataFrame(train_traj, index=report_epochs)
