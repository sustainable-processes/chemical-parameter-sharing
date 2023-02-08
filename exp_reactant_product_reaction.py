# %%
import numpy as np
import pandas as pd
import torch
import typing
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter


import src.learn.model

#unpickle
rxn_diff_fp = np.load("data/ORD_USPTO/USPTO_rxn_diff_fp.pkl.npy", allow_pickle=True)
product_fp = np.load("data/ORD_USPTO/USPTO_product_fp.pkl.npy", allow_pickle=True)
reactant_fp = np.load("data/ORD_USPTO/USPTO_reactant_fp.pkl.npy", allow_pickle=True)

assert rxn_diff_fp.shape[0] == product_fp.shape[0]
assert rxn_diff_fp.shape[0] == reactant_fp.shape[0]

rng = np.random.default_rng(12345)

indexes = np.arange(rxn_diff_fp.shape[0])
rng.shuffle(indexes)

train_test_split = 0.8
train_val_split = 0.8

test_idx = indexes[int(rxn_diff_fp.shape[0] * train_test_split):]
train_val_idx = indexes[:int(rxn_diff_fp.shape[0] * train_test_split)]
train_idx = train_val_idx[:int(train_val_idx.shape[0] * train_val_split)]
val_idx = train_val_idx[int(train_val_idx.shape[0] * train_val_split):]

# %%

train_product_fp = torch.Tensor(product_fp[train_idx])
train_reactant_fp = torch.Tensor(reactant_fp[train_idx])
train_rxn_diff_fp = torch.Tensor(rxn_diff_fp[train_idx])
val_product_fp = torch.Tensor(product_fp[val_idx])
val_reactant_fp = torch.Tensor(reactant_fp[val_idx])
val_rxn_diff_fp = torch.Tensor(rxn_diff_fp[val_idx])

# %%

print(
    train_product_fp.shape,
    train_reactant_fp.shape,
    train_rxn_diff_fp.shape,
)

# %%


class MergeModel(torch.nn.Module):

    def __init__(
        self, 
        *, 
        model_1_input_dim: int, 
        model_2_input_dim: int, 
        mid_dim: int, 
        output_dim: int, 
        upstream_model_kwargs={},
        downstream_model_kwargs={},
    ):
        super(MergeModel, self).__init__()
        self.model_1_input_dim = model_1_input_dim
        self.model_2_input_dim = model_2_input_dim
        self.mid_dim = mid_dim
        self.output_dim = output_dim

        self.model_1 = src.learn.model.SimpleMLP(
            input_dim=model_1_input_dim, 
            output_dim=mid_dim,
            **upstream_model_kwargs,
        )
        
        self.model_2 = src.learn.model.SimpleMLP(
            input_dim=model_2_input_dim, 
            output_dim=mid_dim,
            **upstream_model_kwargs,
        )

        self.downstream_model = src.learn.model.SimpleMLP(
            input_dim=mid_dim+mid_dim, 
            output_dim=output_dim,
            **downstream_model_kwargs,
        )

    def forward(self, input_1, input_2, training=False):
        output_1 = self.model_1(input_1, training=training)
        output_2 = self.model_2(input_2, training=training)
        mid_input = torch.cat((output_1, output_2), dim=1)
        return self.downstream_model(mid_input, training=training)



def get_batch_size(size, length):
    if isinstance(size, int):
        return size / length
    elif isinstance(size, float):
        assert (0. < size) & (size <= 1.0)
        return size


def train_loop(model, x, y, *, epochs, batch_size: typing.Union[float, int], loss_fn, optimizer, 
               val_data=None, train_kwargs: dict = {}, val_kwargs: dict={}, train_eval: bool=True,
               write_summary=True, eval_freq=1):
    if write_summary:
        writer = SummaryWriter()
    train_size = x[0].shape[0]
    batch_size = get_batch_size(batch_size, length=train_size)

    losses = {"train": [], "val": [], "train_eval": []}

    for e in range(epochs):

        output_str = f"{e+1}/{epochs} | "
    
        idxes = np.arange(train_size)
        np.random.shuffle(idxes)

        prev_idx = 0
        interval = int(idxes.shape[0] * batch_size)

        # storage for use during mini-batching, reset every epoch
        epoch_losses = {"train": []}

        # run across training
        for idx in (t := trange(interval, idxes.shape[0]+1, interval, desc='', leave=True)):
            if batch_size < 1.0:
                batch_idxes = idxes[prev_idx:idx]
            else:
                batch_idxes = idxes
            prev_idx = idx

            pred = model.forward(
                *[i[batch_idxes] for i in x],
                training=True,
                **train_kwargs,
            )
            loss = loss_fn(pred, y[batch_idxes])    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses["train"].append(loss.detach().numpy().item())

        # calc the average train loss
        losses["train"].append(np.mean(epoch_losses["train"]))
        output_str += f'Train loss: {losses["train"][-1]:.3f}'

        if write_summary:
            writer.add_scalar('Loss/train', losses["train"][-1], e)

        if e % eval_freq == 0:
            # evaluate with train data
            if train_eval:
                with torch.no_grad():
                    pred = model.forward(
                        *x,
                        training=False,
                        **val_kwargs,
                    )

                    loss = loss_fn(pred, y)
                    losses["train_eval"].append(loss.detach().numpy().item()) 
                    output_str += f' | Train eval loss: {losses["train_eval"][-1]:.3f} '
                if write_summary:
                    writer.add_scalar('Loss/train eval', losses["train_eval"][-1], e)

            # evaluate with validation data
            if val_data is not None:
                with torch.no_grad():
                    pred = model.forward(
                        *val_data[0],
                        training=False,
                        **val_kwargs,
                    )

                    loss = loss_fn(pred, val_data[1]) 
                    losses["val"].append(loss.detach().numpy().item()) 
                    output_str += f' | Val loss: {losses["val"][-1]:.3f} '
                if write_summary:
                    writer.add_scalar('Loss/val', losses["val"][-1], e)
    
        print(output_str)
    return losses


# %%

_input_1 = train_product_fp
_input_2 = train_reactant_fp
_target = train_rxn_diff_fp
kwargs = {
    "hidden_dims":[300, 300, 300],
    "hidden_acts":[torch.nn.ReLU, torch.nn.ReLU],
    "output_act":torch.nn.ReLU,
    "use_batchnorm":True,
    "dropout_prob":0.2,
}

m = MergeModel(
    model_1_input_dim=_input_1.shape[1], 
    model_2_input_dim=_input_2.shape[1], 
    mid_dim=_input_1.shape[1]//3, 
    output_dim=_target.shape[1], 
    upstream_model_kwargs=kwargs,
    downstream_model_kwargs=kwargs,
)
# %%

losses = train_loop(
    m, x=(_input_1, _input_2), y=_target,
    epochs=10, batch_size=256, 
    loss_fn=torch.nn.MSELoss(),
    optimizer=torch.optim.Adam(m.parameters(), lr=1e-4),
    val_data=((val_product_fp, val_product_fp), val_rxn_diff_fp), 
    train_eval=False,
    eval_freq=2,
)
# %%


plt.plot(losses["train"], label="train")
plt.plot(losses["train_eval"], label="train eval")
plt.plot(losses["val"], label="val")
plt.legend()
plt.show()

# %%

