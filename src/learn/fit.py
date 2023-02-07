import numpy as np
import torch
import torchmetrics
from tqdm import trange


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

            pred = model.forward_dict(
                data=train_data,
                indexes=batch_idxes,
                training=True,
            )

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
                pred = model.forward_dict(
                    data=val_data,
                    indexes=slice(None),
                    training=False,
                )

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
