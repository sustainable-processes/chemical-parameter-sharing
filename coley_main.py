# %%
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf

import src.reactions.get
import src.learn.ohe

import src.coley_code.model


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

train_test_split = 0.1
train_val_split = 0.1

test_idx = indexes[int(df.shape[0] * train_test_split):]
train_val_idx = indexes[:int(df.shape[0] * train_test_split)]
train_idx = train_val_idx[:int(train_val_idx.shape[0] * train_val_split)]
val_idx = train_val_idx[int(train_val_idx.shape[0] * train_val_split):]

assert rxn_diff_fp.shape == product_fp.shape

train_product_fp = tf.convert_to_tensor(product_fp[train_idx])
train_rxn_diff_fp = tf.convert_to_tensor(rxn_diff_fp[train_idx])
val_product_fp = tf.convert_to_tensor(product_fp[val_idx])
val_rxn_diff_fp = tf.convert_to_tensor(rxn_diff_fp[val_idx])

train_catalyst, val_catalyst, cat_enc = src.learn.ohe.apply_train_ohe_fit(df[['catalyst_0']], train_idx, val_idx, tensor_func=tf.convert_to_tensor)
train_solvent_0, val_solvent_0, sol0_enc = src.learn.ohe.apply_train_ohe_fit(df[['solvent_0']], train_idx, val_idx, tensor_func=tf.convert_to_tensor)
train_solvent_1, val_solvent_1, sol1_enc = src.learn.ohe.apply_train_ohe_fit(df[['solvent_1']], train_idx, val_idx, tensor_func=tf.convert_to_tensor)
train_reagents_0, val_reagents_0, reag0_enc = src.learn.ohe.apply_train_ohe_fit(df[['reagents_0']], train_idx, val_idx, tensor_func=tf.convert_to_tensor)
train_reagents_1, val_reagents_1, reag1_enc = src.learn.ohe.apply_train_ohe_fit(df[['reagents_1']], train_idx, val_idx, tensor_func=tf.convert_to_tensor)
train_temperature = tf.convert_to_tensor(df['temperature_0'].iloc[train_idx].fillna(0.))
val_temperature = tf.convert_to_tensor(df['temperature_0'].iloc[val_idx].fillna(0.))

# del df

print("Loaded data")

# %%

x_train_data = (
    train_product_fp, 
    train_rxn_diff_fp, 
    train_catalyst, 
    train_reagents_0, 
    train_reagents_1, 
    train_solvent_0, 
    train_solvent_1, 
    # train_temperature, 
)

y_train_data = (
    train_catalyst, 
    train_reagents_0, 
    train_reagents_1, 
    train_solvent_0, 
    train_solvent_1, 
    train_temperature, 
)

x_val_data = (
    val_product_fp, 
    val_rxn_diff_fp, 
    val_catalyst, 
    val_reagents_0, 
    val_reagents_1, 
    val_solvent_0, 
    val_solvent_1, 
)

y_val_data = (
    val_catalyst, 
    val_reagents_0, 
    val_reagents_1, 
    val_solvent_0, 
    val_solvent_1, 
    val_temperature, 
)

# %%

model = src.coley_code.model.build(
    pfp_len=train_product_fp.shape[-1], 
    rxnfp_len=train_rxn_diff_fp.shape[-1], 
    c1_dim=train_catalyst.shape[-1], # TODO check not top 100 
    s1_dim=train_solvent_0.shape[-1], 
    s2_dim=train_solvent_1.shape[-1], 
    r1_dim=train_reagents_0.shape[-1], 
    r2_dim=train_reagents_1.shape[-1], 
    N_h1=1024, 
    N_h2=100,
    l2v=0, # TODO check what coef they used
    lr=1e-4,
)

def mse_ignore_na(y_true, y_pred):
    raise ValueError("This function messes up with autograd")
    return tf.reduce_mean(tf.keras.backend.switch(tf.math.is_nan(y_true), tf.zeros_like(y_true),(y_pred - y_true)**2))


model.compile(
    loss=[
        tf.keras.losses.CategoricalCrossentropy(),
        tf.keras.losses.CategoricalCrossentropy(),
        tf.keras.losses.CategoricalCrossentropy(),
        tf.keras.losses.CategoricalCrossentropy(),
        tf.keras.losses.CategoricalCrossentropy(),
        tf.keras.losses.MeanSquaredError(),
    ], 
    loss_weights=[1, 1, 1, 1, 1, 1e-4], 
    optimizer='adam',
)

h = model.fit(
    x=x_train_data, y=y_train_data, epochs=20, verbose=1, validation_data=(x_val_data, y_val_data),
)


# %%

import matplotlib.pyplot as plt

plt.plot(h.history['loss'], label="loss")
plt.plot(h.history['val_loss'], label="val_loss")
plt.legend()

# %%
