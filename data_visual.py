# %%
import pathlib
import pandas as pd

import matplotlib.pyplot as plt

import param_sharing.reactions.get

# %%

df = param_sharing.reactions.get.get_reaction_df(
    cleaned_data_path=pathlib.Path("data/ORD_USPTO/cleaned_data.pkl"),
    rxn_classes_path=pathlib.Path("data/ORD_USPTO/classified_rxn.smi"),
    verbose=True,
)

# %%

df.columns

# %%

df.describe()

# %%

for i in df.columns:
    col = df[i]
    print(
        f"{i} ({col.dtype}): % nans {col.isna().sum()} / {len(col)} = {100*(col.isna().sum()/len(col))}%"
    )
    col.value_counts().hist(log=True)
    plt.gca().set_title("Histogram of value counts")
    plt.show()
    print(col.value_counts())
    print("-" * 50)
    print()
# %%


print("react 0 == 1:", (df["reactant_0"] == df["reactant_1"]).sum())

print(
    "react 0 == 1 == 2:",
    (
        (df["reactant_0"] == df["reactant_1"]) & (df["reactant_1"] == df["reactant_2"])
    ).sum(),
)

print(
    "react 0 == 1 == 2 == 3:",
    (
        (df["reactant_0"] == df["reactant_1"])
        & (df["reactant_0"] == df["reactant_2"])
        & (df["reactant_0"] == df["reactant_3"])
    ).sum(),
)
# %%

print("reage 0 == 1:", (df["reagents_0"] == df["reagents_1"]).sum())
# %%

print("solvent 0 == 1:", (df["solvent_0"] == df["solvent_1"]).sum())


# %%

print("product 0 == 1:", (df["product_0"] == df["product_1"]).sum())

print("product 1 == 2:", (df["product_2"] == df["product_1"]).sum())

print("product 2 == 3:", (df["product_2"] == df["product_3"]).sum())

print("product 0 == 3:", (df["product_0"] == df["product_3"]).sum())

print(
    "product 0 == 1 == 2:",
    ((df["product_0"] == df["product_1"]) & (df["product_1"] == df["product_2"])).sum(),
)

print(
    "product 0 == 1 == 2 == 3:",
    (
        (df["product_0"] == df["product_1"])
        & (df["product_0"] == df["product_2"])
        & (df["product_0"] == df["product_3"])
    ).sum(),
)
# %%
