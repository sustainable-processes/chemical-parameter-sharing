# %%
import pandas as pd
import pathlib

dataDir = "../data/"


def read_uspto_data(
    file_loc=pathlib.Path(dataDir+"1976_Sep2016_USPTOgrants_smiles.rsmi")
):
    with open(file_loc,'r') as f:
        uspto = f.readlines()

    uspto = [s.replace("\n", "").split("\t") for s in uspto]
    uspto = pd.DataFrame(uspto[1:], columns=uspto[0]).set_index("PatentNumber")
    return uspto
# %%
