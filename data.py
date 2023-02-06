import pathlib
import multiprocessing
import typing
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from tqdm import tqdm

import fingerprint


def get_df(
    cleaned_data_path: pathlib.Path = pathlib.Path("data/ORD_USPTO/cleaned_data.pkl"),
    rxn_classes_path: pathlib.Path = pathlib.Path("data/ORD_USPTO/classified_rxn.smi"),
) -> pd.DataFrame:
    cleaned_df = pd.read_pickle(cleaned_data_path)

    print(f"{cleaned_df.shape=}")

    # read in the reaction classes
    with open(rxn_classes_path) as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines] # remove the \n at the end of each line

    # create df of the reaction classes
    # 2 columns: mapped_rxn, rxn_classes
    rxns = []
    rxn_classes = []
    for line in lines:
        try:
            rxn, rxn_class = line.split(' ')
            rxns += [rxn]
            rxn_classes += [rxn_class]
        except AttributeError:
            continue
        
    rxn_classes_df = pd.DataFrame(
        list(zip(rxns, rxn_classes)),
        columns=['mapped_rxn', 'rxn_class'],
    )

    # combine the two dfs
    data_df_temp = cleaned_df.merge(rxn_classes_df, how='inner', left_on='mapped_rxn_0', right_on='mapped_rxn')
    len(f"{data_df_temp=}")

    # I used the following command to generate the rxn classification:
    # ./namerxn -nomap data/mapped_rxn.smi data/classified_rxn.smi

    # The -nomap I thought would mean that it wouldn't change the atom mapping, yet it clearly did...
    # I'll just have to trust that namerxn didn't change the order of my reactions, and just append the reaction classes, and finally remove any reactions that couldn't be classified
    data_df = cleaned_df.copy().reset_index(drop=True)
    data_df['rxn_class'] = rxn_classes_df['rxn_class']
    data_df = data_df.dropna(subset=['rxn_class'])
    data_df.reset_index()
    print(f"{len(data_df)=}")

    # remove all the unclassified reactions, ie where rxn_class = '0.0'
    remove_unclassified_rxn_data_df = data_df[~data_df.rxn_class.str.contains("0.0")]
    print(f"{len(remove_unclassified_rxn_data_df)=}")


    # print out all catalysts
    #sorted(list(set(df['catalyst_0'].dropna())))

    # initialize a dict that maps catalysts to the humanly cleaned smiles
    catalyst_replacements = {}

    catalyst_wrong = []
    # All the data should have already been cleaned using rdkit.canonsmiles so I'm very surprised that there are some catalysts that are wrong. If you see any wrong catalysts, just remove them

    # Add a catalyst to the catalyst_replacements dict
    catalyst_replacements['CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+3].[Rh+3]'] = 'CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+2].[Rh+2]'
    catalyst_replacements['[CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+3]]'] = 'CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+2].[Rh+2]'
    catalyst_replacements['[CC(C)(C)[P]([Pd][P](C(C)(C)C)(C(C)(C)C)C(C)(C)C)(C(C)(C)C)C(C)(C)C]'] = 'CC(C)(C)[PH]([Pd][PH](C(C)(C)C)(C(C)(C)C)C(C)(C)C)(C(C)(C)C)C(C)(C)C'
    catalyst_replacements['CCCC[N+](CCCC)(CCCC)CCCC.CCCC[N+](CCCC)(CCCC)CCCC.CCCC[N+](CCCC)(CCCC)CCCC.[Br-].[Br-].[Br-]'] = 'CCCC[N+](CCCC)(CCCC)CCCC.[Br-]'
    catalyst_replacements['[CCO.CCO.CCO.CCO.[Ti]]'] = 'CCO[Ti](OCC)(OCC)OCC'
    catalyst_replacements['[CC[O-].CC[O-].CC[O-].CC[O-].[Ti+4]]'] = 'CCO[Ti](OCC)(OCC)OCC'
    catalyst_replacements['[Cl[Ni]Cl.c1ccc(P(CCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1]'] = 'Cl[Ni]1(Cl)[P](c2ccccc2)(c2ccccc2)CCC[P]1(c1ccccc1)c1ccccc1'
    catalyst_replacements['[Cl[Pd](Cl)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1]'] = 'Cl[Pd](Cl)([PH](c1ccccc1)(c1ccccc1)c1ccccc1)[PH](c1ccccc1)(c1ccccc1)c1ccccc1'
    catalyst_replacements['[Cl[Pd+2](Cl)(Cl)Cl.[Na+].[Na+]]'] = 'Cl[Pd]Cl'
    catalyst_replacements['Karstedt catalyst'] = 'C[Si](C)(C=C)O[Si](C)(C)C=C.[Pt]'
    catalyst_replacements["Karstedt's catalyst"] = 'C[Si](C)(C=C)O[Si](C)(C)C=C.[Pt]'
    catalyst_replacements['[O=C([O-])[O-].[Ag+2]]'] = 'O=C([O-])[O-].[Ag+].[Ag+]'
    catalyst_replacements['[O=S(=O)([O-])[O-].[Ag+2]]'] = 'O=S(=O)([O-])[O-].[Ag+].[Ag+]'
    catalyst_replacements['[O=[Ag-]]'] = 'O=[Ag]'
    catalyst_replacements['[O=[Cu-]]'] = 'O=[Cu]'
    catalyst_replacements['[Pd on-carbon]'] = '[C].[Pd]'
    catalyst_replacements['[TEA]'] = 'OCCN(CCO)CCO'
    catalyst_replacements['[Ti-superoxide]'] = 'O=[O-].[Ti]'
    catalyst_replacements['[[Pd].c1ccc(P(c2ccccc2)c2ccccc2)cc1]'] = '[Pd].c1ccc(P(c2ccccc2)c2ccccc2)cc1.c1ccc(P(c2ccccc2)c2ccccc2)cc1.c1ccc(P(c2ccccc2)c2ccccc2)cc1.c1ccc(P(c2ccccc2)c2ccccc2)cc1'
    catalyst_replacements['[c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd-4]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
    catalyst_replacements['[c1ccc([P]([Pd][P](c2ccccc2)(c2ccccc2)c2ccccc2)(c2ccccc2)c2ccccc2)cc1]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
    catalyst_replacements['[c1ccc([P](c2ccccc2)(c2ccccc2)[Pd]([P](c2ccccc2)(c2ccccc2)c2ccccc2)([P](c2ccccc2)(c2ccccc2)c2ccccc2)[P](c2ccccc2)(c2ccccc2)c2ccccc2)cc1]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
    catalyst_replacements['[sulfated tin oxide]'] = 'O=S(O[Sn])(O[Sn])O[Sn]'
    catalyst_replacements['[tereakis(triphenylphosphine)palladium(0)]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
    catalyst_replacements['[zeolite]'] = 'O=[Al]O[Al]=O.O=[Si]=O'

    # add any wrong catalysts you spot, e.g.
    catalyst_wrong += [
        'Catalyst A',
        'catalyst',
        'catalyst 1',
        'catalyst A',
        'catalyst VI',
        'reaction mixture',
        'same catalyst',
        'solution',
    ]

    # drop all rows that contain a 'catalyst_wrong
    df2 = data_df[~data_df["catalyst_0"].isin(catalyst_wrong)]

    # do the catalyst replacements that Alexander found
    df3 = df2.replace(catalyst_replacements)

    df3.reset_index(inplace=True)

    del df2

    count = 0
    for i in range(len(data_df['reagents_0'])):
        r = data_df['reagents_0'][i]
        if r ==r:
            if 'pd' in r or 'Pd' in r or 'palladium' in r or 'Palladium' in r:
                count +=1
    print('Number of Pd in the reagents columns: ', count )

    # Quite a few of the rows have Pd as a reagent. Probably worth going through all of them, and if the value in reagent_0 is already in catalyst_0, then replace the reagent value with np.NaN
    df3["reagents_0"] = df3.apply(lambda x: np.nan if (pd.notna(x["reagents_0"]) and pd.notna(x["catalyst_0"]) and x["reagents_0"] in x["catalyst_0"]) else x["reagents_0"], axis=1)
    df3["reagents_1"] = df3.apply(lambda x: np.nan if (pd.notna(x["reagents_1"]) and pd.notna(x["catalyst_0"]) and x["reagents_1"] in x["catalyst_0"]) else x["reagents_1"], axis=1)

    # That took care of a majority of the cases! Now there are only 9+7 cases left, just drop these rows
    df3 = df3[df3["reagents_0"] != '[Pd]']
    df3 = df3[df3["reagents_0"] != '[Pd+2]']
    df3 = df3[df3["reagents_1"] != '[Pd]']
    df3 = df3[df3["reagents_1"] != '[Pd+2]']
    df3 = df3.reset_index(drop=True)

    count = 0
    for i in range(len(df3['reagents_1'])):
        r = df3['reagents_1'][i]
        if r ==r:
            if 'Pd' in r:
                print(r)
                count +=1
    print(f'Number of Pd in the reagents columns: {count}')

    print(f"{len(df3)=}")

    df3['rxn_super_class'] = df3['rxn_class'].str.rsplit('.', expand=True)[0].astype(int)
    return df3


def generate_data(
    cleaned_data_path: pathlib.Path = pathlib.Path("data/ORD_USPTO/cleaned_data.pkl"),
    rxn_classes_path: pathlib.Path = pathlib.Path("data/ORD_USPTO/classified_rxn.smi"),
    diff_fp_path: typing.Optional[pathlib.Path] = pathlib.Path("data/ORD_USPTO/USPTO_rxn_diff_fp.pkl"),
    product_fp_path: typing.Optional[pathlib.Path] = pathlib.Path("data/ORD_USPTO/USPTO_product_fp.pkl"),
    radius:int=3,
    nBits:int=512,
):

    df = get_df(
        cleaned_data_path=cleaned_data_path,
        rxn_classes_path=rxn_classes_path,
    )

    # test_df = df['rxn_class'].str.rsplit(';', expand=True)
    # 2.5% of reactions have been assigned 2 reaction classes. 3 or 4 reaction classes is very rare.

    # calculate rxn difference fp
    # converting one 500k by 2k list to array takes roughly 15s, so the whole thing should take about 2-3 min
    # need to split into different cells for memory purposes

    num_cores = multiprocessing.cpu_count()
    inputs = tqdm(df['product_0'])
    p0 = Parallel(n_jobs=num_cores)(delayed(fingerprint.calc_fp_individual)(i, radius, nBits) for i in inputs)

    inputs = tqdm(df['product_1'])
    p1 = Parallel(n_jobs=num_cores)(delayed(fingerprint.calc_fp_individual)(i, radius, nBits) for i in inputs)

    inputs = tqdm(df['product_2'])
    p2 = Parallel(n_jobs=num_cores)(delayed(fingerprint.calc_fp_individual)(i, radius, nBits) for i in inputs)

    inputs = tqdm(df['product_3'])
    p3 = Parallel(n_jobs=num_cores)(delayed(fingerprint.calc_fp_individual)(i, radius, nBits) for i in inputs)

    ar_p0 = np.array(p0)
    ar_p1 = np.array(p1)
    ar_p2 = np.array(p2)
    ar_p3 = np.array(p3)

    product_fp = ar_p0 + ar_p1 + ar_p2 + ar_p3

    del ar_p0, ar_p1, ar_p2, ar_p3
    del p0, p1, p2, p3

    num_cores = multiprocessing.cpu_count()
    inputs = tqdm(df['reactant_0'])
    r0 = Parallel(n_jobs=num_cores)(delayed(fingerprint.calc_fp_individual)(i, radius, nBits) for i in inputs)

    inputs = tqdm(df['reactant_1'])
    r1 = Parallel(n_jobs=num_cores)(delayed(fingerprint.calc_fp_individual)(i, radius, nBits) for i in inputs)

    inputs = tqdm(df['reactant_2'])
    r2 = Parallel(n_jobs=num_cores)(delayed(fingerprint.calc_fp_individual)(i, radius, nBits) for i in inputs)

    inputs = tqdm(df['reactant_3'])
    r3 = Parallel(n_jobs=num_cores)(delayed(fingerprint.calc_fp_individual)(i, radius, nBits) for i in inputs)

    ar_r0 = np.array(r0)
    ar_r1 = np.array(r1)
    ar_r2 = np.array(r2)
    ar_r3 = np.array(r3)

    rxn_diff_fp = product_fp - ar_r0 - ar_r1 - ar_r2 - ar_r3

    del ar_r0, ar_r1, ar_r2, ar_r3
    del r0, r1, r2, r3

    #save to pickle
    if diff_fp_path is not None:
        np.save(diff_fp_path, rxn_diff_fp)
    if product_fp_path is not None:
        np.save(product_fp_path, product_fp)
    return rxn_diff_fp, product_fp
    

def get_classified_rxn_data_mask(data_df):
    return ~data_df.rxn_class.str.contains("0.0")
    
if __name__ == "__main__":
    generate_data(
        cleaned_data_path=pathlib.Path("data/ORD_USPTO/cleaned_data.pkl"),
        rxn_classes_path=pathlib.Path("data/ORD_USPTO/classified_rxn.smi"),
        diff_fp_path=pathlib.Path("data/ORD_USPTO/USPTO_rxn_diff_fp.pkl"),
        product_fp_path=pathlib.Path("data/ORD_USPTO/USPTO_product_fp.pkl"),
        radius=3,
        nBits=512,
    )