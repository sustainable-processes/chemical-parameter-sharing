# Clean the USPTO data

"""
# Usage:
python USPTO_cleaning.py 
NB: Requires all the pickle files created by USPTO_extraction.py

# Output:
1) A pickle file with the cleaned data in a pickle file

# Functionality:
1) Merge all the data to one big df
#####2) Move reagents that are also catalysts to the catalyst column
3) Remove reactions where the reagent is Pd
4) Remove reactions with too many components
5) Remove reactions with rare molecules
6) Remove reactions with inconsistent yields
7) Handle molecules with names instead of SMILES
8) Remove duplicate reactions
9) Save the cleaned data to a pickle file

"""

#Still need to implement:
## What should we do with the reactions where the reagent is Pd?
## What should we do when the same molecule appears as both a reagent and a catalyst?


# Imports
import sys
import pandas as pd
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from rdkit import Chem
import pickle
from datetime import datetime
import numpy as np


def merge_pickles():
    #create one big df of all the pickled data
    folder_path = 'data/USPTO/pickled_data/'
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    full_df = pd.DataFrame()
    for file in tqdm(onlyfiles):
        if file[0] != '.': #We don't want to try to unpickle .DS_Store
            filepath = folder_path+file 
            unpickled_df = pd.read_pickle(filepath)
            full_df = pd.concat([full_df, unpickled_df], ignore_index=True)
            
    return full_df

def remove_reactions_with_too_many_of_component(df, component_name, number_of_columns_to_keep):
    
    cols = list(df.columns)
    count = 0
    for col in cols:
        if component_name in col:
            count += 1
    
    columns = []
    for i in range(count):
        if i >= number_of_columns_to_keep:
            columns += [component_name+str(i)]
            
    for col in columns:
        df = df[pd.isnull(df[col])]
        
    df = df.drop(columns, axis=1)
            
    return df

def remove_rare_molecules(df, columns: list, cutoff: int):
    # Remove reactions that include a rare molecule (ie it appears 3 times or fewer)
    
    if len(columns) == 1:
        # Get the count of each value
        value_counts = df[columns[0]].value_counts()
        to_remove = value_counts[value_counts <= cutoff].index
        # Keep rows where the column is not in to_remove
        
        df2 = df[~df[columns[0]].isin(to_remove)]
        return df2
    
    elif len(columns) ==2:
        # Get the count of each value
        value_counts_0 = df[columns[0]].value_counts()
        value_counts_1 = df[columns[1]].value_counts()
        value_counts_2 = value_counts_0.add(value_counts_1, fill_value=0)

        # Select the values where the count is less than 3 (or 5 if you like)
        to_remove = value_counts_2[value_counts_2 <= cutoff].index

        # # Keep rows where the city column is not in to_remove
        df2 = df[~df[columns[0]].isin(to_remove)]
        df3 = df2[~df2[columns[1]].isin(to_remove)]
        
        return df3
        
    else:
        print("Error: Too many columns to remove rare molecules from.")

   
def build_replacements():
    molecule_replacements = {}
     
    # Add a catalyst to the molecule_replacements dict (Done by Alexander)
    molecule_replacements['CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+3].[Rh+3]'] = 'CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+2].[Rh+2]'
    molecule_replacements['[CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+3]]'] = 'CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+2].[Rh+2]'
    molecule_replacements['[CC(C)(C)[P]([Pd][P](C(C)(C)C)(C(C)(C)C)C(C)(C)C)(C(C)(C)C)C(C)(C)C]'] = 'CC(C)(C)[PH]([Pd][PH](C(C)(C)C)(C(C)(C)C)C(C)(C)C)(C(C)(C)C)C(C)(C)C'
    molecule_replacements['CCCC[N+](CCCC)(CCCC)CCCC.CCCC[N+](CCCC)(CCCC)CCCC.CCCC[N+](CCCC)(CCCC)CCCC.[Br-].[Br-].[Br-]'] = 'CCCC[N+](CCCC)(CCCC)CCCC.[Br-]'
    molecule_replacements['[CCO.CCO.CCO.CCO.[Ti]]'] = 'CCO[Ti](OCC)(OCC)OCC'
    molecule_replacements['[CC[O-].CC[O-].CC[O-].CC[O-].[Ti+4]]'] = 'CCO[Ti](OCC)(OCC)OCC'
    molecule_replacements['[Cl[Ni]Cl.c1ccc(P(CCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1]'] = 'Cl[Ni]1(Cl)[P](c2ccccc2)(c2ccccc2)CCC[P]1(c1ccccc1)c1ccccc1'
    molecule_replacements['[Cl[Pd](Cl)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1]'] = 'Cl[Pd](Cl)([PH](c1ccccc1)(c1ccccc1)c1ccccc1)[PH](c1ccccc1)(c1ccccc1)c1ccccc1'
    molecule_replacements['[Cl[Pd+2](Cl)(Cl)Cl.[Na+].[Na+]]'] = 'Cl[Pd]Cl'
    molecule_replacements['Karstedt catalyst'] =   'C[Si](C)(C=C)O[Si](C)(C)C=C.[Pt]'
    molecule_replacements["Karstedt's catalyst"] = 'C[Si](C)(C=C)O[Si](C)(C)C=C.[Pt]'
    molecule_replacements['[O=C([O-])[O-].[Ag+2]]'] = 'O=C([O-])[O-].[Ag+].[Ag+]'
    molecule_replacements['[O=S(=O)([O-])[O-].[Ag+2]]'] = 'O=S(=O)([O-])[O-].[Ag+].[Ag+]'
    molecule_replacements['[O=[Ag-]]'] = 'O=[Ag]'
    molecule_replacements['[O=[Cu-]]'] = 'O=[Cu]'
    molecule_replacements['[Pd on-carbon]'] = '[C].[Pd]'
    molecule_replacements['[TEA]'] = 'OCCN(CCO)CCO'
    molecule_replacements['[Ti-superoxide]'] = 'O=[O-].[Ti]'
    molecule_replacements['[[Pd].c1ccc(P(c2ccccc2)c2ccccc2)cc1]'] = '[Pd].c1ccc(P(c2ccccc2)c2ccccc2)cc1.c1ccc(P(c2ccccc2)c2ccccc2)cc1.c1ccc(P(c2ccccc2)c2ccccc2)cc1.c1ccc(P(c2ccccc2)c2ccccc2)cc1'
    molecule_replacements['[c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd-4]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
    molecule_replacements['[c1ccc([P]([Pd][P](c2ccccc2)(c2ccccc2)c2ccccc2)(c2ccccc2)c2ccccc2)cc1]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
    molecule_replacements['[c1ccc([P](c2ccccc2)(c2ccccc2)[Pd]([P](c2ccccc2)(c2ccccc2)c2ccccc2)([P](c2ccccc2)(c2ccccc2)c2ccccc2)[P](c2ccccc2)(c2ccccc2)c2ccccc2)cc1]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
    molecule_replacements['[sulfated tin oxide]'] = 'O=S(O[Sn])(O[Sn])O[Sn]'
    molecule_replacements['[tereakis(triphenylphosphine)palladium(0)]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
    molecule_replacements['tetrakistriphenylphosphine palladium'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
    molecule_replacements['[zeolite]'] = 'O=[Al]O[Al]=O.O=[Si]=O'
    
    # Molecules found among the most common names in molecule_names
    molecule_replacements['TEA'] = 'OCCN(CCO)CCO'
    molecule_replacements['hexanes'] = 'CCCCCC'
    molecule_replacements['Hexanes'] = 'CCCCCC'
    molecule_replacements['hexanes ethyl acetate'] = 'CCCCCC.CCOC(=O)C'
    molecule_replacements['EtOAc hexanes'] = 'CCCCCC.CCOC(=O)C'
    molecule_replacements['EtOAc-hexanes'] = 'CCCCCC.CCOC(=O)C'
    molecule_replacements['ethyl acetate hexanes'] = 'CCCCCC.CCOC(=O)C'
    molecule_replacements['cuprous iodide'] = '[Cu]I'
    molecule_replacements['N,N-dimethylaminopyridine'] = 'n1ccc(N(C)C)cc1'
    molecule_replacements['dimethyl acetal'] = 'CN(C)C(OC)OC'
    molecule_replacements['cuprous chloride'] = 'Cl[Cu]'
    molecule_replacements["N,N'-carbonyldiimidazole"] = 'O=C(n1cncc1)n2ccnc2'
    # SiO2
    # Went down the list of molecule_names until frequency was 806
    
    # Canonicalise the molecules
    


    # Iterate over the dictionary and canonicalize each SMILES string
    for key, value in molecule_replacements.items():
        mol = Chem.MolFromSmiles(value)
        if mol is not None:
            molecule_replacements[key] = Chem.MolToSmiles(mol)
        
        
    return molecule_replacements
    

def main(clean_data_file_name = 'clean_data', consistent_yield = True, num_reactant=4, num_product=4, num_cat=1, num_solv=2, num_reag=2, rare_solv_0_cutoff=100, rare_solv_1_cutoff=50, rare_reag_0_cutoff=100, rare_reag_1_cutoff=50):
    
    # Merge all the pickled data into one big df
    df = merge_pickles()
    print('All data: ', len(df))
    
    # If something appears both as a reagent and a catalyst, remove it from the reagent column
    
    # Quite a few of the rows have Pd as a reagent. If the value in reagent_0 is already in catalyst_0, then replace the reagent value with np.NaN
    # df3["reagent_0"] = df3.apply(lambda x: np.nan if (pd.notna(x["reagent_0"]) and pd.notna(x["catalyst_0"]) and x["reagent_0"] in x["catalyst_0"]) else x["reagent_0"], axis=1)
    # df3["reagent_1"] = df3.apply(lambda x: np.nan if (pd.notna(x["reagent_1"]) and pd.notna(x["catalyst_0"]) and x["reagent_1"] in x["catalyst_0"]) else x["reagent_1"], axis=1)
    # Let's use this code instead, to keep track of the remvoed items:
    # removed_items = []

    # df["reagent_0"] = df.apply(lambda x: (removed_items.append(x["reagent_0"]) or np.nan) if (pd.notna(x["reagent_0"]) and pd.notna(x["catalyst_0"]) and x["reagent_0"] in x["catalyst_0"]) else x["reagent_0"], axis=1)
    # df["reagent_1"] = df.apply(lambda x: (removed_items.append(x["reagent_1"]) or np.nan) if (pd.notna(x["reagent_1"]) and pd.notna(x["catalyst_0"]) and x["reagent_1"] in x["catalyst_0"]) else x["reagent_1"], axis=1)
    
    # print('Reagents moved to catalyst column: ', list(set(removed_items)))
    
    
    
    
    # Remove any reactions where the reagent is Pd
    for i in range(num_reag):
        df = df[df[f"reagent_{i}"] != '[Pd]']
        df = df[df[f"reagent_{i}"] != '[Pd+2]']
        df = df[df[f"reagent_{i}"] != '[Pd+4]']
    df = df.reset_index(drop=True)
    
    print('After removing reactions with Pd as a reagent: ', len(df))
    
    # Remove reactions with too many components
    
    #reactant
    df = remove_reactions_with_too_many_of_component(df, 'reactant_', num_reactant)
    print('After removing reactions with too many reactants: ', len(df))
    
    #product
    df = remove_reactions_with_too_many_of_component(df, 'product_', num_product)
    df = remove_reactions_with_too_many_of_component(df, 'yield_', num_product)
    print('After removing reactions with too many products: ', len(df))
        
    #cat
    df = remove_reactions_with_too_many_of_component(df, 'catalyst_', num_cat)
    print('After removing reactions with too many catalysts: ', len(df))
    
    #solv
    df = remove_reactions_with_too_many_of_component(df, 'solvent_', num_solv)
    print('After removing reactions with too many solvents: ', len(df))
    
    #reag
    df = remove_reactions_with_too_many_of_component(df, 'reagent_', num_reag)
    print('After removing reactions with too many reagents: ', len(df))
    
    
    # Ensure consistent yield
    if consistent_yield:
        # Keep rows with yield <= 100 or missing yield values
        mask = pd.Series(data=True, index=df.index)  # start with all rows selected
        for i in range(num_product):
            yield_col = 'yield_'+str(i)
            yield_mask = (df[yield_col] >= 0) & (df[yield_col] <= 100) | pd.isna(df[yield_col])
            mask &= yield_mask

        df = df[mask]

        
        
        # sum of yields should be between 0 and 100
        yield_columns = df.filter(like='yield').columns

        # Compute the sum of the yield columns for each row
        df['total_yield'] = df[yield_columns].sum(axis=1)

        # Filter out reactions where the total_yield is less than or equal to 100, or is NaN or None
        mask = (df['total_yield'] <= 100) | pd.isna(df['total_yield']) | pd.isnull(df['total_yield'])
        df = df[mask]

        # Drop the 'total_yield' column from the DataFrame
        df = df.drop('total_yield', axis=1)
        print('After removing reactions with inconsistent yields: ', len(df))
        
    
    
    
    # Remove reactions with rare molecules
    # solv_0
    if rare_solv_0_cutoff != 0:
        df = remove_rare_molecules(df, ['solvent_0'], rare_solv_0_cutoff)
        print('After removing reactions with rare solvent_0: ', len(df))
    
    # solv_1
    if rare_solv_1_cutoff != 0:
        df = remove_rare_molecules(df, ['solvent_1'], rare_solv_1_cutoff)
        print('After removing reactions with rare solvent_1: ', len(df))
    
    # reag_0
    if rare_reag_0_cutoff != 0:
        df = remove_rare_molecules(df, ['reagent_0'], rare_reag_0_cutoff)
        print('After removing reactions with rare reagent_0: ', len(df))
    
    # reag_1
    if rare_reag_1_cutoff != 0:
        df = remove_rare_molecules(df, ['reagent_1'], rare_reag_1_cutoff)
        print('After removing reactions with rare reagent_1: ', len(df))
    
        
    # Hanlding of molecules with names instead of SMILES
    
    # Make replacements for molecules with names instead of SMILES
    # do the catalyst replacements that Alexander found, as well as other replacements
    molecule_replacements = build_replacements()
    df = df.replace(molecule_replacements) 
    
    ## Remove reactions that have a catalyst with a non-molecular name, e.g. 'Catalyst A'
    wrong_cat_names = ['Catalyst A', 'catalyst', 'catalyst 1', 'catalyst A', 'catalyst VI', 'reaction mixture', 'same catalyst', 'solution']
    molecule_names = pd.read_pickle('data/USPTO/molecule_names/molecule_names.pkl')
    
    molecules_to_remove = wrong_cat_names + molecule_names
    
    cols = []
    for col in list(df.columns):
        if 'reagent' in col or 'solvent' in col or 'catalyst' in col:
            cols += [col]
    
    for col in tqdm(cols):
        df = df[~df[col].isin(molecules_to_remove)]
    
    print('After removing reactions with nonsensical/unresolvable names: ', len(df))
    
    # Replace any instances of an empty string with None
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    
    
    
    
    
    # drop duplicates
    df = df.drop_duplicates()
    print('After removing duplicates: ', len(df))
    
    df.reset_index(inplace=True)
    
    # pickle the final cleaned dataset
    with open(f'data/USPTO/{clean_data_file_name}.pkl', 'wb') as f:
        pickle.dump(df, f)
    
    
 
    
    

if __name__ == "__main__":
    start_time = datetime.now()
    
    args = sys.argv[1:]
    # args is a list of the command line args
    # args: num_cat, num_solv, num_reag
    try:
        clean_data_file_name, consistent_yield, num_reactant, num_product, num_cat, num_solv, num_reag, rare_solv_0_cutoff, rare_solv_1_cutoff, rare_reag_0_cutoff, rare_reag_1_cutoff,  = args[0], args[1], int(args[2]), int(args[3]), int(args[4]), int(args[5]), int(args[6]), int(args[7]), int(args[8]), int(args[9]), int(args[10])
        
        assert consistent_yield in ['True', 'False']
        
        if consistent_yield == 'True': # NB: This will not remove nan yields!
            consistent_yield = True
        else:
            consistent_yield = False
            
        main(clean_data_file_name, consistent_yield, num_reactant, num_product, num_cat, num_solv, num_reag, rare_solv_0_cutoff, rare_solv_1_cutoff, rare_reag_0_cutoff, rare_reag_1_cutoff)
    except IndexError:
        print('Please enter the correct number of arguments')
        print('Usage: python USPTO_cleaning.py clean_data_file_name, num_reactant, num_product, num_cat, num_solv, num_reag, rare_solv_0_cutoff, rare_solv_1_cutoff, rare_reag_0_cutoff, rare_reag_1_cutoff, clean_data_file_name')
        print('Example: python USPTO_cleaning.py clean_test True 4 4 1 2 2 100 100 100 100')
        sys.exit(1)
    
        
    end_time = datetime.now()

    print('Duration: {}'.format(end_time - start_time))

