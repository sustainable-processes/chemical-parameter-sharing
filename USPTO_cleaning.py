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
## Bundle all the solvents/reagents/catalysts together
## Apply a map to extract the solvents, given a list of solvents from Summit
## Place metals as the first reagent to give the model the chance to predcit solvents

# https://github.com/sustainable-processes/vle_prediction/blob/master/data/cosmo/solvent_descriptors.csv


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

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True)
        
def build_solvents_list_and_dict():
    solvents = pd.read_csv('data/USPTO/solvents.csv', index_col=0)
    solvents.loc[375, 'smiles'] = 'ClP(Cl)Cl'
    solvents.loc[405, 'smiles'] = 'ClS(Cl)=O'
    
    solvents['canonical_smiles'] = solvents['smiles'].apply(canonicalize_smiles)
    
    solvents_list = list(solvents['canonical_smiles'])
    solvents_list += 'CO'
    
    
    # Combine the lists into a sequence of key-value pairs
    key_value_pairs = zip(list(solvents['stenutz_name']) + list(solvents['cosmo_name']), list(solvents['canonical_smiles']) + list(solvents['canonical_smiles']))

    # Create a dictionary from the sequence
    solvents_dict = dict(key_value_pairs)

    solvents_dict['methanol'] = 'CO'
    
    return solvents_list, solvents_dict

   
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
    
    # Remove reactions with too many reactants or products
    
    #reactant
    df = remove_reactions_with_too_many_of_component(df, 'reactant_', num_reactant)
    print('After removing reactions with too many reactants: ', len(df))
    
    #product
    df = remove_reactions_with_too_many_of_component(df, 'product_', num_product)
    df = remove_reactions_with_too_many_of_component(df, 'yield_', num_product)
    print('After removing reactions with too many products: ', len(df))
    
    # Map to canonical names using the dicts we created
    
    # Hanlding of molecules with names instead of SMILES
    
    # Make replacements for molecules with names instead of SMILES
    # do the catalyst replacements that Alexander found, as well as other replacements
    molecule_replacements = build_replacements()
    df = df.replace(molecule_replacements) 
    
    # Do solvents replacements (in case there are any smiles represented with names)
    
    solvents_list, solvents_dict = build_solvents_list_and_dict()
    df = df.replace(solvents_dict) # This line was taking too long. Prbably good idea to restrict the number of columns this is run on, e.g. only on the agent columns, and we should remove the unnecessary columns before this step (ie if num solv+cat+reag is 10, we should remove all reactions with more than 10 agents).
    
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
    
    
    # Find solvents from the solvents_list that are in any of the solv/reag/cat columns, and put them in their own column
    
    
    
    
    
    
        
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

