# Clean the USPTO data

"""
# Usage:
python USPTO_cleaning.py 
NB: Requires all the pickle files created by USPTO_extraction.py

# Output:
1) A pickle file with the cleaned data in a pickle file

# Functionality:
1) 

"""

# Imports


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


def main(clean_data_file_name = 'clean_data', num_reactant=4, num_product=4, num_cat=1, num_solv=2, num_reag=2, rare_solv_0_cutoff=100, rare_solv_1_cutoff=50, rare_reag_0_cutoff=100, rare_reag_1_cutoff=50):
    
    # Merge all the pickled data into one big df
    df = merge_pickles()
    print('All data: ', len(df))
    
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
        for i in range(num_product):
            df = df[df['yield_'+str(i)] <= 100]
            df = df[df['yield_'+str(i)] >= 0]
            #df = df[df['yield_'+str(i)] <= df['yield_'+str(i)]]
        # All yields should be between 0 and 100
        df = df[df['yield_0'] >= 0]
        
        
        # sum of yields should be between 0 and 100
        yield_columns = df.filter(like='yield').columns

        # Compute the sum of the yield columns for each row
        df['total_yield'] = df[yield_columns].sum(axis=1)

        # Filter the rows where the total yield is less than or equal to 100
        df = df[df['total_yield'] <= 100]

        # Drop the 'total_yield' column from the DataFrame
        df = df.drop('total_yield', axis=1)
        print('After removing reactions with inconsistent yields: ', len(df))
        
        
    
    
    # Hanlding of molecules with names instead of SMILES
    
    
    
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
    
    # pickle the final cleaned dataset
    with open(f'data/USPTO/{clean_data_file_name}.pkl', 'wb') as f:
        pickle.dump(df, f)
    
    
    
    
    
    
    

if __name__ == "__main__":
    args = sys.argv[1:]
    # args is a list of the command line args
    # args: num_cat, num_solv, num_reag
    try:
        clean_data_file_name, num_reactant, num_product, num_cat, num_solv, num_reag, rare_solv_0_cutoff, rare_solv_1_cutoff, rare_reag_0_cutoff, rare_reag_1_cutoff, consistent_yield = args[0], int(args[1]), int(args[2]), int(args[3]), int(args[4]), int(args[5]), int(args[6]), int(args[7]), int(args[8]), int(args[9]), args[10]
        
        assert consistent_yield in ['True', 'False']
        
        if consistent_yield == 'True': # NB: This will not remove nan yields!
            consistent_yield = True
        else:
            consistent_yield = False
            
        main(clean_data_file_name, num_reactant, num_product, num_cat, num_solv, num_reag, rare_solv_0_cutoff, rare_solv_1_cutoff, rare_reag_0_cutoff, rare_reag_1_cutoff)
    except IndexError:
        print('Please enter the correct number of arguments')
        print('Usage: python USPTO_cleaning.py clean_data_file_name, num_reactant, num_product, num_cat, num_solv, num_reag, rare_solv_0_cutoff, rare_solv_1_cutoff, rare_reag_0_cutoff, rare_reag_1_cutoff')
        print('Example: python USPTO_cleaning.py 4 4 1 2 2 100 50 100 50')
        sys.exit(1)
    
    

