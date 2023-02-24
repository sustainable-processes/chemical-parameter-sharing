"""
1) Download the USPTO data from https://github.com/open-reaction-database/ord-data and put it in a folder called "data/USPTO/"
1.1) You need to git clone the repo above, and you'll find the data in ord-data/data/
1.2) It is batched by year, it's best to just maintain this batching, it will make it easier to handle (each file won't get excessively large)
2) python USPTO_extraction.py

# Output:
1) A pickle file with the cleaned data for each folder of uspto data. NB: Temp always in C, time always in hours
"""

# Import modules
#import ord_schema
from ord_schema import message_helpers#, validations
from ord_schema.proto import dataset_pb2


import pandas as pd
import numpy as np
import os

import pickle
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime

from rdkit import Chem

from tqdm import tqdm
import os



"""
# Disables RDKit whiny logging.
# """
# import rdkit.rdBase as rkrb
# import rdkit.RDLogger as rkl

# logger = rkl.logger()
# logger.setLevel(rkl.ERROR)
# rkrb.DisableLog('rdApp.error')
from rdkit.rdBase import BlockLogs



class OrdToPickle():
    """
    Read in an ord file, check if it contains USPTO data, and then:
    1) Extract all the relevant data (raw): reactants, products, catalysts, reagents, yields, temp, time
    2) Canonicalise all the molecules
    3) Write to a pickle file
    """

    def __init__(self, ord_file_path):
        self.ord_file_path = ord_file_path
        self.data = message_helpers.load_message(self.ord_file_path, dataset_pb2.Dataset)
        self.filename = self.data.name
        self.names_list = []

    def find_smiles(self, identifiers):
        block = BlockLogs()
        for i in identifiers:
            if i.type == 2:
                smiles = self.clean_smiles(i.value)
                return smiles
        for ii in identifiers: #if there's no smiles, return the name
            if ii.type == 6:
                name = ii.value
                self.names_list += [name]
                return name
        return None

    def clean_mapped_smiles(self, smiles):
        block = BlockLogs()
        # remove mapping info and canonicalsie the smiles at the same time
        # converting to mol and back canonicalises the smiles string
        try:
            m = Chem.MolFromSmiles(smiles)
            for atom in m.GetAtoms():
                atom.SetAtomMapNum(0)
            cleaned_smiles = Chem.MolToSmiles(m)
            return cleaned_smiles
        except AttributeError:
            self.names_list += [smiles]
            return smiles

    def clean_smiles(self, smiles):
        block = BlockLogs()
        # remove mapping info and canonicalsie the smiles at the same time
        # converting to mol and back canonicalises the smiles string
        try:
            cleaned_smiles = Chem.CanonSmiles(smiles)
            return cleaned_smiles
        except:
            self.names_list += [smiles]
            return smiles

    #its probably a lot faster to sanitise the whole thing at the end
    # NB: And create a hash map/dict



    def build_rxn_lists(self):
        mapped_rxn_all = []
        reactants_all = []
        reagents_all = []
        products_all = []
        solvents_all = []
        catalysts_all = []

        temperature_all = []

        rxn_times_all = []

        yields_all = []

        for i in range(len(self.data.reactions)):
            rxn = self.data.reactions[i]
            # handle rxn inputs: reactants, reagents etc
            reactants = []
            reagents = []
            solvents = []
            catalysts = []
            marked_products = []
            mapped_products = []
            products = []
            not_mapped_products = []
            
            temperatures = []

            rxn_times = []

            yields = []
            mapped_yields = []
            

            #if reaction has been mapped, get reactant and product from the mapped reaction
            #Actually, we should only extract data from reactions that have been mapped
            is_mapped = self.data.reactions[i].identifiers[0].is_mapped
            if is_mapped:
                mapped_rxn_extended_smiles = self.data.reactions[i].identifiers[0].value
                mapped_rxn = mapped_rxn_extended_smiles.split(' ')[0]

                reactant, reagent, mapped_product = mapped_rxn.split('>')

                for r in reactant.split('.'):
                    if '[' in r and ']' in r and ':' in r:
                        reactants += [r]
                    else:
                        reagents += [r]

                reagents += [r for r in reagent.split('.')]

                for p in mapped_product.split('.'):
                    if '[' in p and ']' in p and ':' in p:
                        mapped_products += [p]
                        
                    else:
                        not_mapped_products += [p]


                # inputs
                for key in rxn.inputs: #these are the keys in the 'dict' style data struct
                    try:
                        components = rxn.inputs[key].components
                        for component in components:
                            rxn_role = component.reaction_role #rxn role
                            identifiers = component.identifiers
                            smiles = self.find_smiles(identifiers)
                            if rxn_role == 1: #reactant
                                #reactants += [smiles]
                                # we already added reactants from mapped rxn
                                # So instead I'll add it to the reagents list
                                # A lot of the reagents seem to have been misclassified as reactants
                                # I just need to remember to remove reagents that already appear as reactants
                                #   when I do cleaning

                                reagents += [r for r in smiles.split('.')]
                            elif rxn_role ==2: #reagent
                                reagents += [r for r in smiles.split('.')]
                            elif rxn_role ==3: #solvent
                                solvents += [smiles]
                            elif rxn_role ==4: #catalyst
                                catalysts += [smiles]
                            elif rxn_role in [5,6,7]: #workup, internal standard, authentic standard. don't care about these
                                continue
                            # elif rxn_role ==8: #product
                            #     #products += [smiles]
                            # there are no products recorded in rxn_role == 8, they're all stored in "outcomes"
                    except IndexError:
                        #print(i, key )
                        continue

                # temperature
                try:
                    # first look for the temperature as a number
                    temp_unit = rxn.conditions.temperature.setpoint.units
                        
                    if temp_unit == 1: #celcius
                        temperatures +=[rxn.conditions.temperature.setpoint.units]
                        
                    elif temp_unit == 2: #fahrenheit
                        f = rxn.conditions.temperature.setpoint.units
                        c = (f-32)*5/9
                        temperatures +=[c]
                        
                    elif temp_unit == 3: #kelvin
                        k = rxn.conditions.temperature.setpoint.units
                        c = k - 273.15
                        temperatures +=[c]
                    elif temp_unit == 0:
                        if temp_unit == 0: #unspecified
                            #instead of using the setpoint, use the control type
                            #temperatures are in celcius
                            temp_control_type = rxn.conditions.temperature.control.type
                            if temp_control_type == 2: #AMBIENT
                                temperatures +=[25]
                            elif temp_control_type == 6: #ICE_BATH
                                temperatures +=[0]
                            elif temp_control_type == 9: #DRY_ICE_BATH
                                temperatures +=[-79]
                            elif temp_control_type == 11: #LIQUID_NITROGEN
                                temperatures +=[-196]   
                except IndexError:
                    continue
                    

                #rxn time
                try:
                    if rxn.outcomes[0].reaction_time.units == 1: #hour
                        rxn_times += [rxn.outcomes[0].reaction_time.value]
                    elif rxn.outcomes[0].reaction_time.units == 3: #seconds
                        s = rxn.outcomes[0].reaction_time.value
                        h = s/3600
                        rxn_times += [h]
                    elif rxn.outcomes[0].reaction_time.units == 2: #minutes
                        m =  rxn.outcomes[0].reaction_time.value
                        h = m/60
                        rxn_times += [h]
                    elif rxn.outcomes[0].reaction_time.units == 4: #day
                        d = rxn.outcomes[0].reaction_time.value
                        h = d*24
                        rxn_times += [h]
                except IndexError:
                    continue

                # products & yield
                products_obj = rxn.outcomes[0].products
                y1 = np.nan
                y2 = np.nan
                for marked_product in products_obj:
                    try:
                        identifiers = marked_product.identifiers
                        product_smiles = self.find_smiles(identifiers)
                        measurements = marked_product.measurements
                        for measurement in measurements:
                            if measurement.details =="PERCENTYIELD":
                                y1 = measurement.percentage.value
                            elif measurement.details =="CALCULATEDPERCENTYIELD":
                                y2 = measurement.percentage.value
                        #marked_products += [(product_smiles, y1, y2)]
                        marked_products += [product_smiles]
                        if y1 == y1:
                            yields += [y1]
                        elif y2==y2:
                            yields +=[y2]
                        else:
                            yields += [np.nan]
                    except IndexError:
                        continue
            
            #clean the smiles

            #remove reagents that are integers
            #reagents = [x for x in reagents if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
            # I'm assuming there are no negative integers
            reagents = [x for x in reagents if not (x.isdigit())]

            reactants = [self.clean_mapped_smiles(smi) for smi in reactants]
            reagents = [self.clean_smiles(smi) for smi in reagents]
            solvents = [self.clean_smiles(smi) for smi in solvents]
            catalysts = [self.clean_smiles(smi) for smi in catalysts]

            # if the reagent exists in another list, remove it
            reagents_trimmed = []
            for reag in reagents:
                if reag not in reactants and reag not in solvents and reag not in catalysts:
                    reagents_trimmed += [reag]
            

            mapped_rxn_all += [mapped_rxn]
            reactants_all += [reactants]
            reagents_all += [list(set(reagents_trimmed))]
            solvents_all += [list(set(solvents))]
            catalysts_all += [list(set(catalysts))]
            
            temperature_all = [temperatures]

            rxn_times_all += [rxn_times]


            # products logic
            # handle the products
            # for each row, I will trust the mapped product more
            # loop over the mapped products, and if the mapped product exists in the marked product
            # add the yields, else simply add smiles and np.nan

            # canon and remove mapped info from products
            mapped_p_clean = [self.clean_mapped_smiles(p) for p in mapped_products]
            marked_p_clean = [self.clean_smiles(p) for p in marked_products]
            # What if there's a marked product that only has the correct name, but not the smiles?



            for mapped_p in mapped_p_clean:
                added = False
                for ii, marked_p in enumerate(marked_p_clean):
                    if mapped_p == marked_p and mapped_p not in products:
                        products+= [mapped_p]
                        mapped_yields += [yields[ii]]
                        added = True
                        break

                if not added and mapped_p not in products:
                    products+= [mapped_p]
                    mapped_yields += [np.nan]
            

            products_all += [products] 
            yields_all +=[mapped_yields]


        
        return mapped_rxn_all, reactants_all, reagents_all, solvents_all, catalysts_all, temperature_all, rxn_times_all, products_all, yields_all

    # create the column headers for the df
    def create_column_headers(self, df, base_string):
        column_headers = []
        for i in range(len(df.columns)):
            column_headers += [base_string+str(i)]
        return column_headers
    
    def build_full_df(self):
        headers = ['mapped_rxn_', 'reactant_', 'reagent_',  'solvent_', 'catalyst_', 'temperature_', 'rxn_time_', 'product_', 'yield_']
        #data_lists = [mapped_rxn, reactants_all, reagents_all, solvents_all, catalysts_all, temperature_all, rxn_times_all, products_all]
        data_lists = self.build_rxn_lists()
        for i in range(len(headers)):
            new_df = pd.DataFrame(data_lists[i])
            df_headers = self.create_column_headers(new_df, headers[i])
            new_df.columns = df_headers
            if i ==0:
                full_df = new_df
            else:
                full_df = pd.concat([full_df, new_df], axis=1)
        return full_df
    
    #def clean_df(self,df):
        # In the test case: data/ORD_USPTO/ord-data/data/59/ord_dataset-59f453c3a3d34a89bfd97b6b8b151908.pb.gz
        #   there is only 1 reaction with 9 catalysts, all other reactions have like 1 or 2
        #   Perhaps I should here add some hueristic that there's more than x unique catalysts, just filter out
        #   the whole reaction
        

    def main(self):
        # This function doesn't return anything. Instead, it saves the requested data as a pickle file at the path you see below
        # So you need to unpickle the data to see the output
        if 'uspto' in self.filename:
            full_df = self.build_full_df()
            #cleaned_df = self.clean_df(full_df)
            

            #save data to pickle
            filename = self.data.name
            full_df.to_pickle(f"data/USPTO/pickled_data/{filename}.pkl")
            
            #save names to pickle
            #list of the names used for molecules, as opposed to SMILES strings
            #save the names_list to pickle file
            with open(f"data/USPTO/molecule_names/molecules_{filename}.pkl", 'wb') as f:
                pickle.dump(self.names_list, f)
            
        #else:
            #print(f'The following does not contain USPTO data: {self.filename}')
            
        
def get_file_names():
    # Set the directory you want to look in
    directory = "data/USPTO/ord-data/data/"

    # Use listdir to get a list of all files in the directory
    folders = os.listdir(directory)
    files = []
    # Use a for loop to iterate over the files and print their names
    for folder in folders:
        if not folder.startswith("."):
            new_dir = directory+folder
            file_list = os.listdir(new_dir)
            # Check if the file name starts with a .
            for file in file_list:
                if not file.startswith("."):
                    new_file = new_dir+'/'+file
                    files += [new_file]
    return files

def merge_pickled_mol_names():
    #if the file already exists, delete it
    output_file_path = "data/USPTO/molecule_names/molecule_names.pkl"
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    #create one big list of all the pickled names
    folder_path = 'data/USPTO/molecule_names/'
    onlyfiles = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    full_lst = []
    for file in tqdm(onlyfiles):
        if file[0] != '.': #We don't want to try to unpickle .DS_Store
            filepath = folder_path+file 
            unpickled_lst = pd.read_pickle(filepath)
            full_lst = full_lst + unpickled_lst
            
    unique_molecule_names = list(set(full_lst))
    
    #pickle the list
    with open(output_file_path, 'wb') as f:
        pickle.dump(unique_molecule_names, f)
    


def main(file):
    instance = OrdToPickle(file)
    instance.main()
    
    

if __name__ == "__main__":
    
    start_time = datetime.now()
    
    pickled_data_path = 'data/USPTO/pickled_data'
    molecule_name_path = 'data/USPTO/molecule_names'

    if not os.path.exists(pickled_data_path):
        os.makedirs(pickled_data_path)
    if not os.path.exists(molecule_name_path):
        os.makedirs(molecule_name_path)
    
    files = get_file_names()
    
    num_cores = multiprocessing.cpu_count()
    inputs = tqdm(files)
    Parallel(n_jobs=num_cores)(delayed(main)(i) for i in inputs)
    

    # Create a list of all the unique molecule names
    merge_pickled_mol_names()
    
    
    end_time = datetime.now()

    print('Duration: {}'.format(end_time - start_time))

