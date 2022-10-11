
"""
def get_uspto_data():
    return data

def apply_reaction_utils_cleaning(uspto_data):
    return cleaned_uspto_data

def get_uspto_smiles_in_standard_format(cleaned_uspto_data):
    return smiles_and_metadata 

    # returns: 
        list dictionaries

            reactants list discrete
            products list discrete
            pre-catalyst list discrete
            ligand list discrete
            catalyst list discrete
            solvent list discrete
            
            temperature union[float, range, mean + std]
            ph union[float, range, mean + std, temporal ph]
            raction_duration union[float, range, mean + std]
            cata_conc union[float, range, mean + std]
            base_conc union[float, range, mean + std]

            reaction english text (description)
            date_of_experiment
            experimentor
            experiment_lab


def clean_standard_format_data(sf_data, operations):

    # eg
    # dealing with pre-cata vs cata being reported differentlyby different experiments of the same reaction
    # double check ph only reported when weird, we can fill the expected phs if it is missing?
    # double check canonicalisation
    # check number of carbons / oxygens is +- 2 between reactant and product

    for op in operations:
        sf_data = op(sf_data)
    
    return sf_data



def add_reaction_fingerprint(sf_data, calc_reactfingerprint):
    for d in sf_data.values():
        d.setattr("reaction_fingerprint", calc_reactfingerprint(d))

    return sf_data

def add_reaction_template(sf_data, calc_reacttempalte):
    for d in sf_data.values():
        d.setattr("reaction_template", calc_reacttempalte(d))

    return sf_data


def add_reaction_vector_to_data(sf_data, calc_reactvector):
    for d in sf_data.values():
        d.setattr("reaction_vector", calc_reactvector(d))

    return sf_data


def get_morgan_fingerprint_calc(radius, etc):
    def morgan_fingerprint_calc(experiment):
        experiment = apply_logic(experiment, radius)
        return morgan_fingerprint

    return morgan_fingerprint_calc


def cluster_data(sf_data, cluster_method) -> dict[list[dict]]:
    '''
    this clustering can be performed using reaction vector, reaction fingerprint, reaction template or any other releveant data from the experiments above
    '''

    # if cluster_method is None:
    #     return {"unclustered": sf_data}
    # sf_data_clustered = cluster_method(sf_data)
    # return sf_data_clustered
    if cluster_method is None:
        return {"unclustered": sf_data}
    cluster1 = sf_data[:len(sf_data)//2]
    cluster2 = sf_data[len(sf_data)//2:]
    return {
        "cluster1name": cluster1,
        "cluster2name": cluster2
    }


uspto_data = get_uspto_data()
cleaned_uspto_data = apply_reaction_utils_cleaning(uspto_data)
for subset_of_cleaned_uspto_data in cleaned_uspto_data:
    sf_data = get_uspto_smiles_in_standard_format(subset_of_cleaned_uspto_data)
    sf_data = clean_standard_format_data(subset_sf_data, operations=cleaning_rules)
    sf_data = add_reaction_fingerprint(sf_data, calc_reactfingerprint=morgan_fingerprint)
    sf_data = add_reaction_template(sf_data, calc_reacttempalte)
    sf_data = add_reaction_vector_to_data(sf_data, calc_reactvector=set_as_react_fingerprint)
    sf_data.save()


sf_data_subsampled = load_subsample_from_files()

clustering_model = train_clustering_model(sf_data_subsampled, cluster_method=kmeans_over_reactvector)  # random clustering, temporal clustering, cluster by templates

models = train_model_over_clustered(sf_data, cluster_model, model_parameters, paramsharing_parameteres, filters_and_conditions) # train the model to predict based on the reaction_vector

"""