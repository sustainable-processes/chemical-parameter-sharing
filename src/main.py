
import numpy as np

def pipeline(file_loc, get_data, get_smiles, get_reaction_fingerprints, get_reaction_representation, cluster_reactions, model_reaction_conditions, param_sharing_method, modelling_method):

    data = get_data(file_loc)
    
    reaction_smiles = get_smiles(data)
    assert isinstance(data[0], tuple)
    assert len(data[0]) == 3
    assert isinstance(data[0][0], tuple) # tuple of reactants
    assert isinstance(data[0][0][0], str) # reactants as str form 
    assert isinstance(data[0][1], tuple) # tuple of products
    assert isinstance(data[0][1][0], str) # reactants as str form 
    assert isinstance(data[0][2], dict)
    # tuple of reactants, products and reaction conditions

    fingerprints = get_reaction_fingerprints(reaction_smiles)
    assert isinstance(fingerprints[0], tuple)
    assert len(fingerprints[0]) == 3
    assert isinstance(data[0][0], tuple) # tuple of reactants
    assert isinstance(data[0][0][0], np.ndarray) # reactants as vector form 
    assert isinstance(data[0][1], tuple) # tuple of products
    assert isinstance(data[0][1][0], np.ndarray) # products as vector form  
    assert isinstance(data[0][2], dict)
    # tuple of reactants, products and reaction conditions

    reaction_representations = get_reaction_representation(fingerprints)
    assert isinstance(reaction_representations[0], tuple)
    assert len(reaction_representations[0]) == 2
    assert isinstance(reaction_representations[0][0], np.ndarray)
    assert isinstance(reaction_representations[0][1], dict)
    # list of x,y where x is reaction representaiton, y is reaction conditions

    reaction_representations_clustered = cluster_reactions(reaction_representations)
    assert isinstance(reaction_representations_clustered, dict)
    assert isinstance(reaction_representations_clustered[0][0], tuple)
    assert len(reaction_representations_clustered[0][0]) == 2
    assert isinstance(reaction_representations_clustered[0][0][0], np.ndarray)
    assert isinstance(reaction_representations_clustered[0][0][1], dict)
    # dictionary of clusters where each cluster has a list of reaction representations

    model = model_reaction_conditions(
        reaction_representations_clustered, param_sharing_method, modelling_method,
    )
    # model digests reaction_representations clustered and has a forward function
    return model



if __name__ == "__main__":

    pass

