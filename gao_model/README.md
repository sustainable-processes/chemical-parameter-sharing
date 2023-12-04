# Condition prediction
The code in this folder comes mostly from the following source:
paper: https://pubs.acs.org/doi/full/10.1021/acscentsci.8b00357
repo: https://github.com/Coughy1991/Reaction_condition_recommendation

# Dependency management
1. To run the code in this folder, please create a new environment (python = ">=3.10,<3.12"), pip install poetry and then poetry install.

# Models
1. gao_model: model from the paper above
2. upstream_model: "upstream parameter sharing" in condition prediction we mean that the first layer of the network is specific to each reaction class, such that the network is able to create bespoke representations for each class.