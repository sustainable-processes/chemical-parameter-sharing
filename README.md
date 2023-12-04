# chemical-parameter-sharing
A parameter sharing approach to reaction condition prediction using the implications of named reactions

## Installation

First clone the repository using Git.

Then execute the following commands in the root of the repository (stolen from: https://github.com/MolecularAI/reaction_utils)

    conda env create -f env-dev.yml
    conda activate cps-env
    poetry install
    
 ## Sources of code:
 Reaction difference FP: https://pubs.acs.org/doi/abs/10.1021/ci5006614
 
 NN condition prediction: https://pubs.acs.org/doi/full/10.1021/acscentsci.8b00357

## Schneider

- Build-class-and-superclass-predictive-models
- Create-transformation-FPs-sets
- utilsFunctions
- createFingerprintsReaction

## Code workflow

1) Data handling
    - Data: USPTO
    - Format: SMILES reaction string
    - Cleaning: https://molecularai.github.io/reaction_utils/uspto.html
    - Cleaning: https://github.com/rxn4chemistry/rxn-reaction-preprocessing
    - Smarts viewer: https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c00992 (https://smarts.plus)
2) Reaction clustering
    - Clustering rxn FP (schneider)
    - Meta learning? https://lilianweng.github.io/posts/2018-11-30-meta-learning/
    - Rxn templates matching
    - Novel rxn fp https://www.nature.com/articles/s42256-020-00284-w
3) NN condition prediction conditioned on reaction class
    - Hard vs soft parameter sharing: https://avivnavon.github.io/blog/parameter-sharing-in-deep-learning/
    - Condition prediction: https://github.com/Coughy1991/Reaction_condition_recommendation/tree/64f151e302abcb87e0a14088e08e11b4cea8d5ab
    
    
# Data
1) USPTO from ORD using ORDerly
2) Pistachio
3) Suzuki reactions from Reaxys
4) https://github.com/jshmjs45/data_for_chem#data_from_cjhif_utf8


    
    
