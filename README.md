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
2) Reaction clustering
    - Clustering rxn FP (schneider)
    - Meta learning? https://lilianweng.github.io/posts/2018-11-30-meta-learning/
3) NN condition prediction conditioned on reaction class
