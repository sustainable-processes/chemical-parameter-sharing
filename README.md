# chemical-parameter-sharing
A parameter sharing approach to reaction condition prediction using chemical reactions clustered by reaction class.

## Installation

First clone the repository using Git.

Then execute the following commands in the root of the repository (stolen from: https://github.com/MolecularAI/reaction_utils)

    conda env create -f env-dev.yml
    conda activate cps-env
    poetry install
    
 ## Sources of code:
 Reaction difference FP: https://pubs.acs.org/doi/abs/10.1021/ci5006614
 
 NN condition prediction (Gao model): https://pubs.acs.org/doi/full/10.1021/acscentsci.8b00357

## Code workflow

1) Data
    - ORDerly-condition: https://openreview.net/forum?id=R8FQMsECIS&noteId=y1HEbmW7QZ
    - Source: USPTO
    - Format: SMILES reaction string
2) Reaction clustering
    - NameRxn from Nextmove software
3) NN condition prediction conditioned on reaction class
    - Hard vs soft parameter sharing: https://avivnavon.github.io/blog/parameter-sharing-in-deep-learning/
    - We introduced conditional layers for condition prediction
    
    



    
    
