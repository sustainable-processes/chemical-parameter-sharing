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
3) NN condition prediction conditioned on reaction class
    - Hard vs soft parameter sharing: https://avivnavon.github.io/blog/parameter-sharing-in-deep-learning/
    
# Timeline

- By end of Nov: Complete 1 full path through the pipeline, choosing the easiest options for cleaning data, clustering, and NN. Plot our first parity plot
- December: Perform case study on rxn with only little data (e.g. cyclopropanation? Alexander should suggest a reaction). Consider 3 cases, and if outperformance is shown, work on conference draft:
    - Feed all data to NN
    - Feed only data about said reaction to NN (LOOCV)
    - Parameter sharing
- Jan-Feb: Add features to pipeline, e.g. different data sources (Reaxys & pistachio), different methods of parameter sharing & clustering
- March-April: Draft paper
    
    
