#!/usr/bin/env python
# coding: utf-8

# In[1]:


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import _pickle as cPickle
import gzip
from collections import defaultdict
import random
import createFingerprintsReaction


# In[2]:


dataDir = "../data/"


# Create different transformation FPs (AP3, MG2 and TT) as SparseIntVect

# In[3]:


infile = gzip.open(dataDir+'training_test_set_patent_data.pkl.gz', 'rb')
pklfile = gzip.open(dataDir+'transformationFPs_test_set_patent_data.pkl.gz','wb+')

# %%

lineNo=0
while 1:
    lineNo+=1
    try:
        smi,lbl,klass = cPickle.load(infile) 
    except EOFError:
        break
    try:
        rxn = AllChem.ReactionFromSmarts(smi,useSmiles=True)
        fp_AP3 = createFingerprintsReaction.create_transformation_FP(rxn,AllChem.FingerprintType.AtomPairFP)
        fp_MG2 = createFingerprintsReaction.create_transformation_FP(rxn,AllChem.FingerprintType.MorganFP)
        fp_TT = createFingerprintsReaction.create_transformation_FP(rxn,AllChem.FingerprintType.TopologicalTorsion)
    except:
        print("Cannot build fingerprint/reaction of: %s\n"%smi)
        continue
    cPickle.dump((lbl,klass,fp_AP3, fp_MG2, fp_TT),pklfile,2)
    if not lineNo%5000:
        print("Done: %d"%lineNo)


# Combine AP3 fingerprint with agent feature und Morgan2 FPs

# In[4]:


infile = gzip.open(dataDir+'training_test_set_patent_data.pkl.gz', 'rb')
pklfile = gzip.open(dataDir+'transformationFPs_MG2_agentFPs_test_set_patent_data.pkl.gz','wb+')

lineNo=0
while 1:
    lineNo+=1
    try:
        smi,lbl,klass = cPickle.load(infile) 
    except EOFError:
        break
    try:
        rxn = AllChem.ReactionFromSmarts(smi,useSmiles=True)
        fp_AP3 = createFingerprintsReaction.create_transformation_FP(rxn,AllChem.FingerprintType.AtomPairFP)
        fp_MG2_agents = createFingerprintsReaction.create_agent_morgan2_FP(rxn)
        if fp_MG2_agents is None:
            fp_MG2_agents = DataStructs.UIntSparseIntVect(4096)
        fp_featureAgent = createFingerprintsReaction.create_agent_feature_FP(rxn)
    except:
        print("Cannot build fingerprint/reaction of: %s\n"%smi)
        continue
    cPickle.dump((lbl,klass,fp_AP3,fp_featureAgent,fp_MG2_agents),pklfile,2)
    if not lineNo%5000:
        print("Done: %d"%lineNo)


# Create transformation FP (AP3 + agent featureFP) for external test set A

# In[5]:


infile = gzip.open(dataDir+'training_test_set_patent_data.pkl.gz', 'rb')
pklfile = gzip.open(dataDir+'transformationFPs_agentFPs_external_test_set_A.pkl.gz','wb+')

lineNo=0
while 1:
    lineNo+=1
    try:
        smi,lbl,klass = cPickle.load(infile) 
    except EOFError:
        break
    try:
        rxn = AllChem.ReactionFromSmarts(smi,useSmiles=True)
        fp_AP3 = createFingerprintsReaction.create_transformation_FP(rxn,AllChem.FingerprintType.AtomPairFP)
        fp_featureAgent = createFingerprintsReaction.create_agent_feature_FP(rxn)
    except:
        print("Cannot build fingerprint/reaction of: %s\n"%smi)
        continue
    cPickle.dump((lbl,klass,fp_AP3,fp_featureAgent),pklfile,2)
    if not lineNo%5000:
        print("Done: %d"%lineNo)


# Create transformation FP (AP3 + agent featureFP) for external test set B (unclassified reactions)

# In[6]:


infile = gzip.open(dataDir+'unclassified_reactions_patent_data.pkl.gz', 'rb')
pklfile = gzip.open(dataDir+'transformationFPs_agentFPs_external_test_set_B.pkl.gz','wb+')

lineNo=0
while 1:
    lineNo+=1
    try:
        smi,lbl,klass = cPickle.load(infile) 
    except EOFError:
        break
    try:
        rxn = AllChem.ReactionFromSmarts(smi,useSmiles=True)
        fp_AP3 = createFingerprintsReaction.create_transformation_FP(rxn,AllChem.FingerprintType.AtomPairFP)
        fp_featureAgent = createFingerprintsReaction.create_agent_feature_FP(rxn)
    except:
        print("Cannot build fingerprint/reaction of: %s\n"%smi)
        continue
    cPickle.dump((lbl,smi,fp_AP3,fp_featureAgent),pklfile,2)
    if not lineNo%5000:
        print("Done: %d"%lineNo)


# %%
