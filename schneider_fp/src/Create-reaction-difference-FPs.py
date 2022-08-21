#!/usr/bin/env python
# coding: utf-8

# In[1]:


from rdkit import Chem
from rdkit.Chem import AllChem
import _pickle as cPickle
import gzip
from collections import defaultdict
import random
import createFingerprintsReaction
from rdkit import DataStructs 


# In[4]:


dataDir = "../data/"


# Create different reaction difference FPs (2048 bit, AP FP (max path length = 30))

# In[9]:


infile = gzip.open(dataDir+'training_test_set_patent_data.pkl.gz', 'rb')
pklfile = gzip.open(dataDir+'reaction_FPs_training_test_set_patent_data.pkl.gz','wb+')
lineNo=0
while 1:
    lineNo+=1
    try:
        smi,lbl,klass = cPickle.load(infile) 
    except EOFError:
        break
    try:
        rxn = AllChem.ReactionFromSmarts(smi,useSmiles=True)
        rxn.RemoveUnmappedReactantTemplates()
        params = AllChem.ReactionFingerprintParams()
        # ignore agents 
        fp_woA = AllChem.CreateDifferenceFingerprintForReaction(rxn, params)
        # use default agent settings
        params.includeAgents=True
        params.noAgentWeight=10
        params.agentWeight=1
        fp_wA1 = AllChem.CreateDifferenceFingerprintForReaction(rxn, params)
        # use agents as reactants
        params.includeAgents=True
        params.noAgentWeight=10
        params.agentWeight=-1
        fp_wA2 = AllChem.CreateDifferenceFingerprintForReaction(rxn, params)
        # use equal weighting for reactants, products and agents
        params.includeAgents=True
        params.noAgentWeight=1
        params.agentWeight=1
        fp_wA3 = AllChem.CreateDifferenceFingerprintForReaction(rxn, params)
    except:
        print("Cannot build fingerprint/reaction of: %s\n"%smi)
        continue;
    cPickle.dump((lbl,klass,fp_woA,fp_wA1,fp_wA2,fp_wA3),pklfile,2)
    if not lineNo%5000:
        print("Done: %d"%lineNo)
infile.close()
pklfile.close()


# Create larger reaction difference FPs 4096bit

# In[10]:


infile = gzip.open(dataDir+'training_test_set_patent_data.pkl.gz', 'rb')
pklfile = gzip.open(dataDir+'reaction_FPs_4096bit_training_test_set_patent_data.pkl.gz','wb+')
lineNo=0
while 1:
    lineNo+=1
    try:
        smi,lbl,klass = cPickle.load(infile)        
    except EOFError:
        break
    try:
        rxn = AllChem.ReactionFromSmarts(smi,useSmiles=True)
        #rxn = AllChem.ReactionFromSmarts(smi)
        rxn.RemoveUnmappedReactantTemplates()
        params = AllChem.ReactionFingerprintParams()
        params.fpSize = 4096
        # ignore agents 
        fp_woA = AllChem.CreateDifferenceFingerprintForReaction(rxn, params)
        # use default agent settings
        params.includeAgents=True
        params.noAgentWeight=10
        params.agentWeight=1
        fp_wA1 = AllChem.CreateDifferenceFingerprintForReaction(rxn, params)
        # use agents as reactants
        params.includeAgents=True
        params.noAgentWeight=10
        params.agentWeight=-1
        fp_wA2 = AllChem.CreateDifferenceFingerprintForReaction(rxn, params)
        # use equal weighting for reactants, products and agents
        params.includeAgents=True
        params.noAgentWeight=1
        params.agentWeight=1
        fp_wA3 = AllChem.CreateDifferenceFingerprintForReaction(rxn, params)
    except:
        print("Cannot build fingerprint/reaction of: %s\n"%smi)
        continue;
    cPickle.dump((lbl,klass,fp_woA,fp_wA1,fp_wA2,fp_wA3),pklfile,2)
    if not lineNo%5000:
        print("Done: %d"%lineNo)
infile.close()
pklfile.close()


# Combine difference reaction FP (2048 bit, AP30) with agent feature FPs

# In[11]:


infile = gzip.open(dataDir+'training_test_set_patent_data.pkl.gz', 'rb')
pklfile = gzip.open(dataDir+'reaction_FPs_agentFeatureFPs_training_test_set_patent_data.pkl.gz','wb+')
lineNo=0
while 1:
    lineNo+=1
    try:
        smi,lbl,klass = cPickle.load(infile)        
    except EOFError:
        break
    try:
        rxn = AllChem.ReactionFromSmarts(smi,useSmiles=True)       
        rxn.RemoveUnmappedReactantTemplates()
        params = AllChem.ReactionFingerprintParams()
        # ignore agents 
        fp_woA = AllChem.CreateDifferenceFingerprintForReaction(rxn, params)
        fp_featureAgent = createFingerprintsReaction.create_agent_feature_FP(rxn)
    except:
        print("Cannot build fingerprint/reaction of: %s\n"%smi)
        continue;
    cPickle.dump((lbl,klass,fp_woA,fp_featureAgent),pklfile,2)
    if not lineNo%5000:
        print("Done: %d"%lineNo)
infile.close()
pklfile.close()


# Combine difference reaction FP (2048 bit, AP30) with agent Morgan2 FP

# In[12]:


infile = gzip.open(dataDir+'training_test_set_patent_data.pkl.gz', 'rb')
pklfile = gzip.open(dataDir+'reaction_FPs_agentMG2FPs_training_test_set_patent_data.pkl.gz','wb+')
lineNo=0
while 1:
    lineNo+=1
    try:
        smi,lbl,klass = cPickle.load(infile)        
    except EOFError:
        break
    try:
        rxn = AllChem.ReactionFromSmarts(smi,useSmiles=True)       
        rxn.RemoveUnmappedReactantTemplates()
        params = AllChem.ReactionFingerprintParams()
        # ignore agents 
        fp_woA = AllChem.CreateDifferenceFingerprintForReaction(rxn, params)
        fp_MG2_agents = createFingerprintsReaction.create_agent_morgan2_FP(rxn)
        if fp_MG2_agents is None:
            fp_MG2_agents = DataStructs.UIntSparseIntVect(4096)
    except:
        print("Cannot build fingerprint/reaction of: %s\n"%smi)
        continue;
    cPickle.dump((lbl,klass,fp_woA,fp_MG2_agents),pklfile,2)
    if not lineNo%5000:
        print("Done: %d"%lineNo)
infile.close()
pklfile.close()


# Combine difference reaction FP (2048 bit, AP30) with agent dictionary-based FP

# In[13]:


infile = gzip.open(dataDir+'training_test_set_patent_data.pkl.gz', 'rb')
pklfile = gzip.open(dataDir+'reaction_FPs_agentDictBasedFPs_training_test_set_patent_data.pkl.gz','wb+')
lineNo=0
while 1:
    lineNo+=1
    try:
        smi,lbl,klass = cPickle.load(infile)        
    except EOFError:
        break
    try:
        rxn = AllChem.ReactionFromSmarts(smi,useSmiles=True)       
        rxn.RemoveUnmappedReactantTemplates()
        params = AllChem.ReactionFingerprintParams()
        # ignore agents 
        fp_woA = AllChem.CreateDifferenceFingerprintForReaction(rxn, params)
        fp_dictinarybased_agents = createFingerprintsReaction.create_agent_dictionary_FP(rxn)
    except:
        print("Cannot build fingerprint/reaction of: %s\n"%smi)
        continue;
    cPickle.dump((lbl,klass,fp_woA,fp_dictinarybased_agents),pklfile,2)
    if not lineNo%5000:
        print("Done: %d"%lineNo)
infile.close()
pklfile.close()


# %%
