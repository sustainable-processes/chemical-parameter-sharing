#!/usr/bin/env python
# coding: utf-8

# ### Learning reaction types using a Random Forest classifier and different reaction fingerprints 

# Goal: Include special fingerprints for agents (feature FP, Morgan2, dictionary-based FP), concatenate them with the 2048 bit reaction FP

# In[1]:


import cPickle,gzip
from collections import defaultdict
import random
from sklearn.ensemble import RandomForestClassifier
import utilsFunctions


# Choose some larger text size in the plots

# In[2]:


rcParams.update({'font.size': 14})


# In[4]:


dataDir = "../data/"
reaction_types = cPickle.load(file(dataDir+"reactionTypes_training_test_set_patent_data.pkl"))
names_rTypes = cPickle.load(file(dataDir+"names_rTypes_classes_superclasses_training_test_set_patent_data.pkl"))


# #### First: test the agent feature FPs

# Load the FPs (reaction FP and agent feature FP)

# In[5]:


infile = gzip.open(dataDir+"reaction_FPs_agentFeatureFPs_training_test_set_patent_data.pkl.gz", 'rb')

lineNo=0
fps=[]
idx=0
while 1:
    lineNo+=1
    try:
        lbl,cls,fp_woA,fp_featureAgent = cPickle.load(infile)        
    except EOFError:
        break
    fps.append([idx,lbl,cls,fp_woA,fp_featureAgent])
    idx+=1
    if not lineNo%10000:
        print "Done "+str(lineNo)


# Build the combination of reaction FP and agent feature FPs. Split the FPs in training (20 %) and test data (80 %) per recation type (200, 800)

# In[6]:


random.seed(0xd00f)
indices=range(len(fps))
random.shuffle(indices)

nActive=200
fpsz=2048 # the FPs bit size for converting the FPs to a numpy array
#fpsz=4096
trainFps_rFP_agentFeature=[]
trainActs=[]
testFps_rFP_agentFeature=[]
testActs=[]

print 'building fp collection'

rtypes=sorted(list(reaction_types))
for i,klass in enumerate(rtypes):
    actIds = [x for x in indices if fps[x][2]==klass]
    for x in actIds[:nActive]:
        np1 = utilsFunctions.hashedFPToNPfloat(fps[x][3],fpsz)
        np2 = np.asarray(fps[x][4], dtype=float)
        trainFps_rFP_agentFeature += [np.concatenate([np1, np2])]
    trainActs += [i]*nActive
    nTest=len(actIds)-nActive
    for x in actIds[nActive:]:
        np1 = utilsFunctions.hashedFPToNPfloat(fps[x][3],fpsz)
        np2 = np.asarray(fps[x][4], dtype=float)
        testFps_rFP_agentFeature += [np.concatenate([np1, np2])]
    testActs += [i]*nTest


# Build the RF classifier with max tree depth of 25

# In[7]:


print 'training model'
rf_cls = RandomForestClassifier(n_estimators=200, max_depth=25,random_state=23,n_jobs=1)
result_rf_rFP_agentFeature = rf_cls.fit(trainFps_rFP_agentFeature,trainActs)


# Evaluate the RF classifier using our test data

# In[8]:


cmat_rFP_agentFeature = utilsFunctions.evaluateModel(result_rf_rFP_agentFeature, testFps_rFP_agentFeature, testActs, rtypes, names_rTypes)


# Draw the confusion matrix

# In[9]:


utilsFunctions.labelled_cmat(cmat_rFP_agentFeature,rtypes,figsize=(16,12), labelExtras=names_rTypes)


# #### Second: test the agent Morgan2 FPs

# Load the FPs (reaction FP and agent Morgan2 FP)

# In[10]:


infile = gzip.open(dataDir+"reaction_FPs_agentMG2FPs_training_test_set_patent_data.pkl.gz", 'rb')

lineNo=0
fps=[]
idx=0
while 1:
    lineNo+=1
    try:
        lbl,cls,fp_woA,fp_MG2_agents = cPickle.load(infile)        
    except EOFError:
        break
    fps.append([idx,lbl,cls,fp_woA,fp_MG2_agents])
    idx+=1
    if not lineNo%10000:
        print "Done "+str(lineNo)


# Build the combination of reaction FP and agent Morgan2 FP. Split the FPs in training (20 %) and test data (80 %) per recation type (200, 800)

# In[11]:


random.seed(0xd00f)
indices=range(len(fps))
random.shuffle(indices)

nActive=200
fpsz=2048 # the FPs bit size for converting the FPs to a numpy array
#fpsz=4096

trainFps_rFP_agentMG2=[]
trainActs=[]
testFps_rFP_agentMG2=[]
testActs=[]

print 'building fp collection'

rtypes=sorted(list(reaction_types))
for i,klass in enumerate(rtypes):
    actIds = [x for x in indices if fps[x][2]==klass]
    for x in actIds[:nActive]:
        np1 = utilsFunctions.hashedFPToNP(fps[x][3],fpsz)
        np2 = utilsFunctions.fpToNP(fps[x][4],fpsz)
        trainFps_rFP_agentMG2 += [np.concatenate([np1, np2])]
    trainActs += [i]*nActive
    nTest=len(actIds)-nActive
    for x in actIds[nActive:]:
        np1 = utilsFunctions.hashedFPToNP(fps[x][3],fpsz)
        np2 = utilsFunctions.fpToNP(fps[x][4],fpsz)
        testFps_rFP_agentMG2 += [np.concatenate([np1, np2])]
    testActs += [i]*nTest


# In[12]:


print 'training model'
rf_cls = RandomForestClassifier(n_estimators=200, max_depth=25,random_state=23,n_jobs=1)
result_rf_rFP_agentMG2 = rf_cls.fit(trainFps_rFP_agentMG2,trainActs)


# In[13]:


cmat_rFP_agentMG2 = utilsFunctions.evaluateModel(result_rf_rFP_agentMG2, testFps_rFP_agentMG2, testActs, rtypes, names_rTypes)


# In[14]:


utilsFunctions.labelled_cmat(cmat_rFP_agentMG2,rtypes,figsize=(16,12),labelExtras=names_rTypes)


# #### Third: test the agent dictionary-based FP

# Load the FPs (reaction FP and agent dictionary-based FP)

# In[15]:


infile = gzip.open(dataDir+"reaction_FPs_agentDictBasedFPs_training_test_set_patent_data.pkl.gz", 'rb')

lineNo=0
fps=[]
idx=0
while 1:
    lineNo+=1
    try:
        lbl,cls,fp_woA,fp_dictinarybased_agents = cPickle.load(infile)        
    except EOFError:
        break
    fps.append([idx,lbl,cls,fp_woA,fp_dictinarybased_agents])
    idx+=1
    if not lineNo%10000:
        print "Done "+str(lineNo)


# In[16]:


random.seed(0xd00f)
indices=range(len(fps))
random.shuffle(indices)

nActive=200
fpsz=2048 # the FPs bit size for converting the FPs to a numpy array
#fpsz=4096

trainFps_rFP_agentDictBased=[]
trainActs=[]
testFps_rFP_agentDictBased=[]
testActs=[]

print 'building fp collection'

rtypes=sorted(list(reaction_types))
for i,klass in enumerate(rtypes):
    actIds = [x for x in indices if fps[x][2]==klass]
    for x in actIds[:nActive]:
        np1 = utilsFunctions.hashedFPToNP(fps[x][3],fpsz)
        np2 = utilsFunctions.fpDictToNP(fps[x][4])
        trainFps_rFP_agentDictBased += [np.concatenate([np1, np2])]
    trainActs += [i]*nActive
    nTest=len(actIds)-nActive
    for x in actIds[nActive:]:
        np1 = utilsFunctions.hashedFPToNP(fps[x][3],fpsz)
        np2 = utilsFunctions.fpDictToNP(fps[x][4])
        testFps_rFP_agentDictBased += [np.concatenate([np1, np2])]
    testActs += [i]*nTest


# In[17]:


print 'training model'
rf_cls = RandomForestClassifier(n_estimators=200, max_depth=25,random_state=23,n_jobs=1)
result_rf_rFP_agentDictBased = rf_cls.fit(trainFps_rFP_agentDictBased,trainActs)


# In[18]:


cmat_rFP_agentDictBased = utilsFunctions.evaluateModel(result_rf_rFP_agentDictBased, testFps_rFP_agentDictBased, testActs, rtypes, names_rTypes)


# In[19]:


utilsFunctions.labelled_cmat(cmat_rFP_agentDictBased,rtypes,figsize=(16,12),labelExtras=names_rTypes)

