#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Build-reaction-class-and-super-class-models-(LR,-AP3-+-agent-feature-FP)-and-validate-with-external-test-set-A" data-toc-modified-id="Build-reaction-class-and-super-class-models-(LR,-AP3-+-agent-feature-FP)-and-validate-with-external-test-set-A-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Build reaction class and super-class models (LR, AP3 + agent feature FP) and validate with external test set A</a></span><ul class="toc-item"><li><span><a href="#Build-the-reaction-class-model" data-toc-modified-id="Build-the-reaction-class-model-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Build the reaction class model</a></span></li><li><span><a href="#Build-the-reaction-super-class-model" data-toc-modified-id="Build-the-reaction-super-class-model-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Build the reaction super-class model</a></span></li><li><span><a href="#Load-the-new-models-and-test-on-external-test-set-A" data-toc-modified-id="Load-the-new-models-and-test-on-external-test-set-A-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Load the new models and test on external test set A</a></span></li></ul></li></ul></div>

# ### Build reaction class and super-class models (LR, AP3 + agent feature FP) and validate with external test set A

# In[1]:


import gzip
import random
import _pickle as cPickle
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import numpy as np

import utilsFunctions


# Choose some larger text size in the plots

# In[2]:


# rcParams.update({'font.size': 14})


# In[3]:


dataDir = "../data/"
with open(dataDir+"reactionTypes_training_test_set_patent_data.pkl", 'rb') as f:
    reaction_types = cPickle.load(f)
with open(dataDir+"names_rTypes_classes_superclasses_training_test_set_patent_data.pkl", 'rb') as f:
    names_rTypes = cPickle.load(f)


# Load the AP3 and agent feature fingerprint

# In[4]:


infile = gzip.open(dataDir+"transformationFPs_MG2_agentFPs_test_set_patent_data.pkl.gz", 'rb')

lineNo=0
fps=[]
idx=0
while 1:
    lineNo+=1
    try:
        lbl,cls,fp_AP3,fp_agentFeature, fp_agentMG2 = cPickle.load(infile)        
    except EOFError:
        break
    fps.append([idx,lbl,cls,fp_AP3,fp_agentFeature])
    idx+=1
    if not lineNo%10000:
        print("Done "+str(lineNo))


# Build sets of reaction classes and super-classes

# In[5]:


reaction_classes = set(x.split('.')[0]+"."+x.split('.')[1] for x in reaction_types)
print(reaction_classes)
reaction_superclasses = set(x.split('.')[0] for x in reaction_types)
print(reaction_superclasses)


# #### Build the reaction class model

# Split the FPs in training (20 %) and test data (80 %) per reaction class

# In[6]:


random.seed(0xd00f)
indices=list(range(len(fps)))
random.shuffle(indices)

fpsz=256
trainFps_AP3_agentFeature=[]
trainActs=[]
testFps_AP3_agentFeature=[]
testActs=[]

print('building fp collection')

rclasses=sorted(list(reaction_classes))
for i,klass in enumerate(rclasses):
    actIds = [x for x in indices if (fps[x][2].split('.')[0]+"."+fps[x][2].split('.')[1])==klass]
    nActive = (len(actIds)/5) # take 20% for training
    assert nActive.is_integer()
    nActive = int(nActive)
    for x in actIds[:nActive]:
        np1 = utilsFunctions.fpToNPfloat(fps[x][3],fpsz)
        np2 = np.asarray(fps[x][4], dtype=float)
        trainFps_AP3_agentFeature += [np.concatenate([np1, np2])]
    trainActs += [i]*nActive
    nTest=len(actIds)-nActive
    for x in actIds[nActive:]:
        np1 = utilsFunctions.fpToNPfloat(fps[x][3],fpsz)
        np2 = np.asarray(fps[x][4], dtype=float)
        testFps_AP3_agentFeature += [np.concatenate([np1, np2])]
    testActs += [i]*nTest


# Train the LR classifier

# In[7]:


print('training model')
lr_cls =  LogisticRegression()
result_lr_cls = lr_cls.fit(trainFps_AP3_agentFeature,trainActs)


# Evaluate the class model

# In[8]:


cmat_fp_AP3_feature = utilsFunctions.evaluateModel(result_lr_cls, testFps_AP3_agentFeature, testActs, rclasses, names_rTypes)


# Draw the confusion matrix

# In[9]:


utilsFunctions.labelled_cmat(cmat_fp_AP3_feature,rclasses,figsize=(16,12), labelExtras=names_rTypes)


# In[10]:


# from sklearn.externals import joblib
import joblib


# Store the model as scikit-learn model

# In[11]:


joblib.dump(result_lr_cls, dataDir+'LR_transformationFP256bit.AP3.agent_featureFP_classModel.pkl') 


# #### Build the reaction super-class model

# Split the FPs in training (20 %) and test data (80 %) per reaction super-class

# In[12]:


random.seed(0xd00f)
indices=list(range(len(fps)))
random.shuffle(indices)

fpsz=256
trainFps_AP3_agentFeature=[]
trainActs=[]
testFps_AP3_agentFeature=[]
testActs=[]

print('building fp collection')

rsclasses=sorted(list(reaction_superclasses))
for i,klass in enumerate(rsclasses):
    actIds = [x for x in indices if (fps[x][2].split('.')[0])==klass]
    nActive = (len(actIds)/5) # take 20% for training
    assert nActive.is_integer()
    nActive = int(nActive)
    for x in actIds[:nActive]:
        np1 = utilsFunctions.fpToNPfloat(fps[x][3],fpsz)
        np2 = np.asarray(fps[x][4], dtype=float)
        trainFps_AP3_agentFeature += [np.concatenate([np1, np2])]
    trainActs += [i]*nActive
    nTest=len(actIds)-nActive
    for x in actIds[nActive:]:
        np1 = utilsFunctions.fpToNPfloat(fps[x][3],fpsz)
        np2 = np.asarray(fps[x][4], dtype=float)
        testFps_AP3_agentFeature += [np.concatenate([np1, np2])]
    testActs += [i]*nTest


# Train the LR classifier 

# In[13]:


print('training model')
lr_cls =  LogisticRegression()
result_lr_cls = lr_cls.fit(trainFps_AP3_agentFeature,trainActs)


# Evaluate the super-class model

# In[14]:


cmat_fp_AP3_feature = utilsFunctions.evaluateModel(result_lr_cls, testFps_AP3_agentFeature, testActs, rsclasses, names_rTypes)


# In[15]:


utilsFunctions.labelled_cmat(cmat_fp_AP3_feature,rsclasses,figsize=(16,12), labelExtras=names_rTypes)


# Store the model as scikit-learn model

# In[16]:


joblib.dump(result_lr_cls, dataDir+'LR_transformationFP256bit.AP3.agent_featureFP_superclassModel.pkl') 


# #### Load the new models and test on external test set A

# In[17]:


clf1 = joblib.load(dataDir+'LR_transformationFP256bit.AP3.agent_featureFP_classModel.pkl')
clf2 = joblib.load(dataDir+'LR_transformationFP256bit.AP3.agent_featureFP_superclassModel.pkl')


# Load the fingerprints of external test set A, another 50000 reactions randomly selected form the patent data

# In[18]:


infile = gzip.open(dataDir+"transformationFPs_agentFPs_external_test_set_A.pkl.gz", 'rb')
lineNo=0
fps=[]
idx=0
while 1:
    lineNo+=1
    try:
        lbl,cls,apfp_woA,agent_feature = cPickle.load(infile)        
    except EOFError:
        break
    fps.append([idx,lbl,cls,apfp_woA,agent_feature])
    idx+=1
    if not lineNo%10000:
        print("Done "+str(lineNo))


# Combine the AP3 and agent feature FPs for the class model

# In[19]:


random.seed(0xd00f)
indices=list(range(len(fps)))
random.shuffle(indices)

fpsz=256

testFps_AP3_agentFeature=[]
testActs=[]
print('building fp collection')

rclasses=sorted(list(reaction_classes))
for i,klass in enumerate(rclasses):
    actIds = [x for x in indices if (fps[x][2].split('.')[0]+"."+fps[x][2].split('.')[1])==klass]
    nTest=len(actIds)
    for x in actIds:
        np1 = utilsFunctions.fpToNPfloat(fps[x][3],fpsz)
        np2 = np.asarray(fps[x][4], dtype=float)
        testFps_AP3_agentFeature += [np.concatenate([np1, np2])]
    testActs += [i]*nTest


# Validate the class model on the external test set

# In[21]:


cmat_fp_AP3_feature = utilsFunctions.evaluateModel(clf1, testFps_AP3_agentFeature, testActs, rclasses, names_rTypes)


# In[22]:


utilsFunctions.labelled_cmat(cmat_fp_AP3_feature,rclasses,figsize=(16,12), labelExtras=names_rTypes)


# Combine the AP3 and agent feature FPs for the super-class model

# In[23]:


random.seed(0xd00f)
indices=list(range(len(fps)))
random.shuffle(indices)

fpsz=256

testFps_AP3_agentFeature=[]
testActs=[]
print('building fp collection')

rsclasses=sorted(list(reaction_superclasses))
for i,klass in enumerate(rsclasses):
    actIds = [x for x in indices if (fps[x][2].split('.')[0])==klass]
    nTest=len(actIds)
    for x in actIds:
        np1 = utilsFunctions.fpToNPfloat(fps[x][3],fpsz)
        np2 = np.asarray(fps[x][4], dtype=float)
        testFps_AP3_agentFeature += [np.concatenate([np1, np2])]
    testActs += [i]*nTest


# Validate the super-class model on the external test set

# In[24]:


cmat_fp_AP3_feature = utilsFunctions.evaluateModel(clf2, testFps_AP3_agentFeature, testActs, rsclasses, names_rTypes)


# In[25]:


utilsFunctions.labelled_cmat(cmat_fp_AP3_feature,rsclasses,figsize=(16,12), labelExtras=names_rTypes)


# %%
