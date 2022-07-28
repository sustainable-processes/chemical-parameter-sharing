#!/usr/bin/env python
# coding: utf-8

# ### Test the final model (LR, AP3 + agent feature FP) by Y-scrambling and with external test set A

# In[1]:


import _pickle as cPickle
import gzip
from collections import defaultdict
import random
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import utilsFunctions


# Choose some larger text size in the plots

# In[2]:


rcParams.update({'font.size': 14})


# In[3]:


dataDir = "../data/"
reaction_types = cPickle.load(file(dataDir+"reactionTypes_training_test_set_patent_data.pkl"))
names_rTypes = cPickle.load(file(dataDir+"names_rTypes_classes_superclasses_training_test_set_patent_data.pkl"))


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
        print "Done "+str(lineNo)


# Split the FPs in training (20 %) and test data (80 %) per recation type (200, 800)

# In[5]:


random.seed(0xd00f)
indices=range(len(fps))
random.shuffle(indices)

nActive=200
fpsz=256
trainFps_AP3_agentFeature=[]
trainActs=[]
testFps_AP3_agentFeature=[]
testActs=[]

print 'building fp collection'

rtypes=sorted(list(reaction_types))
for i,klass in enumerate(rtypes):
    actIds = [x for x in indices if fps[x][2]==klass]
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


# Test y-scrambling of our training data

# In[6]:


import copy

lr_cls =  LogisticRegression()

trainActs_temp = copy.deepcopy(trainActs)
random.seed(42)
random.shuffle(trainActs_temp)
result_lr_cls = lr_cls.fit(trainFps_AP3_agentFeature,trainActs_temp)


# Evaluate the y-scrambled model

# In[7]:


cmat_fp_AP3_feature = utilsFunctions.evaluateModel(result_lr_cls, testFps_AP3_agentFeature, testActs, rtypes, names_rTypes)


# In[8]:


utilsFunctions.labelled_cmat(cmat_fp_AP3_feature,rtypes,figsize=(16,12), labelExtras=names_rTypes)


# #### Build the final model and save

# In[9]:


print 'training model'
lr_cls =  LogisticRegression()
result_lr_cls = lr_cls.fit(trainFps_AP3_agentFeature,trainActs)


# In[10]:


cmat_fp_AP3_feature = utilsFunctions.evaluateModel(result_lr_cls, testFps_AP3_agentFeature, testActs, rtypes, names_rTypes)


# Store the model as scikit-learn model

# In[11]:


from sklearn.externals import joblib


# In[12]:


joblib.dump(result_lr_cls, dataDir+'LR_transformationFP256bit.AP3.agent_featureFP.pkl') 


# #### Load the new model and test on external test set A

# In[13]:


clf = joblib.load(dataDir+'LR_transformationFP256bit.AP3.agent_featureFP.pkl')


# Test the final model (LR, AP3 256 bit + agent featureFP) with external test set A (another 50000 reactions randomly selected form the patent data)

# Load the FPs

# In[14]:


infile = gzip.open(dataDir+"transformationFPs_agentFPs_external_test_set_A.pkl.gz", 'rb')
lineNo=0
fps=[]
idx=0
while 1:
    lineNo+=1
    try:
        lbl,cls,fp_AP3,fp_agentFeature = cPickle.load(infile)        
    except EOFError:
        break
    fps.append([idx,lbl,cls,fp_AP3,fp_agentFeature])
    idx+=1
    if not lineNo%10000:
        print "Done "+str(lineNo)


# Combine the FPs of the external test set

# In[15]:


random.seed(0xd00f)
indices=range(len(fps))
random.shuffle(indices)

fpsz=256

testFps_AP3_agentFeature=[]
testActs=[]
print 'building fp collection'

rtypes=sorted(list(reaction_types))
for i,klass in enumerate(rtypes):
    actIds = [x for x in indices if fps[x][2]==klass]
    nTest=len(actIds)
    for x in actIds:
        np1 = utilsFunctions.fpToNPfloat(fps[x][3],fpsz)
        np2 = np.asarray(fps[x][4], dtype=float)
        testFps_AP3_agentFeature += [np.concatenate([np1, np2])]
    testActs += [i]*nTest


# Evaluate the model performance

# In[16]:


cmat_fp_AP3_feature = utilsFunctions.evaluateModel(clf, testFps_AP3_agentFeature, testActs, rtypes, names_rTypes)


# In[17]:


utilsFunctions.labelled_cmat(cmat_fp_AP3_feature,rtypes,figsize=(16,12), labelExtras=names_rTypes)

