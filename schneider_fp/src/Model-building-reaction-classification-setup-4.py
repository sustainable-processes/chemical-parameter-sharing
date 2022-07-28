#!/usr/bin/env python
# coding: utf-8

# ### Learning reaction types using different bit sizes of the transformation fingerprints (AP3) and different agent fingerprints

# Goal: further reduce the size/complexity of the FP/model

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


# Load the AP3 fingerprint, agent feature and MG2 fingerprints

# In[4]:


infile = gzip.open(dataDir+"transformationFPs_MG2_agentFPs_test_set_patent_data.pkl.gz", 'rb')

lineNo=0
fps=[]
idx=0
while 1:
    lineNo+=1
    try:
        lbl,cls,fp_AP3,fp_agentFeature,fp_agentMG2 = cPickle.load(infile)        
    except EOFError:
        break
    fps.append([idx,lbl,cls,fp_AP3,fp_agentFeature,fp_agentMG2])
    idx+=1
    if not lineNo%10000:
        print "Done "+str(lineNo)


# Combine the transformation FP with agent feature as well as Morgan2 FP. Try different bit sizes. Split the FPs in training (20 %) and test data (80 %) per recation type (200, 800)

# In[5]:


random.seed(0xd00f)
indices=range(len(fps))
random.shuffle(indices)

nActive=200
#fpsz=4096
#fpsz=2048
#fpsz=1024
#fpsz=512
fpsz=256
#fpsz=128
trainFps_AP3=[]
trainFps_AP3_agentFeature=[]
trainFps_AP3_agentMG2=[]
trainActs=[]
testFps_AP3=[]
testFps_AP3_agentFeature=[]
testFps_AP3_agentMG2=[]
testActs=[]

print 'building fp collection'

rtypes=sorted(list(reaction_types))
for i,klass in enumerate(rtypes):
    actIds = [x for x in indices if fps[x][2]==klass]
    for x in actIds[:nActive]:
        np1_feature = utilsFunctions.fpToNPfloat(fps[x][3],fpsz)
        np2_feature = np.asarray(fps[x][4], dtype=float)
        trainFps_AP3_agentFeature += [np.concatenate([np1_feature, np2_feature])]
        np1_morgan = utilsFunctions.fpToNP(fps[x][3],fpsz)
        trainFps_AP3 += [np1_morgan]
        np2_morgan = utilsFunctions.fpToNP(fps[x][5],fpsz)
        trainFps_AP3_agentMG2 += [np.concatenate([np1_morgan, np2_morgan])]
    trainActs += [i]*nActive
    nTest=len(actIds)-nActive
    for x in actIds[nActive:]:
        np1_feature = utilsFunctions.fpToNPfloat(fps[x][3],fpsz)
        np2_feature = np.asarray(fps[x][4], dtype=float)
        testFps_AP3_agentFeature += [np.concatenate([np1_feature, np2_feature])]
        np1_morgan = utilsFunctions.fpToNP(fps[x][3],fpsz)
        testFps_AP3 += [np1_morgan]
        np2_morgan = utilsFunctions.fpToNP(fps[x][5],fpsz)
        testFps_AP3_agentMG2 += [np.concatenate([np1_morgan, np2_morgan])]
    testActs += [i]*nTest


# Build the LR classifiers

# In[6]:


print 'training models'
lr_cls_AP3 = LogisticRegression()
result_lr_fp_AP3 = lr_cls_AP3.fit(trainFps_AP3,trainActs)
lr_cls_AP3_feature = LogisticRegression()
result_lr_fp_AP3_feature = lr_cls_AP3_feature.fit(trainFps_AP3_agentFeature,trainActs)
lr_cls_AP3_MG2 = LogisticRegression()
result_lr_fp_AP3_MG2 = lr_cls_AP3_MG2.fit(trainFps_AP3_agentMG2,trainActs)


# Evalutate the LR classifier using our test data

# In[7]:


cmat_fp_AP3 = utilsFunctions.evaluateModel(result_lr_fp_AP3, testFps_AP3, testActs, rtypes, names_rTypes)


# In[8]:


utilsFunctions.labelled_cmat(cmat_fp_AP3,rtypes,figsize=(16,12), labelExtras=names_rTypes)


# In[9]:


cmat_fp_AP3_feature = utilsFunctions.evaluateModel(result_lr_fp_AP3_feature, testFps_AP3_agentFeature, testActs, rtypes, names_rTypes)


# In[11]:


utilsFunctions.labelled_cmat(cmat_fp_AP3_feature,rtypes,figsize=(16,12), labelExtras=names_rTypes, ylabel=False)


# In[12]:


cmat_fp_AP3_MG2 = utilsFunctions.evaluateModel(result_lr_fp_AP3_MG2, testFps_AP3_agentMG2, testActs, rtypes, names_rTypes)


# In[13]:


utilsFunctions.labelled_cmat(cmat_fp_AP3_MG2,rtypes,figsize=(16,12),labelExtras=names_rTypes)


# #### Using DictVectorizer functionality to check the performance of the unfolded version of the fingerprint

# In[14]:


from sklearn.feature_extraction import DictVectorizer


# Build the unfolded FP using the DictVectorizer functionality. In this experiment ignore the agents.

# In[15]:


random.seed(0xd00f)
indices=range(len(fps))
random.shuffle(indices)

nActive=200

trainFps_AP3_unfolded=[]
trainActs=[]
testFps_AP3_unfolded=[]
testActs=[]

print 'building fp collection'

rtypes=sorted(list(reaction_types))
for i,klass in enumerate(rtypes):
    actIds = [x for x in indices if fps[x][2]==klass]
    for x in actIds[:nActive]:
        trainFps_AP3_unfolded.append(fps[x][3])
    trainActs += [i]*nActive
    nTest=len(actIds)-nActive
    for x in actIds[nActive:]:
        testFps_AP3_unfolded.append(fps[x][3])
    testActs += [i]*nTest
dictVectmodel = utilsFunctions.prepareUnfoldedData(trainFps_AP3_unfolded+testFps_AP3_unfolded)
trainFps_new = [x.GetNonzeroElements() for x in trainFps_AP3_unfolded]
testFps_new = [x.GetNonzeroElements() for x in testFps_AP3_unfolded]
trainFps_new = dictVectmodel.transform(trainFps_new)
testFps_new = dictVectmodel.transform(testFps_new)


# Look at the number of unique bits

# In[16]:


len(dictVectmodel.feature_names_)


# Train the LR classifier with the unfolded version of the FP

# In[17]:


print 'training model'
lr_cls_AP3_unfolded =  LogisticRegression()
result_lr_fp_AP3_unfolded = lr_cls_AP3_unfolded.fit(trainFps_new,trainActs)


# In[18]:


cmat_fp_AP3_unfolded = utilsFunctions.evaluateModel(result_lr_fp_AP3_unfolded, testFps_new, testActs, rtypes, names_rTypes)


# In[19]:


utilsFunctions.labelled_cmat(cmat_fp_AP3_unfolded,rtypes,figsize=(16,12),labelExtras=names_rTypes, xlabel=False)


# Build the unfolded FP using the DictVectorizer functionality. Now combine FPs with the agent feature FP.

# In[20]:


random.seed(0xd00f)
indices=range(len(fps))
random.shuffle(indices)

nActive=200

trainFps_AP3_feature_unfolded=[]
trainActs=[]
testFps_AP3_feature_unfolded=[]
testActs=[]

print 'building fp collection'

rtypes=sorted(list(reaction_types))
for i,klass in enumerate(rtypes):
    actIds = [x for x in indices if fps[x][2]==klass]
    for x in actIds[:nActive]:
        trainFps_AP3_feature_unfolded.append(utilsFunctions.mergeDicts(utilsFunctions.fpToFloatDict(fps[x][3]),utilsFunctions.listToFloatDict(fps[x][4])))
    trainActs += [i]*nActive
    nTest=len(actIds)-nActive
    for x in actIds[nActive:]:
        testFps_AP3_feature_unfolded.append(utilsFunctions.mergeDicts(utilsFunctions.fpToFloatDict(fps[x][3]),utilsFunctions.listToFloatDict(fps[x][4])))
    testActs += [i]*nTest
    
v = DictVectorizer(sparse=False)
tfp = [x for x in trainFps_AP3_feature_unfolded]
tfp += [x for x in testFps_AP3_feature_unfolded]
dictVectmodel = v.fit(tfp)
trainFps_new = dictVectmodel.transform(trainFps_AP3_feature_unfolded)
testFps_new = dictVectmodel.transform(testFps_AP3_feature_unfolded)


# Look at the number of unique bits

# In[21]:


len(dictVectmodel.feature_names_)


# In[22]:


print 'training model'
lr_cls_AP3_feature_unfolded =  LogisticRegression()
result_lr_fp_AP3_feature_unfolded = lr_cls_AP3_feature_unfolded.fit(trainFps_new,trainActs)


# In[23]:


cmat_fp_AP3_feature_unfolded = utilsFunctions.evaluateModel(result_lr_fp_AP3_feature_unfolded, testFps_new, testActs, rtypes, names_rTypes)


# In[25]:


utilsFunctions.labelled_cmat(cmat_fp_AP3_feature_unfolded,rtypes,figsize=(16,12),labelExtras=names_rTypes, xlabel=False, ylabel=False)


# Combine the unfolded FPs with the unfolded agent MG2 FP.

# In[26]:


random.seed(0xd00f)
indices=range(len(fps))
random.shuffle(indices)

nActive=200

trainFps_AP3_MG2_unfolded=[]
trainActs=[]
testFps_AP3_MG2_unfolded=[]
testActs=[]

print 'building fp collection'

rtypes=sorted(list(reaction_types))
for i,klass in enumerate(rtypes):
    actIds = [x for x in indices if fps[x][2]==klass]
    for x in actIds[:nActive]:
        trainFps_AP3_MG2_unfolded.append(utilsFunctions.mergeFps2Dict(fps[x][3],fps[x][5]))
    trainActs += [i]*nActive
    nTest=len(actIds)-nActive
    for x in actIds[nActive:]:
        testFps_AP3_MG2_unfolded.append(utilsFunctions.mergeFps2Dict(fps[x][3],fps[x][5]))
    testActs += [i]*nTest
    
    
v = DictVectorizer(sparse=False)
tfp = [x for x in trainFps_AP3_MG2_unfolded]
tfp += [x for x in testFps_AP3_MG2_unfolded]
dictVectmodel = v.fit(tfp)
trainFps_new = dictVectmodel.transform(trainFps_AP3_MG2_unfolded)
testFps_new = dictVectmodel.transform(testFps_AP3_MG2_unfolded)


# Look at the number of unique bits

# In[27]:


len(dictVectmodel.feature_names_)


# In[28]:


print 'training model'
lr_cls_AP3_MG2_unfolded = LogisticRegression()
result_lr_fp_AP3_MG2_unfolded = lr_cls_AP3_MG2_unfolded.fit(trainFps_new,trainActs)


# In[29]:


cmat_fp_AP3_MG2_unfolded = utilsFunctions.evaluateModel(result_lr_fp_AP3_MG2_unfolded, testFps_new, testActs, rtypes, names_rTypes)


# In[30]:


utilsFunctions.labelled_cmat(cmat_fp_AP3_MG2_unfolded,rtypes,figsize=(16,12),labelExtras=names_rTypes)

