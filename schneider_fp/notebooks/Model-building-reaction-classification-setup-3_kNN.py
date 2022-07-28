#!/usr/bin/env python
# coding: utf-8

# ### Learning reaction types using different ML methods (kNN) and more local difference fingerprints (AP3, MG2, TT)

# Goal: generate a baseline ML model for our FPs

# In[1]:


import cPickle,gzip
from collections import defaultdict
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import utilsFunctions


# Choose some larger text size in the plots

# In[2]:


rcParams.update({'font.size': 14})


# In[3]:


dataDir = "../data/"
reaction_types = cPickle.load(file(dataDir+"reactionTypes_training_test_set_patent_data.pkl"))
names_rTypes = cPickle.load(file(dataDir+"names_rTypes_classes_superclasses_training_test_set_patent_data.pkl"))


# Load the different FPs (AP3, MG2, TT)

# In[4]:


infile = gzip.open(dataDir+"transformationFPs_test_set_patent_data.pkl.gz", 'rb')

lineNo=0
fps=[]
idx=0
while 1:
    lineNo+=1
    try:
        lbl,cls,fp_AP3,fp_MG2,fp_TT = cPickle.load(infile)        
    except EOFError:
        break
    fps.append([idx,lbl,cls,fp_AP3,fp_MG2,fp_TT])
    idx+=1
    if not lineNo%10000:
        print "Done "+str(lineNo)


# Split the FPs in training (20 %) and test data (80 %) per recation type (200, 800)

# In[5]:


random.seed(0xd00f)
indices=range(len(fps))
random.shuffle(indices)

nActive=200
fpsz=2048
#fpsz=4096
trainFps_fp_AP3=[]
trainFps_fp_MG2=[]
trainFps_fp_TT=[]
trainActs=[]
testFps_fp_AP3=[]
testFps_fp_MG2=[]
testFps_fp_TT=[]
testActs=[]

print 'building fp collection'

rtypes=sorted(list(reaction_types))
for i,klass in enumerate(rtypes):
    actIds = [x for x in indices if fps[x][2]==klass]
    for x in actIds[:nActive]:
        trainFps_fp_AP3 += [utilsFunctions.fpToNP(fps[x][3],fpsz)]
        trainFps_fp_MG2 += [utilsFunctions.fpToNP(fps[x][4],fpsz)]
        trainFps_fp_TT += [utilsFunctions.fpToNP(fps[x][5],fpsz)]
    trainActs += [i]*nActive
    nTest=len(actIds)-nActive
    for x in actIds[nActive:]:
        testFps_fp_AP3 += [utilsFunctions.fpToNP(fps[x][3],fpsz)]
        testFps_fp_MG2 += [utilsFunctions.fpToNP(fps[x][4],fpsz)]
        testFps_fp_TT += [utilsFunctions.fpToNP(fps[x][5],fpsz)]
    testActs += [i]*nTest


# Build the kNN classifiers with k=3

# In[6]:


kNN_cls_AP3 = KNeighborsClassifier(n_neighbors=3)
result_kNN_fp_AP3 = kNN_cls_AP3.fit(trainFps_fp_AP3,trainActs)
kNN_cls_MG2 = KNeighborsClassifier(n_neighbors=3)
result_kNN_fp_MG2 = kNN_cls_MG2.fit(trainFps_fp_MG2,trainActs)
kNN_cls_TT = KNeighborsClassifier(n_neighbors=3)
result_kNN_fp_TT = kNN_cls_TT.fit(trainFps_fp_TT,trainActs)


# Evaluate the models

# In[7]:


cmat_fp_AP3 = utilsFunctions.evaluateModel(result_kNN_fp_AP3, testFps_fp_AP3, testActs, rtypes, names_rTypes)


# In[15]:


utilsFunctions.labelled_cmat(cmat_fp_AP3,rtypes,figsize=(16,12),labelExtras=names_rTypes, xlabel=False)


# In[9]:


cmat_fp_MG2 = utilsFunctions.evaluateModel(result_kNN_fp_MG2, testFps_fp_MG2, testActs, rtypes, names_rTypes)


# In[16]:


utilsFunctions.labelled_cmat(cmat_fp_MG2,rtypes,figsize=(16,12),labelExtras=names_rTypes)


# In[11]:


cmat_fp_TT = utilsFunctions.evaluateModel(result_kNN_fp_TT, testFps_fp_TT, testActs, rtypes, names_rTypes)


# In[17]:


utilsFunctions.labelled_cmat(cmat_fp_TT,rtypes,figsize=(16,12),labelExtras=names_rTypes)


# Build the kNN classifiers with k=30

# In[18]:


kNN_cls_AP3 = KNeighborsClassifier(n_neighbors=30)
result_kNN_fp_AP3 = kNN_cls_AP3.fit(trainFps_fp_AP3,trainActs)
kNN_cls_MG2 = KNeighborsClassifier(n_neighbors=30)
result_kNN_fp_MG2 = kNN_cls_MG2.fit(trainFps_fp_MG2,trainActs)
kNN_cls_TT = KNeighborsClassifier(n_neighbors=30)
result_kNN_fp_TT = kNN_cls_TT.fit(trainFps_fp_TT,trainActs)


# Evaluate the models

# In[19]:


cmat_fp_AP3 = utilsFunctions.evaluateModel(result_kNN_fp_AP3, testFps_fp_AP3, testActs, rtypes, names_rTypes)


# In[20]:


utilsFunctions.labelled_cmat(cmat_fp_AP3,rtypes,figsize=(16,12),labelExtras=names_rTypes, xlabel=False, ylabel=False)


# In[21]:


cmat_fp_MG2 = utilsFunctions.evaluateModel(result_kNN_fp_MG2, testFps_fp_MG2, testActs, rtypes, names_rTypes)


# In[22]:


utilsFunctions.labelled_cmat(cmat_fp_MG2,rtypes,figsize=(16,12),labelExtras=names_rTypes)


# In[23]:


cmat_fp_TT = utilsFunctions.evaluateModel(result_kNN_fp_TT, testFps_fp_TT, testActs, rtypes, names_rTypes)


# In[24]:


utilsFunctions.labelled_cmat(cmat_fp_TT,rtypes,figsize=(16,12),labelExtras=names_rTypes)

