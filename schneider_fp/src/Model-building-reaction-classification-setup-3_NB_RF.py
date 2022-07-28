#!/usr/bin/env python
# coding: utf-8

# ### Learning reaction types using different ML methods (RF, NB) and more local difference fingerprints (AP3, MG2, TT)

# Goal: find the best/simplest/most appropriate ML model for our FPs

# In[1]:


import _pickle as cPickle
import gzip
from collections import defaultdict
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
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


# Split the FPs in training (20 %) and test data (80 %) per recation type (200, 800).
# Build the "positive" version of the difference transformation fingerprints.

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
        trfp_AP3 = utilsFunctions.hashFP(fps[x][3],fpsz)
        trfp_AP3 = utilsFunctions.removeNegativeCountsFP(trfp_AP3)
        trainFps_fp_AP3 += [utilsFunctions.hashedFPToNP(trfp_AP3,fpsz*2)]
        trfp_MG2 = utilsFunctions.hashFP(fps[x][4],fpsz)
        trfp_MG2 = utilsFunctions.removeNegativeCountsFP(trfp_MG2)
        trainFps_fp_MG2 += [utilsFunctions.hashedFPToNP(trfp_MG2,fpsz*2)]
        trfp_TT = utilsFunctions.hashFP(fps[x][5],fpsz)
        trfp_TT = utilsFunctions.removeNegativeCountsFP(trfp_TT)
        trainFps_fp_TT += [utilsFunctions.hashedFPToNP(trfp_TT,fpsz*2)]
    trainActs += [i]*nActive
    nTest=len(actIds)-nActive
    for x in actIds[nActive:]:
        tefp_AP3 = utilsFunctions.hashFP(fps[x][3],fpsz)
        tefp_AP3 = utilsFunctions.removeNegativeCountsFP(tefp_AP3)
        testFps_fp_AP3 += [utilsFunctions.hashedFPToNP(tefp_AP3,fpsz*2)]
        tefp_MG2 = utilsFunctions.hashFP(fps[x][4],fpsz)
        tefp_MG2 = utilsFunctions.removeNegativeCountsFP(tefp_MG2)
        testFps_fp_MG2 += [utilsFunctions.hashedFPToNP(tefp_MG2,fpsz*2)]
        tefp_TT = utilsFunctions.hashFP(fps[x][5],fpsz)
        tefp_TT = utilsFunctions.removeNegativeCountsFP(tefp_TT)
        testFps_fp_TT += [utilsFunctions.hashedFPToNP(tefp_TT,fpsz*2)]
    testActs += [i]*nTest


# Build the multinomial NB classifer with the "positive" version of the FPs

# In[6]:


print 'training models'
clf_AP3 = MultinomialNB(alpha=0.0001)
res_NB_AP3 = clf_AP3.fit(trainFps_fp_AP3, trainActs)
clf_MG2 = MultinomialNB(alpha=0.0001)
res_NB_MG2 = clf_MG2.fit(trainFps_fp_MG2, trainActs)
clf_TT = MultinomialNB(alpha=0.0001)
res_NB_TT = clf_TT.fit(trainFps_fp_TT, trainActs)


# Evaluate the classifier

# In[7]:


cmat_NB_fp_AP3 = utilsFunctions.evaluateModel(res_NB_AP3, testFps_fp_AP3, testActs, rtypes, names_rTypes)


# In[8]:


cmat_NB_fp_MG2 = utilsFunctions.evaluateModel(res_NB_MG2, testFps_fp_MG2, testActs, rtypes, names_rTypes)


# In[9]:


cmat_NB_fp_TT = utilsFunctions.evaluateModel(res_NB_TT, testFps_fp_TT, testActs, rtypes, names_rTypes)


# Draw the confusion matrices

# In[11]:


utilsFunctions.labelled_cmat(cmat_NB_fp_AP3,rtypes,figsize=(16,12), labelExtras=names_rTypes)


# In[12]:


utilsFunctions.labelled_cmat(cmat_NB_fp_MG2,rtypes,figsize=(16,12), labelExtras=names_rTypes)


# In[13]:


utilsFunctions.labelled_cmat(cmat_NB_fp_TT,rtypes,figsize=(16,12), labelExtras=names_rTypes)


# #### Test the "positive" version of the FPs with RF classifiers (max tree depth = 25)

# In[14]:


print 'training models'
rf_cls_AP3 = RandomForestClassifier(n_estimators=200, max_depth=25,random_state=23,n_jobs=1)
result_rf_fp_AP3 = rf_cls_AP3.fit(trainFps_fp_AP3,trainActs)
rf_cls_MG2 = RandomForestClassifier(n_estimators=200, max_depth=25,random_state=23,n_jobs=1)
result_rf_fp_MG2 = rf_cls_MG2.fit(trainFps_fp_MG2,trainActs)
rf_cls_TT = RandomForestClassifier(n_estimators=200, max_depth=25,random_state=23,n_jobs=1)
result_rf_fp_TT = rf_cls_TT.fit(trainFps_fp_TT,trainActs)


# Evalutate the RF classifier using our test data

# In[15]:


cmat_fp_AP3 = utilsFunctions.evaluateModel(result_rf_fp_AP3, testFps_fp_AP3, testActs, rtypes, names_rTypes)


# In[16]:


cmat_fp_MG2 = utilsFunctions.evaluateModel(result_rf_fp_MG2, testFps_fp_MG2, testActs, rtypes, names_rTypes)


# In[17]:


cmat_fp_TT = utilsFunctions.evaluateModel(result_rf_fp_TT, testFps_fp_TT, testActs, rtypes, names_rTypes)


# In[18]:


utilsFunctions.labelled_cmat(cmat_fp_AP3,rtypes,figsize=(16,12), labelExtras=names_rTypes)


# In[19]:


utilsFunctions.labelled_cmat(cmat_fp_MG2,rtypes,figsize=(16,12), labelExtras=names_rTypes)


# In[20]:


utilsFunctions.labelled_cmat(cmat_fp_TT,rtypes,figsize=(16,12), labelExtras=names_rTypes)

