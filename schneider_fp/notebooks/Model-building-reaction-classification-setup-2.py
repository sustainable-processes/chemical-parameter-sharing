#!/usr/bin/env python
# coding: utf-8

# ### Learning reaction types using a Random Forest classifier and more local difference fingerprints (AP3, MG2, TT)

# In[1]:


import cPickle,gzip
from collections import defaultdict
import random
from sklearn.ensemble import RandomForestClassifier
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
fpsz=2048 # the FPs bit size for converting the FPs to a numpy array
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


# Build the RF classifiers with max tree depth of 25

# In[6]:


print 'training models'
rf_cls_AP3 = RandomForestClassifier(n_estimators=200, max_depth=25,random_state=23,n_jobs=1)
result_rf_fp_AP3 = rf_cls_AP3.fit(trainFps_fp_AP3,trainActs)
rf_cls_MG2 = RandomForestClassifier(n_estimators=200, max_depth=25,random_state=23,n_jobs=1)
result_rf_fp_MG2 = rf_cls_MG2.fit(trainFps_fp_MG2,trainActs)
rf_cls_TT = RandomForestClassifier(n_estimators=200, max_depth=25,random_state=23,n_jobs=1)
result_rf_fp_TT = rf_cls_TT.fit(trainFps_fp_TT,trainActs)


# Evaluate the RF classifier using our test data

# In[7]:


cmat_fp_AP3 = utilsFunctions.evaluateModel(result_rf_fp_AP3, testFps_fp_AP3, testActs, rtypes, names_rTypes)


# Draw the confusion matix

# In[8]:


utilsFunctions.labelled_cmat(cmat_fp_AP3,rtypes,figsize=(16,12), labelExtras=names_rTypes)


# In[9]:


cmat_fp_MG2 = utilsFunctions.evaluateModel(result_rf_fp_MG2, testFps_fp_MG2, testActs, rtypes, names_rTypes)


# In[10]:


utilsFunctions.labelled_cmat(cmat_fp_MG2,rtypes,figsize=(16,12),labelExtras=names_rTypes)


# In[11]:


cmat_fp_TT = utilsFunctions.evaluateModel(result_rf_fp_TT, testFps_fp_TT, testActs, rtypes, names_rTypes)


# In[12]:


utilsFunctions.labelled_cmat(cmat_fp_TT,rtypes,figsize=(16,12),labelExtras=names_rTypes)

