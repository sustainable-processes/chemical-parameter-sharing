#!/usr/bin/env python
# coding: utf-8

# ### Learning reaction types using different ML methods (K-Means, LR) and more local difference fingerprints (AP3, MG2, TT)

# Goal: find the best/simplest/most appropriate ML model for our FPs

# In[1]:


import cPickle,gzip
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
trainFps_fp_AP3=defaultdict(list)
trainFps_fp_MG2=defaultdict(list)
trainFps_fp_TT=defaultdict(list)
trainActs=[]
testFps_fp_AP3=defaultdict(list)
testFps_fp_MG2=defaultdict(list)
testFps_fp_TT=defaultdict(list)
testActs=[]

print 'building fp collection'

rtypes=sorted(list(reaction_types))
for i,klass in enumerate(rtypes):
    actIds = [x for x in indices if fps[x][2]==klass]
    for x in actIds[:nActive]:
        trainFps_fp_AP3[klass] += [utilsFunctions.fpToNP(fps[x][3],fpsz)]
        trainFps_fp_MG2[klass] += [utilsFunctions.fpToNP(fps[x][4],fpsz)]
        trainFps_fp_TT[klass] += [utilsFunctions.fpToNP(fps[x][5],fpsz)]
    trainActs += [i]*nActive
    nTest=len(actIds)-nActive
    for x in actIds[nActive:]:
        testFps_fp_AP3[klass] += [utilsFunctions.fpToNP(fps[x][3],fpsz)]
        testFps_fp_MG2[klass] += [utilsFunctions.fpToNP(fps[x][4],fpsz)]
        testFps_fp_TT[klass] += [utilsFunctions.fpToNP(fps[x][5],fpsz)]
    testActs += [i]*nTest


# Build the K-Means (k=3) clusters for each of the 50 reaction types and store them in a map. Here AP3 FPs were used.

# In[6]:


print 'training model'
clusters_per_rType_AP3 = {}
for cls in trainFps_fp_AP3:
    cl = KMeans(n_clusters=3)
    res = cl.fit(trainFps_fp_AP3[cls])
    clusters_per_rType_AP3[cls] = res


# Evaluate the K-Means cluster using a kind of nearest neighbor approach

# In[7]:


cmat_KM_fp_AP3 = utilsFunctions.evaluateKMeansClustering(clusters_per_rType_AP3, testFps_fp_AP3, rtypes, names_rTypes)


# Draw the confusion matrix

# In[8]:


utilsFunctions.labelled_cmat(cmat_KM_fp_AP3,rtypes,figsize=(16,12), labelExtras=names_rTypes, xlabel=False)


# Build the K-Means (k=3) clusters for each of the 50 reaction types and store them in a map. Here MG2 FPs were used.

# In[9]:


print 'training model'
clusters_per_rType_MG2 = {}
for cls in trainFps_fp_MG2:
    cl = KMeans(n_clusters=3)
    res = cl.fit(trainFps_fp_MG2[cls])
    clusters_per_rType_MG2[cls] = res


# In[10]:


cmat_KM_fp_MG2 = utilsFunctions.evaluateKMeansClustering(clusters_per_rType_MG2, testFps_fp_MG2, rtypes, names_rTypes)


# In[11]:


utilsFunctions.labelled_cmat(cmat_KM_fp_MG2,rtypes,figsize=(16,12), labelExtras=names_rTypes)


# Build the K-Means (k=3) clusters for each of the 50 reaction types and store them in a map. Here TT FPs were used.

# In[12]:


print 'training model'
clusters_per_rType_TT = {}
for cls in trainFps_fp_TT:
    cl = KMeans(n_clusters=3)
    res = cl.fit(trainFps_fp_TT[cls])
    clusters_per_rType_TT[cls] = res


# In[13]:


cmat_KM_fp_TT = utilsFunctions.evaluateKMeansClustering(clusters_per_rType_TT, testFps_fp_TT, rtypes, names_rTypes)


# In[14]:


utilsFunctions.labelled_cmat(cmat_KM_fp_TT,rtypes,figsize=(16,12), labelExtras=names_rTypes)


# #### Test the Logistic regression classifier with all three FPs types (AP3, MG2, TT)

# In[15]:


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


# Build the LR classifier without the need of parameters

# In[16]:


print 'training models'
lr_cls_AP3 = LogisticRegression()
result_lr_fp_AP3 = lr_cls_AP3.fit(trainFps_fp_AP3,trainActs)
lr_cls_MG2 = LogisticRegression()
result_lr_fp_MG2 = lr_cls_MG2.fit(trainFps_fp_MG2,trainActs)
lr_cls_TT = LogisticRegression()
result_lr_fp_TT = lr_cls_TT.fit(trainFps_fp_TT,trainActs)


# Evalutate the LR classifier using our test data

# In[17]:


cmat_fp_AP3 = utilsFunctions.evaluateModel(result_lr_fp_AP3, testFps_fp_AP3, testActs, rtypes, names_rTypes)


# In[19]:


utilsFunctions.labelled_cmat(cmat_fp_AP3,rtypes,figsize=(16,12), labelExtras=names_rTypes)


# In[20]:


cmat_fp_MG2 = utilsFunctions.evaluateModel(result_lr_fp_MG2, testFps_fp_MG2, testActs, rtypes, names_rTypes)


# In[21]:


utilsFunctions.labelled_cmat(cmat_fp_MG2,rtypes,figsize=(16,12),labelExtras=names_rTypes)


# In[22]:


cmat_fp_TT = utilsFunctions.evaluateModel(result_lr_fp_TT, testFps_fp_TT, testActs, rtypes, names_rTypes)


# In[23]:


utilsFunctions.labelled_cmat(cmat_fp_TT,rtypes,figsize=(16,12),labelExtras=names_rTypes)

