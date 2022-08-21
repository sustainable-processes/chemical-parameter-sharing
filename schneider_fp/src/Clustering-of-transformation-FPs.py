#!/usr/bin/env python
# coding: utf-8

# ### Clustering of the AP3 fingerprints to evaluate the applicability for similarity search 

# In[1]:


import _pickle as cPickle
import gzip,time
from collections import defaultdict
import random
from rdkit.ML.Cluster import Butina
from rdkit import DataStructs
import utilsFunctions


# Choose some larger text size in the plots

# In[2]:


# rcParams.update({'font.size': 14})


# In[3]:


dataDir = "../data/"
with open(dataDir+"reactionTypes_training_test_set_patent_data.pkl", 'rb') as f:
    ff = f.read().replace(b'\r\n', b'\n')
    reaction_types = cPickle.loads(ff)


with open(dataDir+"names_rTypes_classes_superclasses_training_test_set_patent_data.pkl", 'rb') as f:
    ff = f.read().replace(b'\r\n', b'\n')
    names_rTypes = cPickle.loads(ff)

# Load the transformation FPs 

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
    fps.append([idx,lbl,cls,fp_AP3])
    idx+=1
    if not lineNo%10000:
        print("Done "+str(lineNo))


# Build a subset of the fingerprints (training set) (10000 FPs) for efficiency reasons.

# In[5]:


random.seed(0xd00f)
indices=range(len(fps))
random.shuffle(indices)

nActive=200
fpsz= 256
fpsubset = []

rtypes=sorted(list(reaction_types))
for i,klass in enumerate(rtypes):
    actIds = [x for x in indices if fps[x][2]==klass]
    for x in actIds[:nActive]:
        trfp_AP3 = utilsFunctions.hashFP(fps[x][3],fpsz)
        trfp_AP3 = utilsFunctions.removeNegativeCountsFP(trfp_AP3)
        fpsubset.append((trfp_AP3,fps[x][2]))


# Cluster this subset using the Butina algorithm from the RDKit with the Dice coefficent as similiarty metric and a threshold of 0.5 for similar FPs.

# In[6]:


t1=time.time()
tf=[x[0] for x in fpsubset]
cs = Butina.ClusterData(tf,len(tf),0.5,distFunc=lambda x,y:1.-DataStructs.DiceSimilarity(x,y))
t2=time.time()
print '%.2f'%(t2-t1)
print len(cs)
hist([len(x) for x in cs],bins=20,log=True)


# Determine the purity of the clusters with at least ten members. The purity of the cluster is calculated by determining the main reaction type, class and super-class and normalizing this quantity by the size of the cluster.

# In[7]:


import operator
purities=[]
nAccountedFor=0
for (i,c) in enumerate(cs):
    sz=len(c)
    if sz<=10:
        continue
    nAccountedFor+=sz
    tcounts1=defaultdict(int)
    tcounts2=defaultdict(int)
    tcounts3=defaultdict(int)
    for idx in c:
        lbl = fpsubset[idx][1]
        slbl = lbl.split(' ')[0].split('.')
        tcounts1[slbl[0]]+=1
        tcounts2['.'.join(slbl[:2])]+=1
        tcounts3['.'.join(slbl[:3])]+=1
        
    tcounts1_sorted = sorted(tcounts1.items(), key=operator.itemgetter(1), reverse=True) 
    maxc1 = tcounts1_sorted[0][1]
    maxlbl1 = tcounts1_sorted[0][0]
    tcounts2_sorted = sorted(tcounts2.items(), key=operator.itemgetter(1), reverse=True) 
    maxc2 = tcounts2_sorted[0][1]
    maxlbl2 = tcounts2_sorted[0][0]
    tcounts3_sorted = sorted(tcounts3.items(), key=operator.itemgetter(1), reverse=True) 
    maxc3 = tcounts3_sorted[0][1]
    maxlbl3 = tcounts3_sorted[0][0]
    purities.append((i,sz,(1.*maxc1/sz,1.*maxc2/sz,1.*maxc3/sz),(maxlbl1, maxlbl2, maxlbl3),(maxc1,maxc2,maxc3)))
print len(purities),nAccountedFor


# Determine the number of members per reaction type, class and super-class in the FP subset.

# In[8]:


members_rtype=defaultdict(int)
members_class=defaultdict(int)
members_superclass=defaultdict(int)
for i in fpsubset:
    members_rtype[i[1]]+=1
    members_class[i[1].split('.')[0]+'.'+i[1].split('.')[1]]+=1
    members_superclass[i[1].split('.')[0]]+=1


# ##### Calculate F-score, precision and recall for the reaction type purity of the clusters.

# The mean purity per reaction type can be regarded as a kind of precision value. We were also interested in the recall, the number of reactions per reaction type which could be actually recovered within the clusters. Hence, the recall is calculated by the number of recovered reactions divided by number of reactions of that type contained in the data. Using the recall and precision the F-score of the clustering for each reaction type can be determined.

# Merge the clusters with the same main reaction type and calculate based on this the mean F-score, precision and recall.

# In[9]:


# input purities list of the Butina clustering, number of members dict per reation type, class or super-class
# level: super-class = 0, class = 1, reation type = 2 
rtype_purity_dict = utilsFunctions.evaluatePurityClusters(purities, members_rtype, 2)


# Sort the final clusters based on the number of clusters they were build of in ascending order.

# In[10]:


rtype_purity_sorted = sorted(rtype_purity_dict.items(), key=operator.itemgetter(1))


# Plot the distribution of the different reaction type clusters considering the number of clusters they were build of and their F-score. 

# In[11]:


labels = [x[0] for x in rtype_purity_sorted]
yvalues1 = [x[1][0] for x in rtype_purity_sorted]
colors = [x[1][5] for x in rtype_purity_sorted]

xvalues = np.arange(len(rtype_purity_sorted))
colors.append(1.0) # to a scale from 0.0. to 1.0
colors.append(0.0) # to a scale from 0.0. to 1.0

width = 0.5
fig=figure(1,figsize(20,5),dpi=200)
ax1=subplot(1,1,1)
ax1.bar(xvalues, yvalues1,width,color=cm.Blues(colors))
xticks(xvalues+width/2., labels,rotation='vertical')
ylabel('Number of clusters')
xlabel('Reaction types')
sm = cm.ScalarMappable(cmap=cm.Blues)
sm.set_array(colors)
cb = colorbar(sm)
cb.set_label('F-score')


# Merge the clusters with the same main reaction class and calculate based on this the mean F-score, precision and recall.

# In[12]:


# input purities list of the Butina clustering, number of members dict per reation type, class or super-class
# level: super-class = 0, class = 1, reation type = 2 
class_purity_dict = utilsFunctions.evaluatePurityClusters(purities, members_class, 1)


# Sort the final clusters based on their mean F-score in ascending order.

# In[13]:


class_purity_sorted = sorted(class_purity_dict.items(), key=lambda x: x[1][5])


# Plot the distribution of the different reaction class clusters their mean F-score. 

# In[14]:


labels = [x[0] for x in class_purity_sorted]
yvalues1 = [x[1][5] for x in class_purity_sorted]
colors = [x[1][5] for x in class_purity_sorted]
xvalues = np.arange(len(class_purity_sorted))
colors.append(1.0) # to a scale from 0.0. to 1.0
colors.append(0.0) # to a scale from 0.0. to 1.0

width = 0.5
fig=figure(1,figsize(20,5),dpi=200)
ax1=subplot(1,1,1)
ax1.bar(xvalues, yvalues1,width)#,color=cm.Blues(colors))
xticks(xvalues+width/2., labels,rotation='vertical')
ylabel('F-score')
_=xlabel('Reaction classes')


# Merge the clusters with the same main reaction super-class and calculate based on this the mean F-score, precision and recall.

# In[15]:


# input purities list of the Butina clustering, number of members dict per reation type, class or super-class
# level: super-class = 0, class = 1, reation type = 2 
superclass_purity_dict = utilsFunctions.evaluatePurityClusters(purities, members_superclass, 0)


# Sort the final clusters based on their mean F-score in ascending order.

# In[16]:


superclass_purity_sorted = sorted(superclass_purity_dict.items(), key=lambda x: x[1][5])


# Plot the distribution of the different reaction super-class clusters their mean F-score. 

# In[17]:


labels = [x[0] for x in superclass_purity_sorted]
yvalues1 = [x[1][5] for x in superclass_purity_sorted]
colors = [x[1][5] for x in superclass_purity_sorted]

xvalues = np.arange(len(superclass_purity_sorted))
colors.append(1.0) # to a scale from 0.0. to 1.0
colors.append(0.0) # to a scale from 0.0. to 1.0

width = 0.5
fig=figure(1,figsize(20,5),dpi=200)
ax1=subplot(1,1,1)
ax1.bar(xvalues, yvalues1,width)#,color=cm.Blues(colors))
xticks(xvalues+width/2., labels,rotation='vertical')
ylabel('F-score')
_=xlabel('Reaction super-classes')

