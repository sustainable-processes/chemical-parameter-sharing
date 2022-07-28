#!/usr/bin/env python
# coding: utf-8

# ### Apply the reaction classification model for recovering unclassified reactions from the patent data

# In[1]:


import cPickle,gzip
from collections import defaultdict
import random
from sklearn.linear_model import LogisticRegression
import utilsFunctions
from sklearn.externals import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
from base64 import b64encode
import types
from rdkit.six.moves import cStringIO as StringIO
from IPython.core.display import display,HTML
import pandas as pd


# Some functions for drawing the reactions

# In[2]:


def removeMappingNumbersFromSmiles(smi):
    new_smi=''
    i=0
    while i < len(smi):
        if smi[i] == ':':     
            i += 1
            j = i
            while smi[j] != ']':
                if smi[j].isdigit():
                    i+=1
                j+=1
            new_smi += smi[i]
            i+= 1
            continue
        else:
            new_smi += smi[i]
        i += 1
    return new_smi

def reactionToHTMLImage(rxn):
    x = Draw.ReactionToImage(rxn)
    """displayhook function for PIL Images, rendered as PNG"""
    sio = StringIO()
    x.save(sio,format='PNG')
    s = b64encode(sio.getvalue())
    pd.set_option('display.max_columns',len(s)+1000)
    pd.set_option('display.max_rows',len(s)+1000)
    pd.set_option("display.max_colwidth",len(s)+10000000000)
    return '<img src="data:image/png;base64,%s" alt="Mol"/>'%s

Chem.rdChemReactions.ChemicalReaction.__str__=reactionToHTMLImage

def AddReactionColumnToFrame(frame, smilesCol='smi', reactCol = 'Reaction'):
    frame[reactCol]=frame.apply(lambda x: AllChem.ReactionFromSmarts(x[smilesCol], useSmiles=True), axis=1)
    PandasTools.RenderImagesInAllDataFrames(images=True)


# In[3]:


rcParams.update({'font.size': 14})


# In[4]:


dataDir = "../data/"
reaction_types = cPickle.load(file(dataDir+"reactionTypes_training_test_set_patent_data.pkl"))
names_rTypes = cPickle.load(file(dataDir+"names_rTypes_classes_superclasses_training_test_set_patent_data.pkl"))


# Load the AP3 and agent feature fingerprint for the 50000 unclassified reactions

# In[5]:


infile = gzip.open(dataDir+"transformationFPs_agentFPs_external_test_set_B.pkl.gz", 'rb')

lineNo=0
fps=[]
smis=[]

fpsz=256
while 1:
    lineNo+=1
    try:
        lbl,smi,apfp_woA,agent_feature = cPickle.load(infile)        
    except EOFError:
        break
    np1 = utilsFunctions.fpToNPfloat(apfp_woA,fpsz)
    np2 = np.asarray(agent_feature, dtype=float)
    smis.append((smi, lbl))
    fps += [np.concatenate([np1, np2])]
    if not lineNo%10000:
        print "Done "+str(lineNo)


# Load the reaction type, reaction class and super-class model

# In[6]:


clf = joblib.load(dataDir+'LR_transformationFP256bit.AP3.agent_featureFP.pkl')
clf1 = joblib.load(dataDir+'LR_transformationFP256bit.AP3.agent_featureFP_classModel.pkl')
clf2 = joblib.load(dataDir+'LR_transformationFP256bit.AP3.agent_featureFP_superclassModel.pkl')


# Predict the reaction types of the unclassified reactions and plot the distribution of the 50 reaction types

# In[7]:


rtypes = sorted(list(reaction_types))
preds = clf.predict(fps)
preds_proba = clf.predict_proba(fps)
newPreds=[rtypes[x] for x in preds]
preds2 = defaultdict(int)
for i in newPreds:
    preds2[i]+=1
preds2_sorted = sorted(preds2.items(), key=lambda x:(int(x[0].split('.')[0]), int(x[0].split('.')[1]), int(x[0].split('.')[2])))    
labels = [x[0] for x in preds2_sorted]
values = [x[1] for x in preds2_sorted]
ind = np.arange(len(values))

figure(1,figsize=(20,5),dpi=300)
subplot(1,1,1)    
_=bar(ind, values,color="#1a62a6")
_=xticks(np.arange(len(labels))+.5,labels,rotation='vertical')
_=title('Reaction type prediction')


# Select the most likely reaction type per reaction

# In[8]:


probs = [(rtypes[x.argmax()], names_rTypes[rtypes[x.argmax()]], x.max()) for x in preds_proba]


# Only keep those with a high probability (>95%) for one of the 50 reaction types. Plot the distribution of the remaining reactions again.

# In[9]:


count = 0
preds = defaultdict(int)
best_reactions={}
for n,i in enumerate(probs):
    if i[2] > 0.95:
        preds[i[0]]+=1
        if i[0] in best_reactions:
            best_reactions[i[0]].append((i[2],n,i[0],i[1]))
        else:
            best_reactions[i[0]]=[(i[2],n,i[0],i[1])]
    else:
        count+=1

preds_sorted = sorted(preds.items(), key=lambda x:(int(x[0].split('.')[0]), int(x[0].split('.')[1]), int(x[0].split('.')[2])))    
labels = [x[0] for x in preds_sorted]
values = [x[1] for x in preds_sorted]
ind = np.arange(len(values))
maxi = np.array(values).max()
figure(1,figsize=(20,5),dpi=300)
subplot(1,1,1)    
_=bar(ind, values,color="#1a62a6")
_=xticks(np.arange(len(labels))+.5,labels,rotation='vertical')
_=title('Reaction type prediction (probability > 95 %)')
print count


# Look at the number of different reaction types found in the unclassified reactions

# In[10]:


len(best_reactions)


# Build lists of reaction classes and super-classes

# In[14]:


reaction_classes = sort(list(set(x.split('.')[0]+"."+x.split('.')[1] for x in reaction_types)))
print reaction_classes
reaction_superclasses = sort(list(set(x.split('.')[0] for x in reaction_types)))
print reaction_superclasses


# Build a new subset of the 1414 remaining reactions

# In[15]:


newTestSubsets = defaultdict(list)
idxdict = defaultdict(list)
for i in best_reactions.keys():
    for n,j in enumerate(best_reactions[i]):
        idx = j[1]
        idxdict[i].append(idx)
        newTestSubsets[i].append(fps[idx])


# Classify this subset using the reaction class and super-class predictive model

# In[16]:


res=defaultdict(list)
for i in newTestSubsets.keys():
    for n,fp in enumerate(newTestSubsets[i]):
        idxFP = idxdict[i][n]
        type_prob = best_reactions[i][n][0]
        cls = reaction_classes[np.array(clf1.predict_proba(fp)).argmax()]
        cls_prob = np.array(clf1.predict_proba(fp)).max()
        scls = reaction_superclasses[np.array(clf2.predict_proba(fp)).argmax()]
        scls_prob = np.array(clf2.predict_proba(fp)).max()
        res[i].append([idxFP, type_prob, cls, cls_prob, scls, scls_prob])


# Confirm a consistent reaction type, class and super-class. Additionally, calculate the mean probability of reaction type, class and superclass for each of the reactions.

# In[17]:


best_res = defaultdict(list)
remaining_reactions=0
for i in res.keys():
    for j in res[i]:
        if j[2] == i.split('.')[0]+'.'+i.split('.')[1] and j[4] == i.split('.')[0]:
            mean_prob = (j[1]+j[3]+j[5])/3.0
            best_res[i].append([mean_prob, j])
            remaining_reactions+=1
print remaining_reactions
print len(best_res.keys())


# Sort the remaining reactions by their mean probability in descending order. Only keep the best 5 per class and ignore duplicates (same probability).

# In[18]:


import operator
res_to_classify=defaultdict(list)
numclassify=0
for i in best_res.keys():
    subset = sorted(best_res[i], key=operator.itemgetter(0), reverse=True)
    prob_before = 10
    counter = 0
    for s in subset:
        prob_cur = s[0]
        if prob_cur != prob_before and counter < 5:
            res_to_classify[i].append(s)
            counter+=1
            numclassify+=1
        prob_before=prob_cur
print numclassify


# Prepare the reactions for storing in a Pandas table

# In[19]:


reacts_to_draw=[]
for i in res_to_classify.keys():
    for j in res_to_classify[i]:
        rtype = i
        mean_prob = j[0]
        smi = smis[j[1][0]][0]
        idx = smis[j[1][0]][1]
        #rxn = AllChem.ReactionFromSmarts(smi, useSmiles=True)
        #rxn.RemoveUnmappedReactantTemplates()
        #smi = AllChem.ReactionToSmiles(rxn)
        smi = removeMappingNumbersFromSmiles(smi)
        typename = probs[j[1][0]][1]
        reacts_to_draw.append((rtype, typename, mean_prob,smi,idx))


# Add the reaction and depictions of them to a Pandas table

# In[20]:


data = pd.DataFrame(reacts_to_draw, columns=['class', 'classname', 'prob','smi','patent'])
AddReactionColumnToFrame(data, smilesCol='smi', reactCol = 'Reaction')
data = data.drop('smi',1)


# Display the results in a Pandas table

# In[21]:


display(HTML(data.to_html()))

