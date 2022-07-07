# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from pylab import *
from rdkit import DataStructs
from sklearn import metrics
from sklearn.cluster import KMeans
from collections import defaultdict

# <codecell>

# convert a SparseIntvect fingerprint into a hashed numpy vector
def fpToNP(fp,fpsz):
    nfp = np.zeros((fpsz,),np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = hash(str(idx))%fpsz
        nfp[nidx]+=v
    return nfp

# convert a SparseIntvect into a hashed numpy float vector
def fpToNPfloat(fp,fpsz):
    nfp = np.zeros((fpsz,),np.float)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = hash(str(idx))%fpsz
        nfp[nidx]+=float(v)
    return nfp

# convert a hashed SparseIntvect fingerprint into a numpy vector
def hashedFPToNP(fp,fpsz):
    nfp = np.zeros((fpsz,),np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nfp[idx]+=v
    return nfp

# convert a hashed SparseIntvect into a numpy float vector
def hashedFPToNPfloat(fp,fpsz):
    nfp = np.zeros((fpsz,),np.float)
    for idx,v in fp.GetNonzeroElements().items():
        nfp[idx]+=float(v)
    return nfp

# convert a dictionary-based fingerprint into a numpy vector
import operator
def fpDictToNP(fp_dict):
    sortDict = sorted(fp_dict.items(), key=operator.itemgetter(0)) 
    nfp = np.zeros((len(fp_dict.keys()),),np.int32)
    count=0
    for v in sortDict:
        nfp[count]+=v[1]
        count+=1
    return nfp

# hash a SparseIntvect
def hashFP(fp,fpsz):
    hashed_fp = DataStructs.UIntSparseIntVect(fpsz)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = hash(str(idx))%fpsz
        hashed_fp.__setitem__(nidx,v)
    return hashed_fp

# convert a SparseIntvect (difference fingerprint) into its positive form
def removeNegativeCountsFP(fp):
    fpsz = fp.GetLength()
    fpn = DataStructs.UIntSparseIntVect(fpsz*2)
    for idx,v in fp.GetNonzeroElements().items():
        if v < 0:
            fpn.__setitem__(idx+fpsz,abs(v))
        else:
            fpn.__setitem__(idx,abs(v))
    return fpn

from sklearn.feature_extraction import DictVectorizer

# use the DictVectorizer to get an unfolded version of the fingerprints
def prepareUnfoldedData(fplist):
    v = DictVectorizer(sparse=False)
    temp =[x.GetNonzeroElements() for x in fplist]
    X = v.fit(temp)
    return X

# convert a fingerprint into a dictionary with float values
def fpToFloatDict(fp):
    res = defaultdict(float)
    for idx,v in fp.GetNonzeroElements().items():
        res[idx]+=float(v)
    return res

# convert a list into a dictionary with float values
def listToFloatDict(fp):
    res = defaultdict(float)
    for i,v in enumerate(fp):
        res[i]=float(v)
    return res

# merge two dictionaries into one containing float values
def mergeDicts(fp1, fp2):
    res = defaultdict(float)
    for i,v in fp1.items():
        res[i]=float(v)
    for i,v in fp2.items():
        res[i]=float(v)
    return res

# merge two fingerprints into one dictionary containing float values
def mergeFps2Dict(fp1, fp2):
    res = defaultdict(float)
    for i,v in fp1.GetNonzeroElements().items():
        res[i]=v
    for i,v in fp2.GetNonzeroElements().items():
        res[i]=v
    return res

# <codecell>

def labelled_cmat(cmat,labels,figsize=(20,15),labelExtras=None, dpi=300,threshold=0.01, xlabel=True, ylabel=True, rotation=90):
    
    rowCounts = np.array(sum(cmat,1),dtype=float)
    cmat_percent = cmat/rowCounts[:,None]
    #zero all elements that are less than 1% of the row contents
    ncm = cmat_percent*(cmat_percent>threshold)

    fig = figure(1,figsize=figsize,dpi=dpi)
    ax = fig.add_subplot(1,1,1)
    fig.set_size_inches(figsize)
    fig.set_dpi(dpi)
    pax=ax.pcolor(ncm,cmap=cm.ocean_r)
    ax.set_frame_on(True)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(cmat.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(cmat.shape[1])+0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    if labelExtras is not None:
        labels = [' %s %s'%(x,labelExtras[x].strip()) for x in labels]
    
    ax.set_xticklabels([], minor=False) 
    ax.set_yticklabels([], minor=False)

    if xlabel:
        ax.set_xticklabels(labels, minor=False, rotation=rotation, horizontalalignment='left') 
    if ylabel:
        ax.set_yticklabels(labels, minor=False)

    ax.grid(True)
    fig.colorbar(pax)
    axis('tight')

# <codecell>

# evaluate model calculating recall, precision and F-score, return the confusion matrix
def evaluateModel(model, testFPs, testReactionTypes, rTypes, names_rTypes):
    
    preds = model.predict(testFPs)
    newPreds=[rTypes[x] for x in preds]
    newTestActs=[rTypes[x] for x in testReactionTypes]
    cmat=metrics.confusion_matrix(newTestActs,newPreds)
    cmat=metrics.confusion_matrix(testReactionTypes,preds)
    colCounts = sum(cmat,0)
    rowCounts = sum(cmat,1)

    print '%2s %7s %7s %7s     %s'%("ID","recall","prec","F-score ","reaction class")
    sum_recall=0
    sum_prec=0
    for i,klass in enumerate(rTypes):
        recall = 0
        if rowCounts[i] > 0:
            recall = float(cmat[i,i])/rowCounts[i]
        sum_recall += recall
        prec = 0
        if colCounts[i] > 0:
            prec = float(cmat[i,i])/colCounts[i]
        sum_prec += prec
        f_score = 0
        if (recall + prec) > 0:
            f_score = 2 * (recall * prec) / (recall + prec)   
        print '%2d % .4f % .4f % .4f % 9s %s'%(i,recall,prec,f_score,klass,names_rTypes[klass])
    
    mean_recall = sum_recall/len(rTypes)
    mean_prec = sum_prec/len(rTypes)
    if (mean_recall+mean_prec) > 0:
        mean_fscore = 2*(mean_recall*mean_prec)/(mean_recall+mean_prec)
    print "Mean:% 3.2f % 7.2f % 7.2f"%(mean_recall,mean_prec,mean_fscore)
    
    return cmat


# evaluate K-Means model calculating recall, precision and F-score, return the confusion matrix
def evaluateKMeansClustering(clusters_per_rType, testFPs, rTypes, names_rTypes):
    newPreds=[]
    newTestActs=[]
    for cls in testFPs:
        for fp in testFPs[cls]:
            min_dist = 10000
            pred_cls = ""
            for cls_train in clusters_per_rType:
                dist = clusters_per_rType[cls_train].transform(fp).min()
                if min_dist > dist:
                    min_dist = dist
                    pred_cls = cls_train
            newPreds.append(pred_cls)
            newTestActs.append(cls)

    cmat=metrics.confusion_matrix(newTestActs,newPreds)
    colCounts = sum(cmat,0)
    rowCounts = sum(cmat,1)

    print '%2s %7s %7s %7s     %s'%("ID","recall","prec","F-score ","reaction class")
    sum_recall=0
    sum_prec=0
    for i,klass in enumerate(rTypes):
        recall = 0
        if rowCounts[i] > 0:
            recall = float(cmat[i,i])/rowCounts[i]
        sum_recall += recall
        prec = 0
        if colCounts[i] > 0:
            prec = float(cmat[i,i])/colCounts[i]
        sum_prec += prec
        f_score = 0
        if (recall + prec) > 0:
            f_score = 2 * (recall * prec) / (recall + prec)   
        print '%2d % .4f % .4f % .4f % 9s %s'%(i,recall,prec,f_score,klass,names_rTypes[klass])
    
    mean_recall = sum_recall/len(rTypes)
    mean_prec = sum_prec/len(rTypes)
    if (mean_recall+mean_prec) > 0:
        mean_fscore = 2*(mean_recall*mean_prec)/(mean_recall+mean_prec)
    print "Mean:% 3.2f % 7.2f % 7.2f"%(mean_recall,mean_prec,mean_fscore)
    
    return cmat

# <codecell>

# input purities list of the Butina clustering, number of members dict per reation type, class or super-class
# level: super-class = 0, class = 1, reation type = 2 
def evaluatePurityClusters(purities, members_per_cls, level):
    res_dict={}
    for i in purities:
        label=i[3][level]
        if label in res_dict:
            nofCluster = res_dict[label][0]+1
            mean_purity = (res_dict[label][1]*(nofCluster-1)+i[2][level])/nofCluster
            clustersize = res_dict[label][2]+i[1]
            nof_members_cls = res_dict[label][3] + i[4][level]
            ratio_members_found = nof_members_cls/float(members_per_cls[label])
            f_score=2*(mean_purity*ratio_members_found)/(mean_purity+ratio_members_found)
            res_dict[label] = [nofCluster, mean_purity, clustersize, nof_members_cls, ratio_members_found, f_score]
        else:
            nofCluster = 1
            mean_purity = i[2][level]/nofCluster
            clustersize = i[1]
            nof_members_cls = i[4][level]
            ratio_members_found = nof_members_cls/float(members_per_cls[label])
            f_score=2*(mean_purity*ratio_members_found)/(mean_purity+ratio_members_found)
            res_dict[label] = [nofCluster, mean_purity, clustersize, nof_members_cls, ratio_members_found, f_score]

    fscores = np.array([x[1][-1] for x in res_dict.items()])
    mean_fscore = fscores.mean()
    median_fscore = np.median(fscores)
    print "Mean F-score: ",mean_fscore
    print "Median F-score: ",median_fscore
    print "Min, Max F-score :", fscores.max(), fscores.min()
    precision = np.array([x[1][1] for x in res_dict.items()])
    mean_precision = precision.mean()
    median_precision = np.median(precision)
    print "Mean precision: ",mean_precision
    print "Median precision: ",median_precision
    print "Min, Max precision :", precision.max(), precision.min()
    recall = np.array([x[1][-2] for x in res_dict.items()])
    mean_recall = recall.mean()
    median_recall = np.median(recall)
    print "Mean recall: ",mean_recall
    print "Median recall: ",median_recall
    print "Min, Max recall :", recall.max(), recall.min()
    
    return res_dict

