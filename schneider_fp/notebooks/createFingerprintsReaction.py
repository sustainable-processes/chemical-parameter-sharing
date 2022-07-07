# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from rdkit import Chem
from rdkit.Chem import AllChem
import cPickle,gzip
from collections import defaultdict
import random
from rdkit.Chem import Descriptors
from rdkit import DataStructs

# <codecell>

def create_transformation_FP(rxn, fptype):
    rxn.RemoveUnmappedReactantTemplates()
    rfp = None
    for react in range(rxn.GetNumReactantTemplates()):
        mol = rxn.GetReactantTemplate(react)
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol)
        try:
            if fptype == AllChem.FingerprintType.AtomPairFP:
                fp = AllChem.GetAtomPairFingerprint(mol=mol,maxLength=3)
            elif fptype == AllChem.FingerprintType.MorganFP:
                fp = AllChem.GetMorganFingerprint(mol=mol,radius=2)   
            elif fptype == AllChem.FingerprintType.TopologicalTorsion:
                fp = AllChem.GetTopologicalTorsionFingerprint(mol=mol)
            else:
                print "Unsupported fp type"
        except:
            print "cannot build reactant fp"
        if rfp is None:
            rfp = fp
        else:
            rfp += fp
    pfp = None
    for product in range(rxn.GetNumProductTemplates()):
        mol = rxn.GetProductTemplate(product)
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol)
        try:
            if fptype == AllChem.FingerprintType.AtomPairFP:
                fp = AllChem.GetAtomPairFingerprint(mol=mol,maxLength=3)
            elif fptype == AllChem.FingerprintType.MorganFP:
                fp = AllChem.GetMorganFingerprint(mol=mol,radius=2)   
            elif fptype == AllChem.FingerprintType.TopologicalTorsion:
                fp = AllChem.GetTopologicalTorsionFingerprint(mol=mol)
            else:
                print "Unsupported fp type"
        except:
            print "cannot build product fp"
        if pfp is None:
            pfp = fp
        else:
            pfp += fp
    if pfp is not None and rfp is not None:
        pfp -= rfp
    return pfp

# <codecell>

def create_agent_feature_FP(rxn):    
    rxn.RemoveUnmappedReactantTemplates()
    agent_feature_Fp = [0.0]*9
    for nra in range(rxn.GetNumAgentTemplates()):
        mol = rxn.GetAgentTemplate(nra)
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol)
        try:
            ri = mol.GetRingInfo()
            agent_feature_Fp[0] += Descriptors.MolWt(mol)
            agent_feature_Fp[1] += mol.GetNumAtoms()
            agent_feature_Fp[2] += ri.NumRings()
            agent_feature_Fp[3] += Descriptors.MolLogP(mol)
            agent_feature_Fp[4] += Descriptors.NumRadicalElectrons(mol)
            agent_feature_Fp[5] += Descriptors.TPSA(mol)
            agent_feature_Fp[6] += Descriptors.NumHeteroatoms(mol)
            agent_feature_Fp[7] += Descriptors.NumHAcceptors(mol)
            agent_feature_Fp[8] += Descriptors.NumHDonors(mol)
        except:
            print "Cannot build agent Fp\n"
    return agent_feature_Fp

# <codecell>

def create_agent_morgan2_FP(rxn):    
    rxn.RemoveUnmappedReactantTemplates()
    morgan2 = None
    for nra in range(rxn.GetNumAgentTemplates()):
        mol = rxn.GetAgentTemplate(nra)
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol)
        try:
            mg2 = AllChem.GetMorganFingerprint(mol,radius=2)
            if morgan2 is None and mg2 is not None:
                morgan2= mg2
            elif mg2 is not None:
                morgan2 += mg2
        except:
            print "Cannot build agent Fp\n"
    return morgan2

# <codecell>

agent_dict_list = ['CCN(CC)CC', '[BH3-]C#N', '[Na+]', 'O=S(Cl)Cl', 'O=S(=O)(O)O', 'CN1CCCC1=O', 'C1COCCO1', 'c1cc[nH+]cc1', 'c1ccncc1', 'CCN(C(C)C)C(C)C',\
 'CCO', 'ClC(Cl)(Cl)Cl', 'CC(C)(C)[O-]', 'O=C([O-])O', 'COCCOC', 'BrB(Br)Br', 'CC(=O)O[BH-](OC(C)=O)OC(C)=O', '[NH4+]', '[F-]', 'O=C([O-])[O-]',\
 'CS(C)=O', 'c1ccc(P(c2ccccc2)c2ccccc2)cc1', '[H-]', '[Na]', 'O=C(Cl)C(=O)Cl', '[Cu]I', '[Al+3]', 'c1ccccc1', 'O=C(/C=C/c1ccccc1)/C=C/c1ccccc1',\
 '[Cs+]', '[K+]', '[OH-]', 'CCCCCC', 'CN(C)C=O', 'Cc1ccccc1', 'O=C(OOC(=O)c1ccccc1)c1ccccc1', 'CI', 'B', 'CO', '[I-]', 'O=C(O)C(F)(F)F', \
 'O=P([O-])([O-])[O-]', 'CCOC(C)=O', 'c1ccc(P(c2ccccc2)(c2ccccc2)[Pd](P(c2ccccc2)(c2ccccc2)c2ccccc2)(P(c2ccccc2)(c2ccccc2)c2ccccc2)P(c2ccccc2)(c2ccccc2)c2ccccc2)cc1',\
 'O', 'N', 'C=O', '[BH4-]', 'CC(=O)O', 'CCOCC', 'CC(C)O', 'Cl[Pd]Cl', 'CC(C)=O', 'CN(C)c1ccncc1', 'Cl', 'ClCCCl', '[Br-]', 'ClC(Cl)Cl', \
 '[Li+]', '[Pd]', '[H][H]', '[Cl-]', 'CC(C)(C#N)N=NC(C)(C)C#N', 'CSC', '[Pd+2]', 'CC(=O)[O-]', 'C', 'CCCC[N+](CCCC)(CCCC)CCCC', 'ClCCl', \
 'CC#N', 'C1CCOC1', 'Br']

def create_agent_dictionary_FP(rxn):
    rxn.RemoveUnmappedReactantTemplates()
    agent_dict_fp = dict.fromkeys(agent_dict_list, 0)
    for nra in range(rxn.GetNumAgentTemplates()):
        mol = rxn.GetAgentTemplate(nra)
        mol.UpdatePropertyCache()
        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        if smi in agent_dict_fp:
            agent_dict_fp[smi]+=1
    return agent_dict_fp


