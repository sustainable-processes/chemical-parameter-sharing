#%%
from rdkit import Chem
# %%
test_data = ['ReactionSmiles\tPatentNumber\tParagraphNum\tYear\tTextMinedYield\tCalculatedYield\n',
 '[Br:1][CH2:2][CH2:3][OH:4].[CH2:5]([S:7](Cl)(=[O:9])=[O:8])[CH3:6].CCOCC>C(N(CC)CC)C>[CH2:5]([S:7]([O:4][CH2:3][CH2:2][Br:1])(=[O:9])=[O:8])[CH3:6]\tUS03930836\t\t1976\t\t\n',
 '[Br:1][CH2:2][CH2:3][CH2:4][OH:5].[CH3:6][S:7](Cl)(=[O:9])=[O:8].CCOCC>C(N(CC)CC)C>[CH3:6][S:7]([O:5][CH2:4][CH2:3][CH2:2][Br:1])(=[O:9])=[O:8]\tUS03930836\t\t1976\t\t\n',
 '[CH2:1]([Cl:4])[CH2:2][OH:3].CCOCC.[CH2:10]([S:14](Cl)(=[O:16])=[O:15])[CH:11]([CH3:13])[CH3:12]>C(N(CC)CC)C>[CH2:10]([S:14]([O:3][CH2:2][CH2:1][Cl:4])(=[O:16])=[O:15])[CH:11]([CH3:13])[CH3:12]\tUS03930836\t\t1976\t\t\n',
 '[Br:1][CH2:2][CH2:3][OH:4].[CH2:5]([S:7](Cl)(=[O:9])=[O:8])[CH3:6].CCOCC>C(N(CC)CC)C>[CH2:5]([S:7]([O:4][CH2:3][CH2:2][Br:1])(=[O:9])=[O:8])[CH3:6]\tUS03930839\t\t1976\t\t\n',
 '[Br:1][CH2:2][CH2:3][CH2:4][OH:5].[CH3:6][S:7](Cl)(=[O:9])=[O:8].CCOCC>C(N(CC)CC)C>[CH3:6][S:7]([O:5][CH2:4][CH2:3][CH2:2][Br:1])(=[O:9])=[O:8]\tUS03930839\t\t1976\t\t\n',
 '[CH2:1]([Cl:4])[CH2:2][OH:3].CCOCC.[CH2:10]([S:14](Cl)(=[O:16])=[O:15])[CH:11]([CH3:13])[CH3:12]>C(N(CC)CC)C>[CH2:10]([S:14]([O:3][CH2:2][CH2:1][Cl:4])(=[O:16])=[O:15])[CH:11]([CH3:13])[CH3:12]\tUS03930839\t\t1976\t\t\n',
 '[Cl:1][C:2]1[N:3]=[CH:4][C:5]2[C:10]([CH:11]=1)=[C:9]([N+:12]([O-])=O)[CH:8]=[CH:7][CH:6]=2.O.[OH-].[Na+]>C(O)(=O)C.[Fe]>[Cl:1][C:2]1[N:3]=[CH:4][C:5]2[C:10]([CH:11]=1)=[C:9]([NH2:12])[CH:8]=[CH:7][CH:6]=2 |f:2.3|\tUS03930837\t\t1976\t\t\n',
 '[CH3:1][C:2]1[N+:3]([O-])=[CH:4][C:5]2[C:10]([CH:11]=1)=[C:9]([N+:12]([O-:14])=[O:13])[CH:8]=[CH:7][CH:6]=2.P(Cl)(Cl)([Cl:18])=O>>[Cl:18][C:4]1[C:5]2[C:10](=[C:9]([N+:12]([O-:14])=[O:13])[CH:8]=[CH:7][CH:6]=2)[CH:11]=[C:2]([CH3:1])[N:3]=1\tUS03930837\t\t1976\t\t\n',
 '[CH3:1][C:2]1[N:3]=[CH:4][C:5]2[C:10]([CH:11]=1)=[C:9]([N+:12]([O-:14])=[O:13])[CH:8]=[CH:7][CH:6]=2.[ClH:15]>>[ClH:15].[CH3:1][C:2]1[N:3]=[CH:4][C:5]2[C:10]([CH:11]=1)=[C:9]([N+:12]([O-:14])=[O:13])[CH:8]=[CH:7][CH:6]=2 |f:2.3|\tUS03930837\t\t1976\t\t\n',
 'CC1N=CC2C(C=1)=C([N+]([O-])=O)C=CC=2.[Cl:15][C:16]1[C:25]2[C:20](=[CH:21][CH:22]=[CH:23][CH:24]=2)[CH:19]=[CH:18][N:17]=1>>[ClH:15].[Cl:15][C:16]1[C:25]2[C:20](=[CH:21][CH:22]=[CH:23][CH:24]=2)[CH:19]=[CH:18][N:17]=1 |f:2.3|\tUS03930837\t\t1976\t\t\n',
 'CC1N=CC2C(C=1)=C([N+]([O-])=O)C=CC=2.[Cl:15][C:16]1[CH:25]=[CH:24][C:23]([N+:26]([O-:28])=[O:27])=[C:22]2[C:17]=1[CH:18]=[CH:19][N:20]=[CH:21]2.Cl.CC1N=CC2C(C=1)=C([N+]([O-])=O)C=CC=2.[IH:44]>>[IH:44].[Cl:15][C:16]1[CH:25]=[CH:24][C:23]([N+:26]([O-:28])=[O:27])=[C:22]2[C:17]=1[CH:18]=[CH:19][N:20]=[CH:21]2 |f:2.3,5.6|\tUS03930837\t\t1976\t\t\n',
 '[N+:1]([C:4]1[CH:13]=[CH:12][CH:11]=[C:10]2[C:5]=1[CH:6]=[CH:7][N:8]=[CH:9]2)([O-:3])=[O:2].[BrH:14]>C(O)C>[BrH:14].[N+:1]([C:4]1[CH:13]=[CH:12][CH:11]=[C:10]2[C:5]=1[CH:6]=[CH:7][N:8]=[CH:9]2)([O-:3])=[O:2] |f:3.4|\tUS03930837\t\t1976\t\t\n',
 '[N+](C1C=CC=C2C=1C=CN=C2)([O-])=O.[CH3:14][C:15]1[C:24]2[C:19](=[CH:20][CH:21]=[CH:22][CH:23]=2)[CH:18]=[CH:17][N:16]=1.Br.[Cl:26][C:27]1[C:32]([OH:33])=[C:31]([Cl:34])[C:30]([Cl:35])=[C:29]([Cl:36])[C:28]=1[Cl:37]>>[Cl:26][C:27]1[C:32]([O-:33])=[C:31]([Cl:34])[C:30]([Cl:35])=[C:29]([Cl:36])[C:28]=1[Cl:37].[CH3:14][C:15]1[C:24]2[C:19](=[CH:20][CH:21]=[CH:22][CH:23]=2)[CH:18]=[CH:17][NH+:16]=1 |f:4.5|\tUS03930837\t\t1976\t\t\n',
 '[N+:1]([C:4]1[CH:13]=[CH:12][CH:11]=[C:10]2[C:5]=1[CH:6]=[CH:7][N:8]=[CH:9]2)([O-])=O.NC1C=CC=C2C=1C=CN=C2.Br.[IH:26]>>[IH:26].[IH:26].[NH2:1][C:4]1[CH:13]=[CH:12][CH:11]=[C:10]2[C:5]=1[CH:6]=[CH:7][N:8]=[CH:9]2 |f:4.5.6|\tUS03930837\t\t1976\t\t\n',
 'Cl.[OH:2][C@@H:3]([CH2:21][CH2:22][CH2:23][CH2:24][CH3:25])[CH:4]=[CH:5][CH:6]1[CH:10]=[CH:9][C:8](=[O:11])[CH:7]1[CH2:12][CH:13]=[CH:14][CH2:15][CH2:16][CH2:17][C:18]([OH:20])=[O:19]>C(O)C>[OH:2][C@@H:3]([CH2:21][CH2:22][CH2:23][CH2:24][CH3:25])[CH:4]=[CH:5][CH:6]1[CH2:10][CH2:9][C:8](=[O:11])[CH:7]1[CH2:12][CH:13]=[CH:14][CH2:15][CH2:16][CH2:17][C:18]([OH:20])=[O:19]\tUS03930952\t\t1976\t\t\n',
 'CC(O[CH2:5][C:6]1[CH2:28][S:27][C@@H:9]2[C@H:10]([NH:13]C(C(OC(C)=O)C3C=CC=CC=3)=O)[C:11](=[O:12])[N:8]2[C:7]=1[C:29]([OH:31])=[O:30])=O>O>[CH3:5][C:6]1[CH2:28][S:27][C@@H:9]2[C@H:10]([NH2:13])[C:11](=[O:12])[N:8]2[C:7]=1[C:29]([OH:31])=[O:30]\tUS03930949\t\t1976\t\t\n',
 '[S:1]([O-:5])([O-:4])(=[O:3])=[O:2].[NH4+:6].[NH4+]>O>[S:1](=[O:3])(=[O:2])([OH:5])[O-:4].[NH4+:6].[S:1]([O-:5])([O-:4])(=[O:3])=[O:2].[NH4+:6].[NH4+:6] |f:0.1.2,4.5,6.7.8|\tUS03930988\t\t1976\t\t\n',
 'CO[C:3]1[CH:4]=[C:5]([C:9]2([CH2:12][C:13]([Cl:16])([Cl:15])[Cl:14])[CH2:11][O:10]2)[CH:6]=[CH:7][CH:8]=1.ClC1C=C(C2(CC(Cl)(Cl)Cl)CO2)C=CC=1.FC1C=C(C2(CC(Cl)(Cl)Cl)CO2)C=CC=1.ClC1C=C(C2(CC(Cl)(Cl)Cl)CO2)C=CC=1Cl.C(OC1C=C(C2(CC(Cl)(Cl)Cl)CO2)C=CC=1)C.C(OC1C=C(C2(CC(Cl)(Cl)Cl)CO2)C=CC=1)C1C=CC=CC=1.ClC1C=CC(C2(CC(Cl)(Cl)Cl)CO2)=CC=1.[Br:117]C1C=CC(C2(CC(Cl)(Cl)Cl)CO2)=CC=1>>[Br:117][C:3]1[CH:4]=[C:5]([C:9]2([CH2:12][C:13]([Cl:16])([Cl:15])[Cl:14])[CH2:11][O:10]2)[CH:6]=[CH:7][CH:8]=1\tUS03930835\t\t1976\t\t\n',
 '[C:1]1(O)[CH:6]=[CH:5][CH:4]=[CH:3][CH:2]=1.[CH2:8]=[O:9].[S:10]([O-:13])([O-:12])=[O:11].[Na+:14].[Na+]>O>[OH:9][CH:8]([S:10]([O-:13])(=[O:12])=[O:11])[C:1]1[CH:6]=[CH:5][CH:4]=[CH:3][CH:2]=1.[Na+:14] |f:2.3.4,6.7|\tUS03931083\t\t1976\t\t\n']
# %%
test_data[1][:147]

# %%
Chem.MolToSmiles(Chem.MolFromSmiles('[Br:1][CH2:2][CH2:3][OH:4].[CH2:5]([S:7](Cl)(=[O:9])=[O:8])[CH3:6]'))
# %%
mol = Chem.MolFromSmiles('[Br:1][CH2:2][CH2:3][OH:4].[CH2:5]([S:7](Cl)(=[O:9])=[O:8])[CH3:6]') 
Chem.RDKFingerprint(mol)
# %%
def unscramble_rxn_smi(rxn_smi):
    """
    Input: reaction smiles string with separators . and >
    Return: 
    """