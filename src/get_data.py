# %%
import pandas as pd
import pathlib

dataDir = "../data/"


def read_uspto_data(
    file_loc=pathlib.Path(dataDir+"1976_Sep2016_USPTOgrants_smiles.rsmi")
):
    with open(file_loc,'r') as f:
        uspto = f.readlines()

    uspto = [s.replace("\n", "").split("\t") for s in uspto]
    uspto = pd.DataFrame(uspto[1:], columns=uspto[0]).set_index("PatentNumber")
    return uspto
# %%

import xmltodict

file = pathlib.Path(dataDir) / "grants" / "1976" / "pftaps19760106_wk01.xml"
with open(file, "r") as f:
    file_data = f.read()


file_dict = xmltodict.parse(file_data)
reactions = file_dict['reactionList']['reaction']


# %%

def parse_product(product):
    if isinstance(product, list):
        return [parse_product(i) for i in product]
    else:
        # parse dict
        # print(product)
        name = product.get("molecule", {}).get('dl:nameResolved', None)
        print("name1",name)
        if not name:
            name = product.get("molecule", {}).get('name', {}).get('#text', None)
        print("name2",name)

        return {
            # 'name': product['molecule']['dl:nameResolved'],
            'id': [i['@value'] for i in product.get('identifier', []) if i['@dictRef'] == 'cml:smiles'],
            'state': product.get('dl:state', None),
            'appearance': product.get('dl:appearance', None),
        }

def parse_reactant(reactant):
    if isinstance(reactant, list):
        return [parse_reactant(i) for i in reactant]
    else:
        # parse dict
        return reactant


def parse_spectator(spectator):
    if isinstance(spectator, list):
        return [parse_spectator(i) for i in spectator]
    else:
        # parse dict
        return spectator


def parse_reactionAction(reactionAction):
    if isinstance(reactionAction, list):
        return [parse_reactionAction(i) for i in reactionAction]
    else:
        # parse dict
        return reactionAction


# %%


for r in reactions:
    source = r.get('dl:source', {})
    documentId, headingText, paragraphText = None, None, None
    if source:
        documentId = source.get('dl:documentId')
        headingText = source.get('dl:headingText')
        paragraphText = source.get('dl:paragraphText')
    
    reactionSmiles = r.get('dl:reactionSmiles', '')
    productList = r.get('productList', {})
    product = {}
    if productList:
        product = productList.get('product')
    reactantList = r.get('reactantList', {})
    reactant = {}
    if reactantList:
        reactant = reactantList.get('reactant')
    spectatorList = r.get('spectatorList', {})
    spectator = {}
    if spectatorList:
        spectator = spectatorList.get('spectator')
    reactionActionList = r.get('dl:reactionActionList', {})
    reactionAction = {}
    if reactionActionList:
        reactionAction = reactionActionList.get('dl:reactionAction')
    print(f"reactionSmiles {reactionSmiles[:10]} \t product {type(product)}, {len(product)} \t reactant {type(reactant)}, {len(reactant)} \t spectator {type(spectator)}, {len(spectator)} \t reactionAction {type(reactionAction)}, {len(reactionAction)}")

    print(product)
    product = parse_product(product)
    print(product)

    print("=======================")
    

    

    # if isinstance(product, list):
    #     print(type(product), len(product))
    #     for i in product:
    #         print(i)
    #     break
    # else:
    #     print(type(product), len(product), product)
        

    # if isinstance(reactant, dict):
    #     print(type(reactant), len(reactant), reactant)
    #     break
    # else:
    #     print(type(reactant), len(reactant), reactant)

    # if isinstance(spectator, list):
    #     print(spectator)

    # if isinstance(reactionAction, dict):
    #     print(reactionAction)




# %%

# %%
