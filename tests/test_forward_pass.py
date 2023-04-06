import pytest


@pytest.fixture(scope="session")
def get_rxn_df():
    import pathlib
    import param_sharing.reactions.get

    df = param_sharing.reactions.get.get_reaction_df(
        cleaned_data_path=pathlib.Path("data/ORD_USPTO/cleaned_data.pkl"),
        rxn_classes_path=pathlib.Path("data/ORD_USPTO/classified_rxn.smi"),
        verbose=True,
    )
    return df


def test_get_rxn_df(get_rxn_df):
    rxn_df = get_rxn_df.copy()
    rxn_df.iloc[0, 0] = "WOWOW this shouldnt be here"


def test_get_rxn_df_immutable(get_rxn_df):
    rxn_df = get_rxn_df.copy()
    assert rxn_df.iloc[0, 0] != "WOWOW this shouldnt be here"


@pytest.fixture(scope="session")
def load_rxn_fp():
    import numpy as np

    return np.load("data/ORD_USPTO/USPTO_rxn_diff_fp.pkl.npy", allow_pickle=True)


@pytest.fixture(scope="session")
def load_product_fp():
    import numpy as np

    return np.load("data/ORD_USPTO/USPTO_product_fp.pkl.npy", allow_pickle=True)


@pytest.fixture(scope="session")
def get_classified_rxn_mask(get_rxn_df):
    rxn_df = get_rxn_df.copy()

    import param_sharing.reactions.filters

    return param_sharing.reactions.filters.get_classified_rxn_data_mask(rxn_df)


@pytest.fixture(scope="session")
def rxn_with_mask(get_rxn_df, load_rxn_fp, load_product_fp, get_classified_rxn_mask):
    rxn_df = get_rxn_df.copy()
    rxn_fp = load_rxn_fp.copy()
    product_fp = load_product_fp.copy()
    classified_rxn_mask = get_classified_rxn_mask.copy()

    assert rxn_df.shape[0] == classified_rxn_mask.shape[0]
    assert rxn_df.shape[0] == rxn_fp.shape[0]
    assert rxn_df.shape[0] == product_fp.shape[0]

    if True:
        rxn_df = rxn_df[classified_rxn_mask]
        rxn_fp = rxn_fp[classified_rxn_mask]
        product_fp = product_fp[classified_rxn_mask]
        print("  - Applied mask")

    assert rxn_df.shape[0] == rxn_fp.shape[0]
    assert rxn_df.shape[0] == product_fp.shape[0]

    return rxn_df, rxn_fp, product_fp


@pytest.fixture(scope="session")
def rxn_without_mask(get_rxn_df, load_rxn_fp, load_product_fp):
    rxn_df = get_rxn_df.copy()
    rxn_fp = load_rxn_fp.copy()
    product_fp = load_product_fp.copy()

    assert rxn_df.shape[0] == rxn_fp.shape[0]
    assert rxn_df.shape[0] == product_fp.shape[0]

    return rxn_df, rxn_fp, product_fp


def test_load_rxn_with_mask(rxn_with_mask):
    tmp_rxn_df, tmp_rxn_fp, tmp_product_fp = rxn_with_mask
    rxn_df, rxn_fp, product_fp = (
        tmp_rxn_df.copy(),
        tmp_rxn_fp.copy(),
        tmp_product_fp.copy(),
    )
    # testing the pipeline runs without error


def test_load_rxn_without_mask(rxn_without_mask):
    tmp_rxn_df, tmp_rxn_fp, tmp_product_fp = rxn_without_mask
    rxn_df, rxn_fp, product_fp = (
        tmp_rxn_df.copy(),
        tmp_rxn_fp.copy(),
        tmp_product_fp.copy(),
    )
    # testing the pipeline runs without error


@pytest.fixture(scope="session")
def get_masked_train_val_indexes(rxn_with_mask):
    tmp_rxn_df, _, _ = rxn_with_mask
    rxn_df = tmp_rxn_df.copy()

    import numpy as np

    rng = np.random.default_rng(12345)

    indexes = np.arange(rxn_df.shape[0])
    rng.shuffle(indexes)

    train_test_split = 0.01  # only use a very small subset for training / val
    train_val_split = 0.5  # 50:50 split between train val, data is small so ok

    test_idx = indexes[int(rxn_df.shape[0] * train_test_split) :]
    train_val_idx = indexes[: int(rxn_df.shape[0] * train_test_split)]
    train_idx = train_val_idx[: int(train_val_idx.shape[0] * train_val_split)]
    val_idx = train_val_idx[int(train_val_idx.shape[0] * train_val_split) :]

    return train_idx, val_idx


@pytest.fixture(scope="session")
def get_unmasked_train_val_indexes(rxn_without_mask):
    tmp_rxn_df, _, _ = rxn_without_mask
    rxn_df = tmp_rxn_df.copy()

    import numpy as np

    rng = np.random.default_rng(12345)

    indexes = np.arange(rxn_df.shape[0])
    rng.shuffle(indexes)

    train_test_split = 0.01  # only use a very small subset for training / val
    train_val_split = 0.5  # 50:50 split between train val, data is small so ok

    test_idx = indexes[int(rxn_df.shape[0] * train_test_split) :]
    train_val_idx = indexes[: int(rxn_df.shape[0] * train_test_split)]
    train_idx = train_val_idx[: int(train_val_idx.shape[0] * train_val_split)]
    val_idx = train_val_idx[int(train_val_idx.shape[0] * train_val_split) :]

    return train_idx, val_idx


def test_get_masked_indexes(get_masked_train_val_indexes):
    train_idx, val_idx = get_masked_train_val_indexes
    assert train_idx.shape[0] == 2095
    assert val_idx.shape[0] == 2096


def test_get_unmasked_indexes(get_unmasked_train_val_indexes):
    train_idx, val_idx = get_unmasked_train_val_indexes
    assert train_idx.shape[0] == 2634
    assert val_idx.shape[0] == 2634
