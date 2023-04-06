import pytest



def test_get_rxn_df(get_rxn_df):
    rxn_df = get_rxn_df.copy()
    rxn_df.iloc[0, 0] = "WOWOW this shouldnt be here"


def test_get_rxn_df_immutable(get_rxn_df):
    rxn_df = get_rxn_df.copy()
    assert rxn_df.iloc[0, 0] != "WOWOW this shouldnt be here"


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


def test_get_masked_indexes(get_masked_train_val_indexes):
    train_idx, val_idx = get_masked_train_val_indexes
    assert train_idx.shape[0] == 2095
    assert val_idx.shape[0] == 2096


def test_get_unmasked_indexes(get_unmasked_train_val_indexes):
    train_idx, val_idx = get_unmasked_train_val_indexes
    assert train_idx.shape[0] == 2634
    assert val_idx.shape[0] == 2634


def test_get_torch_data(get_torch_data):
    train_data, val_data = get_torch_data


def test_get_tf_data(get_tf_data):
    (
        _x_train_data,
        _x_train_eval_data,
        _y_train_data,
        _x_val_data,
        _x_val_eval_data,
        _y_val_data,
    ) = get_tf_data


@pytest.mark.parametrize(
    "build_model",
    [
        "build_tf_hard_select_model",
        "build_tf_soft_select_model",
        "build_tf_teacher_force_model",
    ],
)
def test_tf_teacher_force_model(build_model, get_tf_data, request):
    (
        x_train_data,
        x_train_eval_data,
        y_train_data,
        x_val_data,
        x_val_eval_data,
        y_val_data,
    ) = get_tf_data

    model, train_mode = request.getfixturevalue(build_model)

    import param_sharing.coley_code.model

    cat_pred = model(
        x_train_data
        if train_mode == param_sharing.coley_code.model.TEACHER_FORCE
        else x_train_eval_data
    )[0]
