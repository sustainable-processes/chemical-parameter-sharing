import pytest


@pytest.fixture(scope="session")
def get_torch_data(rxn_with_mask, get_masked_train_val_indexes):
    tmp_train_idx, tmp_val_idx = get_masked_train_val_indexes
    train_idx = tmp_train_idx.copy()
    val_idx = tmp_val_idx.copy()

    tmp_rxn_df, tmp_rxn_fp, tmp_product_fp = rxn_with_mask
    rxn_df = tmp_rxn_df.copy()
    rxn_fp = tmp_rxn_fp.copy()
    product_fp = tmp_product_fp.copy()

    import torch

    train_product_fp = torch.Tensor(product_fp[train_idx])
    train_rxn_diff_fp = torch.Tensor(rxn_fp[train_idx])
    val_product_fp = torch.Tensor(product_fp[val_idx])
    val_rxn_diff_fp = torch.Tensor(rxn_fp[val_idx])

    import param_sharing.learn.ohe

    train_catalyst, val_catalyst, cat_enc = param_sharing.learn.ohe.apply_train_ohe_fit(
        rxn_df[["catalyst_0"]].fillna("NULL"),
        train_idx,
        val_idx,
        tensor_func=torch.Tensor,
    )
    (
        train_solvent_0,
        val_solvent_0,
        sol0_enc,
    ) = param_sharing.learn.ohe.apply_train_ohe_fit(
        rxn_df[["solvent_0"]].fillna("NULL"),
        train_idx,
        val_idx,
        tensor_func=torch.Tensor,
    )
    (
        train_solvent_1,
        val_solvent_1,
        sol1_enc,
    ) = param_sharing.learn.ohe.apply_train_ohe_fit(
        rxn_df[["solvent_1"]].fillna("NULL"),
        train_idx,
        val_idx,
        tensor_func=torch.Tensor,
    )
    (
        train_reagents_0,
        val_reagents_0,
        reag0_enc,
    ) = param_sharing.learn.ohe.apply_train_ohe_fit(
        rxn_df[["reagents_0"]].fillna("NULL"),
        train_idx,
        val_idx,
        tensor_func=torch.Tensor,
    )
    (
        train_reagents_1,
        val_reagents_1,
        reag1_enc,
    ) = param_sharing.learn.ohe.apply_train_ohe_fit(
        rxn_df[["reagents_1"]].fillna("NULL"),
        train_idx,
        val_idx,
        tensor_func=torch.Tensor,
    )
    (
        train_temperature,
        val_temperature,
        temp_enc,
    ) = param_sharing.learn.ohe.apply_train_ohe_fit(
        rxn_df[["temperature_0"]].fillna(-1),
        train_idx,
        val_idx,
        tensor_func=torch.Tensor,
    )

    cut_off = None
    train_data = {
        "product_fp": train_product_fp[:cut_off],
        "rxn_diff_fp": train_rxn_diff_fp[:cut_off],
        "catalyst": train_catalyst[:cut_off],
        "solvent_1": train_solvent_0[:cut_off],
        "solvent_2": train_solvent_1[:cut_off],
        "reagents_1": train_reagents_0[:cut_off],
        "reagents_2": train_reagents_1[:cut_off],
        "temperature": train_temperature[:cut_off],
    }

    cut_off = None
    val_data = {
        "product_fp": val_product_fp[:cut_off],
        "rxn_diff_fp": val_rxn_diff_fp[:cut_off],
        "catalyst": val_catalyst[:cut_off],
        "solvent_1": val_solvent_0[:cut_off],
        "solvent_2": val_solvent_1[:cut_off],
        "reagents_1": val_reagents_0[:cut_off],
        "reagents_2": val_reagents_1[:cut_off],
        "temperature": val_temperature[:cut_off],
    }
    return train_data, val_data
