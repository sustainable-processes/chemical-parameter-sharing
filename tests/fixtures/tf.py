import pytest


@pytest.fixture(scope="session")
def get_tf_data(rxn_with_mask, get_masked_train_val_indexes):
    tmp_train_idx, tmp_val_idx = get_masked_train_val_indexes
    train_idx = tmp_train_idx.copy()
    val_idx = tmp_val_idx.copy()

    tmp_rxn_df, tmp_rxn_fp, tmp_product_fp = rxn_with_mask
    rxn_df = tmp_rxn_df.copy()
    rxn_fp = tmp_rxn_fp.copy()
    product_fp = tmp_product_fp.copy()

    import tensorflow as tf

    train_product_fp = tf.convert_to_tensor(product_fp[train_idx])
    train_rxn_diff_fp = tf.convert_to_tensor(rxn_fp[train_idx])
    val_product_fp = tf.convert_to_tensor(product_fp[val_idx])
    val_rxn_diff_fp = tf.convert_to_tensor(rxn_fp[val_idx])

    import param_sharing.learn.ohe

    train_catalyst, val_catalyst, cat_enc = param_sharing.learn.ohe.apply_train_ohe_fit(
        rxn_df[["catalyst_0"]].fillna("NULL"),
        train_idx,
        val_idx,
        tensor_func=tf.convert_to_tensor,
    )
    (
        train_solvent_0,
        val_solvent_0,
        sol0_enc,
    ) = param_sharing.learn.ohe.apply_train_ohe_fit(
        rxn_df[["solvent_0"]].fillna("NULL"),
        train_idx,
        val_idx,
        tensor_func=tf.convert_to_tensor,
    )
    (
        train_solvent_1,
        val_solvent_1,
        sol1_enc,
    ) = param_sharing.learn.ohe.apply_train_ohe_fit(
        rxn_df[["solvent_1"]].fillna("NULL"),
        train_idx,
        val_idx,
        tensor_func=tf.convert_to_tensor,
    )
    (
        train_reagents_0,
        val_reagents_0,
        reag0_enc,
    ) = param_sharing.learn.ohe.apply_train_ohe_fit(
        rxn_df[["reagents_0"]].fillna("NULL"),
        train_idx,
        val_idx,
        tensor_func=tf.convert_to_tensor,
    )
    (
        train_reagents_1,
        val_reagents_1,
        reag1_enc,
    ) = param_sharing.learn.ohe.apply_train_ohe_fit(
        rxn_df[["reagents_1"]].fillna("NULL"),
        train_idx,
        val_idx,
        tensor_func=tf.convert_to_tensor,
    )
    train_temperature = tf.convert_to_tensor(
        rxn_df["temperature_0"].iloc[train_idx].fillna(-1.0)
    )
    val_temperature = tf.convert_to_tensor(
        rxn_df["temperature_0"].iloc[val_idx].fillna(-1.0)
    )

    x_train_data = (
        train_product_fp,
        train_rxn_diff_fp,
        train_catalyst,
        train_solvent_0,
        train_solvent_1,
        train_reagents_0,
        train_reagents_1,
    )

    x_train_eval_data = (
        train_product_fp,
        train_rxn_diff_fp,
    )

    y_train_data = (
        train_catalyst,
        train_solvent_0,
        train_solvent_1,
        train_reagents_0,
        train_reagents_1,
        train_temperature,
    )

    x_val_data = (
        val_product_fp,
        val_rxn_diff_fp,
        val_catalyst,
        val_solvent_0,
        val_solvent_1,
        val_reagents_0,
        val_reagents_1,
    )

    x_val_eval_data = (
        val_product_fp,
        val_rxn_diff_fp,
    )

    y_val_data = (
        val_catalyst,
        val_solvent_0,
        val_solvent_1,
        val_reagents_0,
        val_reagents_1,
        val_temperature,
    )

    return (
        x_train_data,
        x_train_eval_data,
        y_train_data,
        x_val_data,
        x_val_eval_data,
        y_val_data,
    )


def build_tf_models(mode, x_train_data):
    (
        train_product_fp,
        train_rxn_diff_fp,
        train_catalyst,
        train_solvent_0,
        train_solvent_1,
        train_reagents_0,
        train_reagents_1,
    ) = x_train_data

    import param_sharing.coley_code.model

    model = param_sharing.coley_code.model.build_teacher_forcing_model(
        pfp_len=train_product_fp.shape[-1],
        rxnfp_len=train_rxn_diff_fp.shape[-1],
        c1_dim=train_catalyst.shape[-1],
        s1_dim=train_solvent_0.shape[-1],
        s2_dim=train_solvent_1.shape[-1],
        r1_dim=train_reagents_0.shape[-1],
        r2_dim=train_reagents_1.shape[-1],
        N_h1=1024,
        N_h2=100,
        l2v=0,
        mode=mode,
        dropout_prob=0.2,
        use_batchnorm=True,
    )

    return model


@pytest.fixture(scope="session")
def build_tf_hard_select_model(get_tf_data):
    (
        _x_train_data,
        _,
        _,
        _,
        _,
        _,
    ) = get_tf_data

    import param_sharing.coley_code.model

    mode = param_sharing.coley_code.model.HARD_SELECTION
    return build_tf_models(mode=mode, x_train_data=_x_train_data), mode


@pytest.fixture(scope="session")
def build_tf_soft_select_model(get_tf_data):
    (
        _x_train_data,
        _,
        _,
        _,
        _,
        _,
    ) = get_tf_data

    import param_sharing.coley_code.model

    mode = param_sharing.coley_code.model.SOFT_SELECTION
    return build_tf_models(mode=mode, x_train_data=_x_train_data), mode


@pytest.fixture(scope="session")
def build_tf_teacher_force_model(get_tf_data):
    (
        _x_train_data,
        _,
        _,
        _,
        _,
        _,
    ) = get_tf_data

    import param_sharing.coley_code.model

    mode = param_sharing.coley_code.model.TEACHER_FORCE
    return build_tf_models(mode=mode, x_train_data=_x_train_data), mode
