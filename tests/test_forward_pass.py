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
def test_tf_teacher_force_model_forward_pass(build_model, get_tf_data, request):
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


@pytest.mark.parametrize(
    "build_model",
    [
        "build_tf_hard_select_model",
        "build_tf_soft_select_model",
        "build_tf_teacher_force_model",
    ],
)
def test_tf_teacher_force_model_train(build_model, get_tf_data, request):
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

    import tensorflow as tf

    model.compile(
        loss=[
            tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            tf.keras.losses.MeanSquaredError(),
        ],
        loss_weights=[1, 1, 1, 1, 1, 1e-4],
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics={
            "c1": [
                "acc",
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
            ],
            "s1": [
                "acc",
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
            ],
            "s2": [
                "acc",
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
            ],
            "r1": [
                "acc",
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
            ],
            "r2": [
                "acc",
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
            ],
        },
    )

    import param_sharing.learn.util

    h = model.fit(
        x=x_train_data
        if train_mode == param_sharing.coley_code.model.TEACHER_FORCE
        else x_train_eval_data,
        y=y_train_data,
        epochs=1,
        verbose=1,
        batch_size=10000,
        validation_data=(
            x_val_data
            if train_mode == param_sharing.coley_code.model.TEACHER_FORCE
            else x_val_eval_data,
            y_val_data,
        ),
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=param_sharing.learn.util.log_dir(
                    prefix="TF_", comment="_MOREDATA_REG_HARDSELECT"
                )
            ),
        ],
    )
