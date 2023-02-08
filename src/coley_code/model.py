import tensorflow as tf

TEACHER_FORCE = 0
HARD_SELECTION = 1
SOFT_SELECTION = 2


def hard_selection(pred):
    return tf.stop_gradient(
        tf.one_hot(
            tf.math.argmax(pred, axis=1), pred.shape[1]
        )
    )


def build_teacher_forcing_model(
	pfp_len=2048, rxnfp_len=2048, c1_dim=100, s1_dim=100, s2_dim=100, r1_dim=100, r2_dim=100, N_h1=1024, N_h2=100, l2v=0, mode=TEACHER_FORCE
) -> tf.keras.models.Model:
    input_pfp = tf.keras.layers.Input(shape = (pfp_len,), name = 'input_pfp')
    input_rxnfp = tf.keras.layers.Input(shape = (rxnfp_len,), name = 'input_rxnfp')

    if mode == TEACHER_FORCE:
        input_c1 = tf.keras.layers.Input(shape = (c1_dim,), name = 'input_c1')
        input_s1 = tf.keras.layers.Input(shape = (s1_dim,), name = 'input_s1')
        input_s2 = tf.keras.layers.Input(shape = (s2_dim,), name = 'input_s2')
        input_r1 = tf.keras.layers.Input(shape = (r1_dim,), name = 'input_r1')
        input_r2 = tf.keras.layers.Input(shape = (r2_dim,), name = 'input_r2')
    elif mode == HARD_SELECTION or mode == SOFT_SELECTION:
        pass
    else:
        raise NotImplementedError(f"unknown for {mode=}")

    concat_fp = tf.keras.layers.Concatenate(axis = 1)([input_pfp,input_rxnfp])

    h1 = tf.keras.layers.Dense(1000, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(l2v),name = 'fp_transform1')(concat_fp)
    # h1_dropout = Dropout(0.3)(h1)
    h2 = tf.keras.layers.Dense(1000, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(l2v),name = 'fp_transform2')(h1)
    h2_dropout = tf.keras.layers.Dropout(0.5)(h2,training=False)
    # h1 = concat_fp

    c1_h1 = tf.keras.layers.Dense(N_h1, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(l2v), name = 'c1_h1')(h2_dropout)
    c1_h2 = tf.keras.layers.Dense(N_h1, activation = 'tanh', kernel_regularizer = tf.keras.regularizers.l2(l2v), name = 'c1_h2')(c1_h1)
    # c1_h2_dropout = Dropout(0.3)(c1_h2)
    c1_output = tf.keras.layers.Dense(c1_dim, activation = "softmax",name = "c1")(c1_h2)
    if mode == HARD_SELECTION:
        input_c1 = hard_selection(c1_output)
    elif mode == SOFT_SELECTION:
        input_c1 = c1_output
    elif mode == TEACHER_FORCE:
        pass
    else:
        raise NotImplementedError(f"unknown for {mode=}")
    c1_dense = tf.keras.layers.Dense(N_h2, activation = 'relu',name = 'c1_dense')(input_c1)

    concat_fp_c1 = tf.keras.layers.Concatenate(axis = -1,name = "concat_fp_c1")([h2_dropout,c1_dense])

    s1_h1 = tf.keras.layers.Dense(N_h1, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(l2v), name = "s1_h1")(concat_fp_c1)
    s1_h2 = tf.keras.layers.Dense(N_h1, activation = 'tanh', kernel_regularizer = tf.keras.regularizers.l2(l2v), name = "s1_h2")(s1_h1)
    # s1_h2_dropout = Dropout(0.3)(s1_h2)
    s1_output = tf.keras.layers.Dense(s1_dim, activation = "softmax", name = "s1")(s1_h2)
    if mode == HARD_SELECTION:
        input_s1 = hard_selection(s1_output)
    elif mode == SOFT_SELECTION:
        input_s1 = s1_output
    elif mode == TEACHER_FORCE:
        pass
    else:
        raise NotImplementedError(f"unknown for {mode=}")
    s1_dense = tf.keras.layers.Dense(N_h2, activation = 'relu',name = 's1_dense')(input_s1)
    # rgt_output = Lambda(lambda x: x / K.sum(x, axis=-1),output_shape = (rgt_dim,))(rgt_unscaled)

    concat_fp_c1_s1 = tf.keras.layers.Concatenate(axis = -1,name = "concat_fp_c1_s1")([h2_dropout,c1_dense,s1_dense])

    s2_h1 = tf.keras.layers.Dense(N_h1, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(l2v), name = "s2_h1")(concat_fp_c1_s1)
    s2_h2 = tf.keras.layers.Dense(N_h1, activation = 'tanh', kernel_regularizer = tf.keras.regularizers.l2(l2v), name = "s2_h2")(s2_h1)
    # s2_h2_dropout = Dropout(0.3)(s2_h2)
    s2_output = tf.keras.layers.Dense(s2_dim, activation = "softmax", name = "s2")(s2_h2)    
    if mode == HARD_SELECTION:
        input_s2 = hard_selection(s2_output)
    elif mode == SOFT_SELECTION:
        input_s2 = s2_output
    elif mode == TEACHER_FORCE:
        pass
    else:
        raise NotImplementedError(f"unknown for {mode=}")
    s2_dense = tf.keras.layers.Dense(N_h2, activation = 'relu',name = 's2_dense')(input_s2)

    concat_fp_c1_s1_s2 = tf.keras.layers.Concatenate(axis = -1,name = "concat_fp_c1_s1_s2")([h2_dropout,c1_dense, s1_dense, s2_dense])

    r1_h1 = tf.keras.layers.Dense(N_h1, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(l2v), name = "r1_h1")(concat_fp_c1_s1_s2)
    r1_h2 = tf.keras.layers.Dense(N_h1, activation = 'tanh', kernel_regularizer = tf.keras.regularizers.l2(l2v), name = "r1_h2")(r1_h1)
    # r1_h2_dropout = Dropout(0.3)(r1_h2)
    r1_output = tf.keras.layers.Dense(r1_dim, activation = "softmax", name = "r1")(r1_h2)
    if mode == HARD_SELECTION:
        input_r1 = hard_selection(r1_output)
    elif mode == SOFT_SELECTION:
        input_r1 = r1_output
    elif mode == TEACHER_FORCE:
        pass
    else:
        raise NotImplementedError(f"unknown for {mode=}")
    r1_dense = tf.keras.layers.Dense(N_h2, activation = 'relu',name = 'r1_dense')(input_r1)
    # rgt_output = Lambda(lambda x: x / K.sum(x, axis=-1),output_shape = (rgt_dim,))(rgt_unscaled)

    concat_fp_c1_s1_s2_r1 = tf.keras.layers.Concatenate(axis = -1,name = "concat_fp_c1_s1_s2_r1")([h2_dropout,c1_dense,s1_dense,s2_dense,r1_dense])

    r2_h1 = tf.keras.layers.Dense(N_h1, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(l2v), name = "r2_h1")(concat_fp_c1_s1_s2_r1)
    r2_h2 = tf.keras.layers.Dense(N_h1, activation = 'tanh', kernel_regularizer = tf.keras.regularizers.l2(l2v), name = "r2_h2")(r2_h1)
    # r2_h2_dropout = Dropout(0.3)(r2_h2)
    r2_output = tf.keras.layers.Dense(r2_dim, activation = "softmax", name = "r2")(r2_h2)
    if mode == HARD_SELECTION:
        input_r2 = hard_selection(r2_output)
    elif mode == SOFT_SELECTION:
        input_r2 = r2_output
    elif mode == TEACHER_FORCE:
        pass
    else:
        raise NotImplementedError(f"unknown for {mode=}")
    r2_dense = tf.keras.layers.Dense(N_h2, activation = 'relu',name = 'r2_dense')(input_r2)    

    concat_fp_c1_s1_s2_r1_r2 = tf.keras.layers.Concatenate(axis = -1,name = "concat_fp_c1_s1_s2_r1_r2")([h2_dropout,c1_dense,s1_dense,s2_dense,r1_dense,r2_dense])

    T_h1 = tf.keras.layers.Dense(N_h1, activation = 'relu', name = "T_h1")(concat_fp_c1_s1_s2_r1_r2)
    # T_h1_dropout = Dropout(0.3)(T_h1)
    # T_h2 = keras.layers.Dense(N_h1, activation = 'relu', kernel_regularizer = l2(l2v), name = "T_h2")(T_h1)
    T_output = tf.keras.layers.Dense(1, activation = "linear", name = "T")(T_h1)    
    #just for the purpose of shorter print message
    c1 = c1_output
    s1 = s1_output
    s2 = s2_output
    r1 = r1_output
    r2 = r2_output
    Temp = T_output
    output = [c1,s1,s2,r1,r2,Temp]
    if mode == TEACHER_FORCE:
        model = tf.keras.models.Model(
            [input_pfp, input_rxnfp, input_c1, input_s1, input_s2, input_r1, input_r2],
            output
        )
    elif mode == HARD_SELECTION or mode == SOFT_SELECTION:
        model = tf.keras.models.Model(
            [input_pfp, input_rxnfp],
            output
        )   
    else:
        raise NotImplementedError(f"unknown for {mode=}")

    model.count_params()
    model.summary()
    return model


def update_teacher_forcing_model_weights(update_model, to_copy_model):
    layers = [
        'fp_transform1', 'fp_transform2', 'c1_dense', 's1_dense', 
        's2_dense', 'r1_dense', 'r2_dense', 'c1_h1', 's1_h1', 's2_h1', 
        'r1_h1', 'r2_h1', 'c1_h2', 's1_h2', 's2_h2', 'r1_h2', 'r2_h2', 
        'T_h1', 'c1', 's1', 's2', 'r1', 'r2', 'T'
    ]

    for l in layers:
        update_model.get_layer(l).set_weights(
            to_copy_model.get_layer(l).get_weights()
        )