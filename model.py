import typing
import torch


class SimpleMLP(torch.nn.Module):

    def __init__(
        self, 
        *, 
        input_dim: int, 
        hidden_dims: typing.List[int], 
        output_dim: int,
        hidden_acts: typing.List[torch.nn.Module], 
        output_act, 
        use_batchnorm, 
        dropout_prob
    ):
        super(SimpleMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        if not isinstance(hidden_acts, list):
            hidden_acts = [hidden_acts] * len(hidden_dims)
        for dim, hidden_act in zip(hidden_dims, hidden_acts):
            layers.append(torch.nn.Linear(prev_dim, dim))
            layers.append(hidden_act())
            if use_batchnorm:
                layers.append(torch.nn.BatchNorm1d(dim))
            if dropout_prob > 0:
                layers.append(torch.nn.Dropout(p=dropout_prob))
            prev_dim = dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        if output_act is torch.nn.Softmax:
            layers.append(output_act(dim=1))
        else:
            layers.append(output_act())
        self.layers = torch.nn.Sequential(*layers)


    def forward(self, x, training=True):
        if training:
            self.train()
        else:
            self.eval()
        return self.layers(x)


class ColeyUpstreamModel(torch.nn.Module):

    def __init__(self, *, input_dim, hidden_dims, output_dim, hidden_acts=torch.nn.ReLU, output_act=torch.nn.ReLU) -> None:
        super(ColeyUpstreamModel, self).__init__()
        self._mlp = SimpleMLP(
            input_dim=input_dim, 
            hidden_dims=hidden_dims, 
            output_dim=output_dim,
            hidden_acts=hidden_acts, 
            output_act=output_act, 
            use_batchnorm=False, 
            dropout_prob=0.0
        )


    def forward(self, x, training=True):
        if training:
            self.train()
        else:
            self.eval()
        return self._mlp(x, training=training)


class ColeyDownstreamModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, hidden_acts, output_act=torch.nn.Identity) -> None:
        super(ColeyDownstreamModel, self).__init__() 
        self._mlp = SimpleMLP(
            input_dim=input_dim, 
            hidden_dims=hidden_dims, 
            output_dim=output_dim,
            hidden_acts=hidden_acts, 
            output_act=output_act, 
            use_batchnorm=False, 
            dropout_prob=0.0
        )


    def forward(self, x, training=True):
        if training:
            self.train()
        else:
            self.eval()
        return self._mlp(x, training=training)


class ColeyModel(torch.nn.Module):
    def __init__(
        self,
        product_fp_dim,
        rxn_diff_fp_dim,
        cat_dim,
        sol1_dim,
        sol2_dim,
        reag1_dim,
        reag2_dim,
        temp_dim,
    ) -> None:
        super(ColeyModel, self).__init__()
        self.product_fp_dim = product_fp_dim
        self.rxn_diff_fp_dim = rxn_diff_fp_dim
        self.cat_dim = cat_dim
        self.sol1_dim = sol1_dim
        self.sol2_dim = sol2_dim
        self.reag1_dim = reag1_dim
        self.reag2_dim = reag2_dim
        self.temp_dim = temp_dim

        highway_dim = 0

        output_dim = 1000
        self.cat_upstream = ColeyUpstreamModel(input_dim=product_fp_dim+rxn_diff_fp_dim, hidden_dims=[1000], output_dim=output_dim)
        self.cat_downstream = ColeyDownstreamModel(input_dim=output_dim, hidden_dims=[300, 300], output_dim=cat_dim, hidden_acts=[torch.nn.ReLU, torch.nn.Tanh], output_act=torch.nn.Softmax)
        highway_dim += output_dim

        output_dim = 100
        self.sol1_upstream = ColeyUpstreamModel(input_dim=cat_dim, hidden_dims=[], output_dim=output_dim)
        self.sol1_downstream = ColeyDownstreamModel(input_dim=highway_dim+output_dim, hidden_dims=[300, 300], output_dim=sol1_dim, hidden_acts=[torch.nn.ReLU, torch.nn.Tanh], output_act=torch.nn.Softmax)
        highway_dim += output_dim

        output_dim = 100
        self.sol2_upstream = ColeyUpstreamModel(input_dim=sol1_dim, hidden_dims=[], output_dim=output_dim)
        self.sol2_downstream = ColeyDownstreamModel(input_dim=highway_dim+output_dim, hidden_dims=[300, 300], output_dim=sol2_dim, hidden_acts=[torch.nn.ReLU, torch.nn.Tanh], output_act=torch.nn.Softmax)
        highway_dim += output_dim

        output_dim = 100
        self.reag1_upstream = ColeyUpstreamModel(input_dim=sol2_dim, hidden_dims=[], output_dim=output_dim)
        self.reag1_downstream = ColeyDownstreamModel(input_dim=highway_dim+output_dim, hidden_dims=[300, 300], output_dim=reag1_dim, hidden_acts=[torch.nn.ReLU, torch.nn.Tanh], output_act=torch.nn.Softmax)
        highway_dim += output_dim

        output_dim = 100
        self.reag2_upstream = ColeyUpstreamModel(input_dim=reag1_dim, hidden_dims=[], output_dim=output_dim)
        self.reag2_downstream = ColeyDownstreamModel(input_dim=highway_dim+output_dim, hidden_dims=[300, 300], output_dim=reag2_dim, hidden_acts=[torch.nn.ReLU, torch.nn.Tanh], output_act=torch.nn.Softmax)
        highway_dim += output_dim

        output_dim = 100
        self.temp_upstream = ColeyUpstreamModel(input_dim=reag2_dim, hidden_dims=[], output_dim=output_dim)
        self.temp_downstream = ColeyDownstreamModel(input_dim=highway_dim+output_dim, hidden_dims=[300, 300], output_dim=temp_dim, hidden_acts=[torch.nn.ReLU, torch.nn.Tanh], output_act=torch.nn.Softmax)
        highway_dim += output_dim


    def argmax_matrix(self, m):
        max_idx = torch.argmax(m, 1, keepdim=True)
        one_hot = torch.FloatTensor(m.shape)
        one_hot.zero_()
        one_hot.scatter_(1, max_idx, 1)
        return one_hot


    def forward(
        self,
        *,
        product_fp,
        rxn_diff_fp,
        cat,
        sol1,
        sol2,
        reag1,
        reag2,
        training=True,
        force_teach=True,
        hard_select=True,
        stochastic_mid=True,
    ):
        fp_input = torch.cat((product_fp, rxn_diff_fp), dim=1)

        cat_mid = self.cat_upstream(fp_input, training=training)
        if stochastic_mid:
            cat_mid = torch.nn.Dropout(p=0.5)(cat_mid)
        cat_output = self.cat_downstream(cat_mid, training=training)

        _cat = cat if force_teach else cat_output
        if hard_select and not force_teach:
            _cat = self.argmax_matrix(_cat)
        sol1_mid = self.sol1_upstream(_cat, training=training)
        concat_cat_sol1 = torch.cat((cat_mid, sol1_mid), dim=1)
        sol1_output = self.sol1_downstream(concat_cat_sol1, training=training)

        _sol1 = sol1 if force_teach else sol1_output
        if hard_select and not force_teach:
            _sol1 = self.argmax_matrix(_sol1)
        sol2_mid = self.sol2_upstream(_sol1, training=training)
        concat_cat_sol1_sol2 = torch.cat((concat_cat_sol1, sol2_mid), dim=1)
        sol2_output = self.sol2_downstream(concat_cat_sol1_sol2, training=training)

        _sol2 = sol2 if force_teach else sol2_output
        if hard_select and not force_teach:
            _sol2 = self.argmax_matrix(_sol2)
        reag1_mid = self.reag1_upstream(_sol2, training=training)
        concat_cat_sol1_sol2_reag1 = torch.cat((concat_cat_sol1_sol2, reag1_mid), dim=1)
        reag1_output = self.reag1_downstream(concat_cat_sol1_sol2_reag1, training=training)

        _reag1 = reag1 if force_teach else reag1_output
        if hard_select and not force_teach:
            _reag1 = self.argmax_matrix(_reag1)
        reag2_mid = self.reag2_upstream(_reag1, training=training)
        concat_cat_sol1_sol2_reag1_reag2 = torch.cat((concat_cat_sol1_sol2_reag1, reag2_mid), dim=1)
        reag2_output = self.reag2_downstream(concat_cat_sol1_sol2_reag1_reag2, training=training)

        _reag2 = reag2 if force_teach else reag2_output
        if hard_select and not force_teach:
            _reag2 = self.argmax_matrix(_reag2)
        temp_mid = self.temp_upstream(_reag2, training=training)
        concat_cat_sol1_sol2_reag1_reag2_temp = torch.cat((concat_cat_sol1_sol2_reag1_reag2, temp_mid), dim=1)
        temp_output = self.temp_downstream(concat_cat_sol1_sol2_reag1_reag2_temp, training=training)

        return cat_output, sol1_output, sol2_output, reag1_output, reag2_output, temp_output
