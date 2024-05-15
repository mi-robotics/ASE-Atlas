from learning.ase_models import ModelASEContinuous

class ModelTripASEContinuous(ModelASEContinuous):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('ase', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelTripASEContinuous.Network(net)

    class Network(ModelASEContinuous.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            result = super().forward(input_dict)

            if (is_train):
                amp_obs = input_dict['amp_obs']
                enc_pred = self.a2c_network.eval_enc(amp_obs)
                result["enc_pred"] = enc_pred
       

            return result