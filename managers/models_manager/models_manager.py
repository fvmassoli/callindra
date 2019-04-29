import os
import torch


class ModelManager(object):
    def __init__(self, gpu=False, verbose=0):
        self._models = {}
        self._device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
        self._verbose = verbose

    def load_model(self, model, path, k, load_state_dict=True):
        """ Load a model and adds it to the models dictionary

        :param model: model class (only required with state dict)
        :param path: path to model checkpoint file
        :param k: key of the state dict
        :param load_state_dict: True (False) to load model with state dict (entire model)

        :return: loaded model

        """
        assert os.path.exists(path), "Model checkpoint not found at: {}".format(path)
        if load_state_dict:
            state_dict = torch.load(path)[k]
            assert k in state_dict, "State dict not found with key: {}".format(k)
            model.load_state_dict(state_dict, map_location='cpu')
        else:
            model = torch.load(path)
        model.to(self._device)
        model_name = model.__class__.__name__
        self._models[model_name] = model
        self._print_model_info(model_name, model)
        return model

    def get_models(self):
        """ Returns the list of loaded models

        """
        return self._models

    def get_model_by_key(self, k):
        """ Returns model with the specified key

        :param k: name of the model

        """
        assert k in self._models, "Key Not found. Available models are: {}".format(self._models.keys())
        return self._models[k]

    def move_model_to_device(self, k, device):
        """Move the model k to the specified device

        :param k: model name
        :param device: device on which to move the model

        """
        self._models[k].to(device)

    def _print_model_info(self, model_name, model):
        """ Prints info about loaded model

        """
        if self._verbose > 0:
            print("="*20, "Model info", "="*19)
            print("\t Model name: {}".format(model_name))
            print("\t Total number of model parameters: {}".format(sum(p.numel() for p in model.parameters())))
            print("="*51)

    def forward(self, k, x):
        """Returns the output of the forward method for the specified model

        :param k: model name
        :param x: input to the model

        :return: output of forward pass

        """
        return self._models[k].forward(x)
