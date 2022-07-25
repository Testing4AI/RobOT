from adf_model.model import Model
from adf_model.layer import *

class MLP(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """

    def __init__(self, layers, input_shape):
        """
        Construct a multilayer perceptron (MLP)
        :param layers: a sequence of layers
        :param input_shape: the shape of dataset
        """
        super(MLP, self).__init__()

        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
                layer.name = name
            self.layer_names.append(name)

            layer.set_input_shape(input_shape)
            input_shape = layer.get_output_shape()

    def fprop(self, x, set_ref=False):
        """
        Get the symbolic output of each layer
        :param x: the input placeholder
        :param set_ref: whether set reference
        :return: a dictionary of layers' name and tensor
        """
        states = []
        for layer in self.layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states