import math
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from nonlinears import Linear
from updaters.gradient_descent import GradientDescent
from layers import Fullconnect

class BiRecurrent(Fullconnect):
    def __init__(self, input_size, output_size,
            nonlinear_function=Linear.function, derivative_function=Linear.derivative,
            updater=GradientDescent() ):
        super(BiRecurrent, self).__init__(input_size+(output_size/2), output_size/2,
                nonlinear_function, derivative_function, updater)
        self.forward_layer  = self#.folk(shared=False)
        self.backward_layer = self#.folk(shared=False)

    def forward(self, x):
        time_splited_x = [_ for _ in np.split( x, x.shape[0] )]
        self.times = len(time_splited_x)

        shape = list(time_splited_x[0].shape)
        shape[-1] = self.output_size

        forward_outputs = [np.array([])] * (self.times+1)
        forward_outputs[-1] = np.zeros(shape)

        self.shared_forward_layers = []
        for t, splited in           zip(range(self.times), time_splited_x):
            layer = self.forward_layer.folk(shared=True)
            forward_outputs[t] = layer.forward( np.concatenate( [splited, forward_outputs[t-1]], axis=-1 ) )
            self.shared_forward_layers.append( layer )

        backward_outputs = [np.array([])] * (self.times+1)
        backward_outputs[-1] = np.zeros(shape)

        self.shared_backward_layers = []
        for t, splited in reversed( zip(range(self.times), time_splited_x) ):
            layer = self.backward_layer.folk(shared=True)
            backward_outputs[t] = layer.forward( np.concatenate( [splited, backward_outputs[t+1]], axis=-1 ) )
            self.shared_backward_layers.append( layer )

        return np.concatenate( [np.concatenate( forward_outputs[:self.times] ), np.concatenate( backward_outputs[:self.times] )], axis=-1 )

    def backward(self, delta):
        forward_delta, backward_delta = np.split( delta, 2, axis=-1 )
        time_splited_forward_delta  = [_ for _ in np.split(  forward_delta, delta.shape[0] )]
        time_splited_backward_delta = [_ for _ in np.split( backward_delta, delta.shape[0] )]
        self.times = len(time_splited_forward_delta)

        shape = list(time_splited_forward_delta[0].shape)
        shape[-1] = self.output_size

        forward_deltas = [np.array([])] * (self.times+1)
        forward_deltas[-1] = np.zeros(shape)
        forward_outputs = []
        for t, layer, splited in reversed( zip(range(self.times), self.shared_forward_layers, time_splited_forward_delta) ):
            _, forward_deltas[t] = np.split( layer.backward( splited+forward_deltas[t+1] ), [self.input_size-self.output_size], axis=-1 )
            forward_outputs.append( _ )

        backward_deltas = [np.array([])] * (self.times+1)
        backward_deltas[-1] = np.zeros(shape)
        backward_outputs = []
        for t, layer, splited in           zip(range(self.times), self.shared_backward_layers, time_splited_backward_delta):
            _, backward_deltas[t] = np.split( layer.backward(splited+backward_deltas[t-1] ), [self.input_size-self.output_size], axis=-1 )
            backward_outputs.append( _ )

        return np.concatenate( [forward+backward for forward, backward in zip(forward_outputs[:self.times], backward_outputs[:self.times])] )

    def update(self):
        dW = np.zeros_like( self.W )
        db = np.zeros_like( self.b )
        for layer in self.shared_forward_layers:
            _ = layer.get_gradient()
            dW += _[0]
            db += _[1]
        self.forward_layer.W = self.updater.update(self.forward_layer.W, dW)
        self.forward_layer.b = self.updater.update(self.forward_layer.b, db)

        dW = np.zeros_like( self.W )
        db = np.zeros_like( self.b )
        for layer in self.shared_backward_layers:
            _ = layer.get_gradient()
            dW += _[0]
            db += _[1]
        self.backward_layer.W = self.updater.update(self.forward_layer.W, dW)
        self.backward_layer.b = self.updater.update(self.forward_layer.b, db)

        #self.W = self.updater.update(self.W, dW/(self.times+2))
        #self.b = self.updater.update(self.b, db/(self.times+2))

    def __test(self):
        '''
        >>> np.random.seed(0xC0FFEE)
        >>> x = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
        >>> x.shape
        (2, 2, 3)
        >>> l = BiRecurrent(3, 4)
        >>> y = l.forward( x )
        >>> y.shape
        (2, 2, 4)
        >>> print ['%.1f'%_ for _ in y[0][0]]
        ['0.4', '1.6', '1.4', '1.5']
        >>> np.array_equal( y, l.forward( x ) )
        True
        >>> delta = np.array([[[1,1,1,1], [1,1,1,1]], [[0,0,0,0], [2,2,2,2]]])
        >>> d = l.backward( delta )
        >>> x.shape == d.shape
        True
        >>> l.update()
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
