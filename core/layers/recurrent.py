import math
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from nonlinears import Linear
from updaters.gradient_descent import GradientDescent
from layers import Fullconnect

class Recurrent(Fullconnect):
    def __init__(self, input_size, output_size,
            nonlinear_function=Linear.function, derivative_function=Linear.derivative,
            updater=GradientDescent() ):
        super(Recurrent, self).__init__(input_size+output_size, output_size, nonlinear_function, derivative_function, updater)

    def forward(self, x):
        self.x = x
        self.time_size = len(x)
        self.minibatch_size = x[0].shape[1]

        recurrent_output = [np.array([])] * (self.time_size+1)
        recurrent_output[0] = np.zeros( (self.output_size, self.minibatch_size)  )

        self.unfolded_layers = []
        for t in xrange( self.time_size ):
            splited_x = x[t]
            splited_h = recurrent_output[t]
            layer = Fullconnect(self.input_size, self.output_size,
                self.nonlinear_function, self.derivative_function, self.updater)
            layer.W = self.W
            layer.b = self.b

            recurrent_output[t+1] = layer.forward( np.concatenate( [splited_x, splited_h], axis=0 ) )
            self.unfolded_layers.append( layer )
        return np.concatenate( recurrent_output[1:], axis=1 )

    def backward(self, delta):
        time_splited_delta = np.hsplit( delta, [self.minibatch_size*i for i in range(1, self.time_size)] )

        recurrent_delta = [np.array([])] * (self.time_size+1)
        recurrent_delta[-1] = np.zeros( (self.output_size, self.minibatch_size) )
        input_delta = [np.array([])] * (self.time_size)

        for t in reversed(range( self.time_size )):
            splited_delta = time_splited_delta[t]

            _ = self.unfolded_layers[t].backward( splited_delta + recurrent_delta[t+1] )
            input_delta[t], recurrent_delta[t] = np.vsplit( _, [self.input_size-self.output_size] )

        return input_delta

    def __test(self):
        '''
        >>> np.random.seed(0xC0FFEE)
        >>> x = [np.array([[1],[2],[3]])] * 2
        >>> l = Recurrent(3, 4)
        >>> y = l.forward( x )
        >>> y.shape
        (4, 2)
        >>> print ['%.1f'%_ for _ in np.asarray( y.T[0] )]
        ['2.9', '0.7', '3.4', '0.0']
        >>> np.array_equal( y.T[0], y.T[1] )
        False
        >>> np.array_equal( y, l.forward( x ) )
        True
        >>> x = [np.array([[1, 2], [2, 3], [3, 4]])] * 2
        >>> y = l.forward( x )
        >>> y.shape
        (4, 4)
        >>> delta = np.array([[1,2, 3, 4], [1,2, 3, 4], [1,2, 3, 4], [1,2, 3, 4]])
        >>> d = l.backward( delta )
        >>> len(x) == len(d)
        True
        >>> x[0].shape == d[0].shape
        True
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
