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

    def init(self):
        self.recurrent_output = None
        self.recurrent_delta = None

    def forward(self, x):
        if self.recurrent_output == None:
            self.recurrent_output = np.zeros( (self.output_size, x.shape[1]) )
        self.x = np.concatenate( [x, self.recurrent_output], axis=0 )
        self.recurrent_output = super(Recurrent, self).forward( self.x )
        return self.recurrent_output

    def backward(self, delta):
        if self.recurrent_delta == None:
            self.recurrent_delta = np.zeros_like( delta )
        _, self.recurrent_delta = np.vsplit( super(Recurrent, self).backward(delta + self.recurrent_delta), [self.input_size-self.output_size] )
        return _

    def update(self):
        self.reucrrent_layer.update()

    def __test(self):
        '''
        >>> np.random.seed(0xC0FFEE)
        >>> x = np.array([[1],[2],[3]])
        >>> l = Recurrent(3, 4)
        >>> y = l.forward( x )
        >>> y.shape
        (4, 1)
        >>> print ['%.1f'%_ for _ in np.asarray( y.T[0] )]
        ['2.9', '0.7', '3.4', '0.0']
        >>> np.array_equal( y, l.forward( x ) )
        False
        >>> np.array_equal( y, l.forward( x ) )
        False
        >>> l.init()
        >>> x = np.array([[1, 2], [2, 3], [3, 4]])
        >>> y = l.forward( x )
        >>> y.shape
        (4, 2)
        >>> delta = np.array([[1,2], [1,2], [1,2], [1,2]])
        >>> d = l.backward( delta )
        >>> x.shape == d.shape
        True
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
