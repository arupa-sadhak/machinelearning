import math
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from nonlinears import Linear
from updaters.gradient_descent import GradientDescent
from layers import Fullconnect

class Dropout(Fullconnect):
    def __init__(self, input_size, output_size, drop_ratio=0.5,
            nonlinear_function=Linear.function, derivative_function=Linear.derivative,
            updater=GradientDescent() ):
        super(Dropout, self).__init__(input_size, output_size, nonlinear_function, derivative_function, updater)
        self.drop_ratio = drop_ratio
        self.is_testing = False

    def forward(self, x):
        y = super(Dropout, self).forward( x )
        if self.is_testing:
            return y * self.drop_ratio
        self.drop_map = np.array( [1.0 if v>=self.drop_ratio else 0.0 for v in np.random.uniform(0, 1, np.prod( y.shape ))] ).reshape( y.shape )
        return np.multiply(y, self.drop_map)

    def backward(self, delta):
        return super(Dropout, self).backward( np.multiply(delta, self.drop_map) )

    def __test(self):
        '''
        >>> np.random.seed(0xC0FFEE)
        >>> x = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
        >>> x.shape
        (2, 2, 3)
        >>> l = Dropout(3, 4)
        >>> y = l.forward( x )
        >>> y.shape
        (2, 2, 4)
        >>> print ['%.1f'%_ for _ in y[0][0]]
        ['0.0', '2.8', '0.6', '-3.6']
        >>> np.array_equal( y, l.forward( x ) )
        False
        >>> delta = np.array([[[1,1,1,1], [1,1,1,1]], [[0,0,0,0], [2,2,2,2]]])
        >>> d = l.backward( delta )
        >>> print ['%.1f'%_ for _ in d[0][0]]
        ['-0.2', '-0.1', '-0.2']
        >>> x.shape == d.shape
        True
        >>> l.update()
        >>> type( l ).__name__
        'Dropout'
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
