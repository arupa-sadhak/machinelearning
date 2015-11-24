import math
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from nonlinears import Linear
from updaters.gradient_descent import GradientDescent

class Fullconnect:
    def __init__(self, input_size, output_size,
            nonlinear_function=Linear.function, derivative_function=Linear.derivative,
            updater=GradientDescent() ):
        self.input_size = input_size
        self.output_size = output_size
        # xavier initializer
        self.W = math.sqrt(6./(output_size+input_size)) * np.random.uniform( -1.0, 1.0, (output_size, input_size) )
        self.b = np.zeros( (output_size, 1) )
        self.params = [self.W, self.b]
        self.nonlinear_function = nonlinear_function
        self.derivative_function = derivative_function
        self.updater = updater

    def forward(self, x):
        self.x = x
        self.a = np.dot( self.W, x ) + self.b
        return self.nonlinear_function( self.a )

    def backward(self, delta):
        self.delta_a = delta * self.derivative_function(self.a)
        return np.dot( self.W.T, self.delta_a )

    def get_gradient(self):
        return ( np.dot(self.delta_a, self.x.T), np.dot(self.delta_a, np.ones((self.delta_a.shape[1], 1))) )

    def update(self):
        for param, gradient in zip(self.params, self.get_gradient()):
            param = self.updater.update(param, gradient)

    def __test(self):
        '''
        >>> x = np.array([[1],[2],[3]])
        >>> l = Fullconnect(3, 4)
        >>> l.W = np.eye(4, 3)
        >>> l.b = np.array([[0.3], [0.5], [0], [0]])
        >>> y = l.forward( x )
        >>> y.shape
        (4, 1)
        >>> print [_ for _ in np.asarray( y.T[0] )]
        [1.3, 2.5, 3.0, 0.0]
        >>> delta = np.array([[1], [1], [1], [1]])
        >>> d = l.backward( delta )
        >>> print  [_ for _ in np.asarray( d.T[0] )]
        [1.0, 1.0, 1.0]
        >>> dW, db = l.get_gradient()
        >>> dW.shape
        (4, 3)
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
