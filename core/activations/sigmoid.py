import numpy as np

class Sigmoid(object):
    def __init__(self):
        pass

    def forward(self, x):
        return 1. / (1. + np.exp(-x))

    def backward(self, y, target):
        return y - target

    def loss(self, y, target):
        return - np.sum( np.log(y) * target + np.log(1.-y) * (1.0 - target) )

    def __test(self):
        '''
        >>> x = np.log( np.array([[1, 2, 12], [1, 6, 4]]) )
        >>> t = np.array([[1, 0, 1], [0, 1, 0]])
        >>> f = Sigmoid()
        >>> y = f.forward( x )
        >>> print [['%.2f'%_ for _ in v] for v in y]
        [['0.50', '0.67', '0.92'], ['0.50', '0.86', '0.80']]
        >>> d = f.backward(y, t)
        >>> print [['%.2f'%_ for _ in v] for v in d]
        [['-0.50', '0.67', '-0.08'], ['0.50', '-0.14', '0.80']]
        >>> l = f.loss(y, t)
        >>> print '%.2f'%l
        4.33
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
