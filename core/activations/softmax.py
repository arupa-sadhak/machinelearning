import numpy as np

class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        return np.exp(x) / sum( np.exp(x), 0 )

    def backward(self, y, target):
        return y - target

    def loss(self, y, target):
        return - np.sum( np.log(y) * target )

    def __test(self):
        '''
        >>> x = np.log( np.array([[1, 2, 12], [1, 6, 4]]) )
        >>> t = np.array([[1, 0, 1], [0, 1, 0]])
        >>> f = Softmax()
        >>> y = f.forward( x )
        >>> print [['%.2f'%_ for _ in v] for v in y]
        [['0.50', '0.25', '0.75'], ['0.50', '0.75', '0.25']]
        >>> d = f.backward(y, t)
        >>> print [['%.2f'%_ for _ in v] for v in d]
        [['-0.50', '0.25', '-0.25'], ['0.50', '-0.25', '0.25']]
        >>> l = f.loss(y, t)
        >>> print '%.2f'%l
        1.27
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
