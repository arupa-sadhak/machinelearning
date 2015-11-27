import numpy as np

class Softmax(object):
    def __init__(self):
        self.eps = 1e-6

    def forward(self, x):
        _ = np.exp(x)
        return np.divide(_, np.array(_.sum( axis=-1 ), ndmin=2).T)

    def backward(self, y, target):
        return y - target

    def loss(self, y, target):
        return - np.sum( np.log(y+self.eps) * target )

    def __test(self):
        '''
        >>> x = np.log( np.array([[1, 1], [12, 6], [3, 8]]) )
        >>> t = np.array([[1, 0], [0, 1], [1, 0]])
        >>> f = Softmax()
        >>> y = f.forward( x )
        >>> print [['%.2f'%_ for _ in v] for v in y]
        [['0.50', '0.50'], ['0.67', '0.33'], ['0.27', '0.73']]
        >>> d = f.backward(y, t)
        >>> print [['%.2f'%_ for _ in v] for v in d]
        [['-0.50', '0.50'], ['0.67', '-0.67'], ['-0.73', '0.73']]
        >>> l = f.loss(y, t)
        >>> print '%.2f'%l
        3.09
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
