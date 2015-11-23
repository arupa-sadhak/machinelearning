import numpy as np

class ReLu:
    def __init__(self):
        pass

    @staticmethod
    def function(x):
        mapper = np.zeros_like( x )
        return np.fmax( x, mapper )

    @staticmethod
    def derivative(x):
        return np.array( [[(1 if _>0 else 0) for _ in v] for v in x] )

    def __test(self):
        '''
        >>> x = np.array( [[-1, 3, -1, 1, 2], [1, -1, 0.5, -1, -2]] )
        >>> f = ReLu.function
        >>> y = f( x )
        >>> [['%.1f'%_ for _ in v] for v in y]
        [['0.0', '3.0', '0.0', '1.0', '2.0'], ['1.0', '0.0', '0.5', '0.0', '0.0']]
        >>> d = ReLu.derivative
        >>> y = d( x )
        >>> [['%.1f'%_ for _ in v] for v in y]
        [['0.0', '1.0', '0.0', '1.0', '1.0'], ['1.0', '0.0', '1.0', '0.0', '0.0']]
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
