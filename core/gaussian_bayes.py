import numpy as np
from scipy.stats import multivariate_normal

class GaussianNaivBayes(object):
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.priors    = [0] * num_classes
        self.means     = [None] * num_classes
        self.stddevs = [None] * num_classes

    def __str__(self):
        return 'num_classes:%d, priors:%s, means:%s, stddevs:%s'%(
                self.num_classes, str(self.priors), str(self.means), str(self.stddevs) )

    def predict(self, x):
        return [multivariate_normal.pdf(x, mean=m, cov=np.diag(s), allow_singular=True) for m, s in zip(self.means, self.stddevs)]

    def train(self, x, target):
        self.priors = [sum([1.0 for t in target if t==k])/len(target) for k in range(self.num_classes)]
        self.means = [ np.mean( [v for v, t in zip(x, target) if t==k], axis=0 ) for k in range(self.num_classes) ]
        self.stddevs = [ np.std( [v for v, t in zip(x, target) if t==k], axis=0 ) for k in range(self.num_classes) ]

    def dump_params(self):
        odict = {}
        odict['num_classes'] = self.num_classes
        odict['priors']      = self.priors
        odict['means']       = self.means
        odict['stddevs']   = self.stddevs
        return odict

    def load_params(self, idict):
        self.num_classes = idict['num_classes']
        self.priors      = idict['priors']
        self.means       = idict['means']
        self.stddevs   = idict['stddevs']

    def __test(self):
        '''
        >>> # Naive Vayes classification
        >>> c = GaussianNaivBayes(2)
        >>> c.train( x = np.array([ [1, 2, 1, 2,  5, 6, 5, 5],
        ...                         [5, 4, 4, 5,  1, 2, 2, 1]]).T,
        ...     target = np.array(  [0, 0, 0, 0,  1, 1, 1, 1] ) )
        >>> str(c)
        'num_classes:2, priors:[0.5, 0.5], means:[array([ 1.5,  4.5]), array([ 5.25,  1.5 ])], stddevs:[array([ 0.25,  0.25]), array([ 0.1875,  0.25  ])]'
        >>> y = c.predict( np.array( [[1, 6, 3], [5, 1, 4]] ).T )
        >>> print ['%.2f'%_ for _ in y[0]]
        ['0.99', '0.01']
        >>> [_ for _ in np.argmax(y, -1)]
        [0, 1, 0]
        '''

        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
