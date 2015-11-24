import numpy as np
from activations.softmax import Softmax

class Network:
    def __init__(self):
        self.layers = []
        self.activation = Softmax()
        self.init()

    def init(self):
        for layer in self.layers:
            layer.init()

    def predict(self, x):
        _ = x
        for layer in self.layers:
            _ = layer.forward( _ )
        return self.activation.forward( _ )

    def train(self, x, target):
        y = self.predict( x )
        _ = self.activation.backward( y, target )
        for layer in reversed( self.layers ):
            _ = layer.backward( _ )
            layer.update()
        return self.activation.loss( y, target )

    def __test(self):
        '''
        >>> from layers import Fullconnect
        >>> from nonlinears import ReLu, Tanh
        >>> from activations import Softmax, Sigmoid, Identity
        >>> from updaters import GradientDescent
        >>> np.random.seed(0xC0FFEE)
        >>> n = Network()
        >>> n.layers.append( Fullconnect(2, 10, ReLu.function, ReLu.derivative, updater=GradientDescent(learning_rate=0.01)) )
        >>> n.layers.append( Fullconnect(10, 2, updater=GradientDescent(learning_rate=0.01)) )
        >>> n.activation = Softmax()
        >>> for epoch in range(0, 20):
        ...     loss = n.train( x = np.array([ [1, 2, 1, 2,  5, 6, 5, 6],
        ...                                    [5, 4, 4, 5,  1, 2, 2, 1]]),
        ...                target = np.array([ [1, 1, 1, 1,  0, 0, 0, 0],
        ...                                    [0, 0, 0, 0,  1, 1, 1, 1]]) )
        ...     if epoch%5 == 0:
        ...         print 'epoch:%04d loss:%.2f'%(epoch, loss)
        epoch:0000 loss:9.84
        epoch:0005 loss:0.37
        epoch:0010 loss:0.24
        epoch:0015 loss:0.18
        >>> y = n.predict( np.array( [[1, 6, 3], [5, 1, 4]] ) )
        >>> [_ for _ in np.argmax(y, 0)]
        [0, 1, 0]
        >>> n = Network()
        >>> n.layers.append( Fullconnect(2, 10, ReLu.function, ReLu.derivative, updater=GradientDescent(learning_rate=0.01)) )
        >>> n.layers.append( Fullconnect(10, 2, updater=GradientDescent(learning_rate=0.01)) )
        >>> n.activation = Sigmoid()
        >>> for epoch in range(0, 20):
        ...     loss = n.train( x = np.array([ [1, 2, 1, 2,  5, 6, 5, 6,  5, 6, 5, 6],
        ...                                    [5, 4, 4, 5,  5, 4, 5, 4,  1, 2, 2, 1]]),
        ...                target = np.array([ [1, 1, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0],
        ...                                    [0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1]]) )
        ...     if epoch%5 == 0:
        ...         print 'epoch:%04d loss:%.2f'%(epoch, loss)
        epoch:0000 loss:17.45
        epoch:0005 loss:9.05
        epoch:0010 loss:5.83
        epoch:0015 loss:3.97
        >>> y = n.predict( np.array( [[1, 6, 3, 5], [5, 1, 4, 5]] ) )
        >>> [['%.2f'%_ for _ in v] for v in y]
        [['0.96', '0.06', '0.95', '0.95'], ['0.13', '0.99', '0.56', '0.86']]
        >>> y = n.predict( np.array( [[1, 6, 3], [5, 1, 4]] ) )
        >>> [_ for _ in np.argmax(y, 0)]
        [0, 1, 0]
        >>> n = Network()
        >>> n.layers.append( Fullconnect(2, 10, ReLu.function, ReLu.derivative, updater=GradientDescent(learning_rate=0.01)) )
        >>> n.layers.append( Fullconnect(10, 2, updater=GradientDescent(learning_rate=0.01)) )
        >>> n.activation = Identity()
        >>> for epoch in range(0, 20):
        ...     loss = n.train( x = np.array([ [1, 2, 1, 2,  5, 6, 5, 6,  5, 6, 5, 6],
        ...                                    [5, 4, 4, 5,  5, 4, 5, 4,  1, 2, 2, 1]]),
        ...                target = np.array([ [1, 1, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0],
        ...                                    [0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1]]) )
        ...     if epoch%5 == 0:
        ...         print 'epoch:%04d loss:%.2f'%(epoch, loss)
        epoch:0000 loss:18.67
        epoch:0005 loss:3.17
        epoch:0010 loss:2.46
        epoch:0015 loss:2.00
        >>> y = n.predict( np.array( [[1, 6, 3, 5], [5, 1, 4, 5]] ) )
        >>> [['%.2f'%_ for _ in v] for v in y]
        [['1.36', '0.43', '0.72', '0.54'], ['0.15', '0.69', '0.52', '0.63']]
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
