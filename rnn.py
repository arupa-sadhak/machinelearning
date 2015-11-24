import numpy as np

from core.network import Network
from core.layers import Fullconnect, Recurrent
from core.activations import Softmax
from core.nonlinears import Linear, ReLu, Tanh
from core.updaters import GradientDescent

if __name__ == '__main__':
    np.random.seed(0xC0FFEE)

    learning_rate = 0.01
    n = Network()
    n.layers.append( Recurrent(2, 10, Tanh.function, Tanh.derivative, updater=GradientDescent(learning_rate)) )
    n.layers.append( Fullconnect(10, 2, updater=GradientDescent(learning_rate)) )

    input  = np.array([[1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
                       [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]])
    target = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

    for epoch in range(0, 1001):
        loss = 0
        n.init()
        for x, t in zip(input.T, target.T):
            loss += n.train( x.reshape(2, 1), t.reshape(2, 1) )
        if epoch%10 == 0:
            print 'epoch:%04d loss:%.2f'%(epoch, loss)

    n.init()
    for x, t in zip(input.T, target.T):
        y = n.predict( x.reshape(2, 1) )
        print 'x=', ','.join(['%.2f'%_ for _ in x]), 'y=', ','.join(['%.2f'%_ for _ in y])

