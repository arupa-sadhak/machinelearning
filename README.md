# machinelearning [![Build Status](https://travis-ci.org/wbaek/machinelearning.svg?branch=master)](https://travis-ci.org/wbaek/machinelearning)

내용만 알면 간단하게 구현할 수 있는 neural network에 대한 이론 정리 및 구현내용 입니다.

각 코드는 초기화 및 테스트코드를 제외한 10~20줄 내외로 정리되어 있습니다.

현재는 코드만 있고 이론적인 내용은 추가해갈 예정입니다.


## Features
* Network
  * [Feedforward Network](https://github.com/wbaek/machinelearning/blob/master/core/network.py)
  * ~~Recurrent Network~~
  * ~~Autoencoder Network~~
* Layer
  * [Fullconnect Layer](https://github.com/wbaek/machinelearning/blob/master/core/layer.py)
  * ~~Dropout Layer~~
  * ~~Convolution Layer~~
  * ~~SharedWeight Layer~~
* Nonlinear Function
  * [Linear](https://github.com/wbaek/machinelearning/blob/master/core/nonlinears/linear.py)
  * [ReLu](https://github.com/wbaek/machinelearning/blob/master/core/nonlinears/relu.py)
  * [tanh](https://github.com/wbaek/machinelearning/blob/master/core/nonlinears/tanh.py)
* Activation \w Negative Log Likelihood Loss
  * [Softmax \w cross-entropy error](https://github.com/wbaek/machinelearning/blob/master/core/activations/softmax.py)
  * [Sigmoid \w cross-entropy error](https://github.com/wbaek/machinelearning/blob/master/core/activations/sigmoid.py)
  * ~~Identity \w sum-of-squre error~~
* Updater
  * Vanila [Gradient Descent](https://github.com/wbaek/machinelearning/blob/master/core/updaters/gradient_descent.py)
  * ~~Momentum~~
  * ~~AdaGradient~~
* Initializer
  * Xavier (implement in Network init function)
  * ~~Kaiming Initializer~~
* Aggregator
  * ~~Comcat~~
  * ~~Reshape~~

## Requirements
```
pip install -r requirements.txt
```


## Usage
### Multiclass Classification
* each input is assigned to one of K mutually exclusive classes.
```python
>>> from core.network import Network
>>> from core.layer import Layer
>>> from core.nonlinears import ReLu
>>> from core.activations import Softmax
>>> from core.updaters import GradientDescent
>>> np.random.seed(0xC0FFEE)
>>> n = Network()
>>> n.layers.append( Layer(input_size=2, output_size=10, ReLu.function, ReLu.derivative, updater=GradientDescent(learning_rate=0.01)) )
>>> n.layers.append( Layer(input_size=10, output_size=2, updater=GradientDescent(learning_rate=0.01)) )
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
```


### Multiple-class Classification
```python
>>> from core.network import Network
>>> from core.layer import Layer
>>> from core.nonlinears import ReLu
>>> from core.activations import Sigmoid
>>> from core.updaters import GradientDescent
>>> np.random.seed(0xC0FFEE)
>>> n = Network()
>>> n.layers.append( Layer(input_size=2, output_size=10, ReLu.function, ReLu.derivative, updater=GradientDescent(learning_rate=0.01)) )
>>> n.layers.append( Layer(input_size=10, output_size=2, updater=GradientDescent(learning_rate=0.01)) )
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
```



### Regression
* not yet


