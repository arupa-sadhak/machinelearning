# machinelearning [![Build Status](https://travis-ci.org/wbaek/machinelearning.svg?branch=master)](https://travis-ci.org/wbaek/machinelearning)

내용만 알면 간단한 neural network에 대한 이론 정리 및 구현내용 입니다.

각 코드는 test코드를 제외한 30줄 내외로 정리되어 있습니다.

현재는 코드만 있고 이론적인 내용은 추가해갈 예정입니다.


## Features
* Layer
  * Fullconnect Layer
  * ~~Convolution Layer~~
* Nonlinear Function
  * tanh
  * ReLu
* Activation \w Loss
  * Softmax \w cross-entropy error
  * ~~Sigmoid \w cross-entropy error~~
  * ~~Identity \w sum-of-squre error~~
* Updater
  * Vanila Gradient Updater
  * ~~Momentum~~
  * ~~AdaGradient~~
* Initializer
  * Xavier Initializer
  * ~~Kaiming Initializer~~

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
>>> from core.nonlinears.relu import ReLu
>>> from core.activations.softmax import Softmax
>>> np.random.seed(0xC0FFEE)
>>> n = Network()
>>> n.layers.append( Layer(2, 10, ReLu.function, ReLu.derivative) )
>>> n.layers.append( Layer(10, 2) )
>>> n.activation = Softmax()
>>> for epoch in range(0, 20):
...     loss = n.train( x = np.array([ [1, 2, 1, 2,  5, 6, 5, 6],
...                                    [5, 4, 4, 5,  1, 2, 2, 1]]),
...                target = np.array([ [1, 1, 1, 1,  0, 0, 0, 0],
...                                    [0, 0, 0, 0,  1, 1, 1, 1]]) )
...     if epoch%5 == 0:
...         print 'epoch:%04d loss:%.2f'%(epoch, loss)
epoch:0000 loss:4.92
epoch:0005 loss:0.19
epoch:0010 loss:0.12
epoch:0015 loss:0.09
>>> y = n.predict( np.array( [[1, 6, 3], [5, 1, 4]] ) )
>>> [_ for _ in np.argmax(y, 0)]
[0, 1, 0]
```


### Multiple-class Classification
* not yet



### Regression
* not yet


