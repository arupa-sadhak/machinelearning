import cPickle as pkl
import numpy as np
import random
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from core.network import Network
from core.layers import Fullconnect, Recurrent, BiRecurrent
from core.activations import Softmax, Sigmoid
from core.nonlinears import Linear, ReLu, Tanh
from core.updaters import GradientDescent

def contextwin(l, win):
    l = list(l)
    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]
    return out

def onehotvector(cwords, vocsize, y=[], nclasses=1):
    words = np.zeros( (len(cwords), vocsize) )
    labels = np.zeros( (len(cwords), nclasses) )
    for i, cword in enumerate( cwords ):
        for idx in cword:
            words[i][idx] = 1

    for i, label in enumerate( y ):
        labels[i][label] = 1
    return (words, labels)

def main(args):
    np.random.seed(0xC0FFEE)

    train, test, dicts = pkl.load( open('datas/atis.pkl', 'r') )
    index2words = {value:key for key, value in dicts['words2idx'].iteritems()}
    index2tables = {value:key for key, value in dicts['tables2idx'].iteritems()}
    index2labels = {value:key for key, value in dicts['labels2idx'].iteritems()}

    datas = [
            {'name':'train', 'x':train[0], 'y':train[2], 'size':len(train[0])},
            {'name':'test',  'x':test[0],  'y':test[2], 'size':len(test[0])},
            ]

    vocsize = len(dicts['words2idx']) + 1
    nclasses = len(dicts['labels2idx'])
    context_window_size = args.window_size
    learning_rate = args.learning_rate
    logging.info('vocsize:%d, nclasses:%d window-size:%d learning-rate:%.5f'%(vocsize, nclasses, context_window_size, learning_rate))

    n = Network()
    n.layers.append( Fullconnect(vocsize, 100, Tanh.function, Tanh.derivative, updater=GradientDescent(learning_rate)) )
    n.layers.append( Recurrent(100, 100, Tanh.function, Tanh.derivative, updater=GradientDescent(learning_rate)) )
    n.layers.append( Fullconnect(100, 100, ReLu.function, ReLu.derivative, updater=GradientDescent(learning_rate)) )
    n.layers.append( Fullconnect(100, nclasses, updater=GradientDescent(learning_rate)) )
    n.activation = Softmax()

    for epoch in xrange(args.epoch):
        for data in datas:
            epoch_loss = 0
            epoch_error_rate = 0
            for i in xrange(data['size']):
                idx = random.randint(0, data['size']-1)
                cwords = contextwin(data['x'][i], context_window_size)
                words, labels = onehotvector(cwords, vocsize, data['y'][i], nclasses)

                if data['name'] == 'train':
                    loss = n.train( words, labels ) / len(words) # sequence normalized loss
                    predictions = n.y
                else:
                    predictions = n.predict( words )
                    loss = n.activation.loss( predictions, labels ) / len(words) # sequence normalized loss

                y = np.zeros_like(labels)
                for index1, index2 in enumerate([np.argmax(prediction) for prediction in predictions]):
                    y[index1][index2] = 1
                error_rate = np.sum(np.absolute( y - labels )) / np.sum(y.shape)

                epoch_loss += loss
                epoch_error_rate += error_rate
                if i%1000 == 0 and i != 0 and data['name'] == 'train':
                    logging.info( 'epoch:%04d iter:%04d loss:%.5f error-rate:%.5f'%(epoch, i, epoch_loss/(i+1), epoch_error_rate/(i+1)) )

            logging.info( '[%5s] epoch:%04d loss:%.5f error-rate:%.5f'%(data['name'], epoch, epoch_loss/data['size'], epoch_error_rate/data['size']) )

    for i in range(20):
        idx = random.randint(0, datas[1]['size']-1)
        x = datas[1]['x'][idx]
        labels = datas[1]['y'][idx]

        cwords = contextwin(datas[1]['x'][idx], context_window_size)
        words = onehotvector(cwords, vocsize)[0]
        _ = n.predict(words)

        y = [np.argmax(prediction) for prediction in _]

        print 'word:   ', ' '.join([index2words[_] for _ in x])
        print 'label:  ', ' '.join([index2labels[_] for _ in labels])
        print 'predict:', ' '.join([index2labels[_] for _ in y])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-size',          type=int,   default=1)
    parser.add_argument('--epoch',                type=int,   default=10)
    parser.add_argument('--learning-rate',        type=float, default=0.01)
    parser.add_argument('--log-filename',         type=str,   default='')
    args = parser.parse_args()

    if not args.log_filename:
        logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s %(levelname)s] %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', filename=args.log_filename)

    main( args )

