import cPickle as pkl
import numpy as np
import random
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from core.network import Network
from core.layers import Fullconnect, Recurrent, BiRecurrent
from core.activations import Softmax
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

    logging.info('load data start')
    train_lex, train_y = pkl.load( open('datas/kowiki_spacing_train.pkl', 'r') )
    words2idx = pkl.load( open('datas/kowiki_dict.pkl', 'r') )
    logging.info('load data done')

    index2words = {value:key for key, value in words2idx.iteritems()}

    vocsize = len(words2idx) + 1
    nclasses = 2
    nsentences = len(train_lex)
    max_iter = min(args.samples, nsentences)
    logging.info('vocsize:%d, nclasses:%d, nsentences:%d, samples:%d, max_iter:%d'%(vocsize, nclasses, nsentences, args.samples, max_iter))

    context_window_size = args.window_size

    logging.info('learning_rate: %f'%args.learning_rate)
    learning_rate = args.learning_rate
    n = Network()
    n.layers.append( Fullconnect(vocsize, 256, Tanh.function, Tanh.derivative,  updater=GradientDescent(learning_rate)) )
    n.layers.append( Recurrent(256, 256, Tanh.function, Tanh.derivative, updater=GradientDescent(learning_rate)) )
    n.layers.append( Fullconnect(256, 256, ReLu.function, ReLu.derivative, updater=GradientDescent(learning_rate)) )
    n.layers.append( Fullconnect(256, nclasses, updater=GradientDescent(learning_rate)) )
    n.activation = Softmax()

    if os.path.isfile( args.params ):
        logging.info('load parameters from %s'%args.params)
        n.load_params( pkl.load(open(args.params, 'rb')) )

    logging.info('train start')
    for epoch in xrange(0, args.epoch):
        epoch_loss = 0
        epoch_error_rate = 0
        for i in xrange( max_iter ):
            idx = random.randint(0, nsentences-1)
            cwords = contextwin(train_lex[idx], context_window_size)
            words, labels = onehotvector(cwords, vocsize, train_y[idx], nclasses)

            loss = n.train( words, labels ) / len(words) # sequence normalized loss

            y = np.zeros_like(labels)
            for index1, index2 in enumerate([np.argmax(prediction) for prediction in n.y]):
                y[index1][index2] = 1
            error_rate = np.sum(np.absolute( y - labels )) / np.sum(y.shape)

            epoch_loss += loss
            epoch_error_rate += error_rate
            if i%10 == 0 and i != 0:
                logging.info('[%.4f%%] epoch:%04d iter:%04d loss:%.5f error-rate:%.5f'%((i+1)/float(max_iter), epoch, i, epoch_loss/(i+1), epoch_error_rate/(i+1)))

        logging.info('epoch:%04d loss:%.5f, error-rate:%.5f'%(epoch, epoch_loss/max_iter, epoch_error_rate/max_iter))
        pkl.dump( n.dump_params(), open(args.params, 'wb') )
        logging.info('dump parameters at %s'%(args.params))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-size',          type=int,   default=1)
    parser.add_argument('--epoch',                type=int,   default=10)
    parser.add_argument('--learning-rate',        type=float, default=0.001)
    parser.add_argument('-p', '--params',         type=str, required=True)
    parser.add_argument('-n', '--samples',        type=int, default=100000 )
    parser.add_argument('--log-filename',         type=str, default='')
    args = parser.parse_args()

    if not args.log_filename:
        logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s %(levelname)s] %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', filename=args.log_filename)

    main( args )

