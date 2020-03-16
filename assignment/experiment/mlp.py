#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv

sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = '../../data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')


    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 20, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)

def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784, 32, 10])
    # train the network using SGD
    eval_cost, eval_accuracy,train_cost, train_accuracy = model.SGD(
        training_data=train_data,
        epochs=100,
        mini_batch_size=64,
        eta=1e-2,
        lmbda = 0.0,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
    
    test_total = model.accuracy(test_data, convert=False)
    results = [np.argmax(model.feedforward(x))
                        for (x, y) in zip(*test_data)]
    one_hot = [network2.vectorized_result(i) for i in results]
    with open('predictions.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for i in one_hot:
            spamwriter.writerow(int(j) for j in i)

    print("Testing: {}/{}".format(test_total,len(test_data[0])))
    # Plot Learning curves
    print("PLOTING")
    #Loss
    plt.plot(list(range(len(eval_cost))),eval_cost,'b-',list(range(len(train_cost))),train_cost,'r-')
    plt.legend(('Validation Loss','Training Loss'))
    plt.title("Loss Curve: batch=128, eta=e-2, hl=20")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    
    #Error
    plt.plot(list(range(len(eval_accuracy))),[x/len(valid_data[0]) for x in eval_accuracy],'b-',list(range(len(train_accuracy))),[x/len(train_data[0]) for x in train_accuracy],'r-')
    plt.legend(('Validation Accuracy','Training Accuracy'))
    plt.title("Accuracy Curve: batch=128, eta=e-2, hl=20")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
    


if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()

#if __name__ == '__main__':
#    load_data()
#    test_sigmoid()
#    train_data, valid_data, test_data = load_data()
#    main()
