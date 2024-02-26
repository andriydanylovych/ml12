#!/usr/bin/env python
# coding: utf-8

# In[367]:


import numpy
import math
import csv
import random

def open_csv_file(file_name): # 1 label + 784 pixels; 256/60000 images
    matrix = []
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            matrix.append(row)
    return matrix

def normalize_pixels(matrix_int):
    MAX_PX_VALUE: float = 255. # Max pixel value
    train_float = numpy.zeros((len(matrix_int), len(matrix_int[0])))
    for item in range (len(matrix_int)):
        for pixel in range (len(matrix_int[0]) - 1):
            train_float[item][pixel] = float(matrix_int[item][pixel + 1]) / MAX_PX_VALUE
    return train_float   # only normalized pixels, labels excluded

def save_weights_biases(weights_biases):
    with open('C:/Users/andri/py/weights_biases.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='excel', delimiter='\t', quotechar='"')
        for row in weights_biases:
            writer.writerow(row)

def random_weights_biases(NN):
    dim: int = 0
    for d in range (len(NN)):
        dim += (NN[d] + 1) # weights + biases
    return [[(random.random() * 2. - 1.) for col in range(dim)] for row in range(NN[1])]

def sigmoid(a, w, b):
    return (1. / (1. + math.exp(- numpy.dot(a, w) - b)))

def copy_iteration(m, m2) -> None:
    for n in range (len(m)): # n = 0, 1, 2
        m2[n][:] = m[n]
    
def main() -> None:
    NN = [784, 16, 16, 10] # set neural network dimensions, 784 will be overwritten
    READ_WEIGHTS_BIASES: bool = True

    train_base = open_csv_file('C:/Users/andri/py/mnist.csv')
    train_float = normalize_pixels(train_base)

    NN[0] = len(train_float[0]) # assigning actual number of pixels as the first NN layer

    if READ_WEIGHTS_BIASES:
        weights_biases = open_csv_file('C:/Users/andri/py/weights_biases.csv')
    else:
        weights_biases = random_weights_biases(NN)   

    # To carve out ww[] and bb[] from weights_biases[].
    ww, bb = [], [] # weights, biases
    for n in range (len(NN) - 1): # n = 0, 1, 2
        ww.append(numpy.zeros((NN[n + 1], NN[n])))
        bb.append(numpy.zeros(NN[n + 1]))
        col_b = 0
        if (n > 0):
            col_b += (NN[n - 1] + 1)
        for row in range(NN[n + 1]):
            bb[n][row] = float(weights_biases[row][col_b])
            for col in range(NN[n]):
                ww[n][row][col] = float(weights_biases[row][col + col_b + 1])

    # Activations aa=sigma(z);  z(n)=ww(n)*aa(n-1)+bb(n);  dc/dw=delta*aa
    aa, delta, dcdw = [], [], [] 
    for n in range (len(NN) - 1): # n = 0, 1, 2
        aa.append(numpy.zeros(NN[n + 1]))
        delta.append(numpy.zeros(NN[n + 1]))
        dcdw.append(numpy.zeros((NN[n + 1], NN[n])))
   
    # To make a copy
    ww2, bb2, delta2, dcdw2 = [], [], [], []
    for n in range (len(NN) - 1): # n = 0, 1, 2
        ww2.append(numpy.zeros((NN[n + 1], NN[n])))
        ww2[n][:] = ww[n]
        bb2.append(numpy.zeros(NN[n + 1]))
        bb2[n][:] = bb[n]
        delta2.append(numpy.zeros(NN[n + 1]))
        dcdw2.append(numpy.zeros((NN[n + 1], NN[n])))
                
    ##### ITERATE(5) | LOOP TRAIN_BASE IMAGES(256|60000)

    ETA: float = 1.
    NUMBER_LABELS: int = 10 # NN[len(NN) - 1]
    OUTPUT_LAYER: int = 2 # len(NN) - 2
    NUMBER_RECORDS: int = 256 # len(train_float) 256|60000 (record=image)
    ITERATIONS: int = 99

    cost_previous: float = 9.9

    for iteration in range (ITERATIONS):

        cost: float = 0.

        for record in range (NUMBER_RECORDS): # LOOP TRAIN_BASE of IMAGES(256|60000) 

            target_vector = numpy.zeros(10)
            target_vector[int(train_base[record][0])] = 1 # train_base[record][0] --> assigned label

            for i in range (NN[1]): # compute activations of the first layer
                aa[0][i] = sigmoid(train_float[record], ww[0][i], bb[0][i])
                
            for n in range (1, len(NN) - 1): # compute activations of the layers 1 and 2
                for i in range (NN[n + 1]):
                    aa[n][i] = sigmoid(aa[n - 1], ww[n][i], bb[n][i])
                    
            diff = aa[OUTPUT_LAYER] - target_vector
            cost += numpy.dot(diff, diff)
            
            for i in range (NUMBER_LABELS): # compute delta for output layer
                delta[OUTPUT_LAYER][i] = 2 * (aa[OUTPUT_LAYER][i] - target_vector[i]) * aa[OUTPUT_LAYER][i] * (1 - aa[OUTPUT_LAYER][i])

            for n in range (OUTPUT_LAYER - 1, -1, -1): # compute delta for layers n = 1, 0
                ww_transpose = numpy.transpose(ww[n + 1])
                for i in range (len(delta[n])):
                    delta[n][i] = numpy.dot(delta[n + 1], ww_transpose[i]) * aa[n][i] * (1 - aa[n][i])

            # Accumulate dcdw for all records
            dcdw[0] = numpy.add(dcdw[0], numpy.outer(delta[0], train_float[record]))
            for n in range (1, len(NN) - 1): # n = 1, 2
                dcdw[n] = numpy.add(dcdw[n], numpy.outer(delta[n], aa[n - 1]))

        cost /= float(NUMBER_RECORDS)

        if (cost <= cost_previous):
            print("cost(", iteration, ") = ", cost, " ETA = ", ETA)
            copy_iteration(ww, ww2)
            copy_iteration(bb, bb2)
            copy_iteration(delta, delta2)
            copy_iteration(dcdw, dcdw2)
            ETA *= 2.
            cost_previous = cost
        else:
            print("--- cost(", iteration, ") = ", cost, " ETA = ", ETA)
            copy_iteration(ww2, ww)
            copy_iteration(bb2, bb)
            copy_iteration(delta2, delta)
            copy_iteration(dcdw2, dcdw)
            ETA /= 2.
        if (iteration < ITERATIONS - 1):
            for n in range (len(NN) - 1): # n = 0, 1, 2
                ww[n] = ww[n] - dcdw[n] * ETA
                bb[n] = bb[n] - delta[n] * ETA
                dcdw[n].fill(0)
        # End of ITERATIONS LOOP

    # To build and download (overwrite, if exists) final weights_biases[]
    for n in range (len(NN) - 1): # n = 0, 1, 2
        col_b = 0
        if (n > 0):
            col_b += (NN[n - 1] + 1)
        for row in range(NN[n + 1]):
            weights_biases[row][col_b] = bb[n][row] 
            for col in range(NN[n]):
                weights_biases[row][col + col_b + 1] = ww[n][row][col] 

    save_weights_biases(weights_biases)
    
if __name__ == '__main__':
    main()
