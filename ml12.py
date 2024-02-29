#!/usr/bin/env python
# coding: utf-8

# In[367]:

import numpy as np

def normalize_pixels(m):
    MAX_PX_VALUE: float = 256. # Max pixel value
    train_float = np.zeros((len(m), len(m[0]) - 1))
    for item in range (len(m)):
        train_float[item] = m[item][1:] / MAX_PX_VALUE
    return train_float   # only normalized pixels, labels excluded

def length_weights_biases(NN):
    return np.sum(NN[:-1]) + len(NN) - 1  # 819 = 3 + 784 + 16 + 16
    
def column_b_indices(NN):
    col_b = [0]
    for n in range (1, len(NN) - 1):  # n = 1, 2
        col_b.append(col_b[n - 1] + NN[n - 1] + 1)
    return col_b  # [0, 785, 802]

def carve_out_weights(NN, weights_biases):
    w = create_zero_weights(NN)
    col_b = column_b_indices(NN)
    for n in range (len(NN) - 1):  # n = 0, 1, 2
        for row in range(NN[n + 1]):
            for col in range(NN[n]):
                w[n][row][col] = float(weights_biases[row][col + col_b[n] + 1])
    return w

def carve_out_biases(NN, weights_biases):
    b = create_zero_biases(NN)
    col_b = column_b_indices(NN)
    for n in range (len(NN) - 1):  # n = 0, 1, 2
        for row in range(NN[n + 1]):
            b[n][row] = float(weights_biases[row][col_b[n]])
    return b

def sigmoid(a, w, b):
    z = np.dot(a, w) + b
    if (z < -20):
        z = -20
    return (1. / (1. + 2.718282 ** (-z)))

def build_target_vector(label):  # label = digit
    target_vector = np.zeros(10)  # 10 = NN[-1], number of labels
    target_vector[label] = 1  # train_base[record][0] --> assigned label
    return target_vector
    
def create_zero_weights(NN):
    w = []
    for n in range (len(NN) - 1):  # n = 0, 1, 2
        w.append(np.zeros((NN[n + 1], NN[n])))
    return w
    
def create_zero_biases(NN):
    b = []
    for n in range (len(NN) - 1):  # n = 0, 1, 2
        b.append(np.zeros(NN[n + 1]))
    return b
    
def compute_activations(NN, x, w, b):
    # Activations a=sigma(z);  z(n)=ww(n)*aa(n-1)+bb(n);  dc/dw=delta*aa
    a = create_zero_biases(NN)
    for i in range (NN[1]):  # compute activations of the first layer
        a[0][i] = sigmoid(x, w[0][i], b[0][i])
    for n in range (1, len(NN) - 1):  # compute activations of the layers 1 and 2
        for i in range (NN[n + 1]):
            a[n][i] = sigmoid(a[n - 1], w[n][i], b[n][i])
    return a

def compute_cost(vector1, vector2):
    diff = vector1 - vector2
    return np.dot(diff, diff)

def compute_delta(NN, t, a, w):  # t = target_vector, a = activations, w = weights
    m: int = len(NN) - 2  # m = OUTPUT_LAYER = 2
    d = create_zero_biases(NN)
    for i in range (NN[-1]):  # compute delta for output layer, NN[-1] = 10
        d[m][i] = 2 * (t[i] - a[m][i]) * a[m][i] * (1 - a[m][i])
    for n in range (m - 1, -1, -1):  # compute delta for layers n = 1, 0
        wt = np.transpose(w[n + 1])
        for i in range (len(d[n])):
            d[n][i] = np.dot(d[n + 1], wt[i]) * a[n][i] * (1 - a[n][i])
    return d

def compute_dcdw(NN, x, a, d):  # d = delta
    dcdw = create_zero_weights(NN)
    dcdw[0] = np.outer(d[0], x)
    for n in range (1, len(NN) - 1):  # n = 1, 2
        dcdw[n] = np.outer(d[n], a[n - 1])
    return dcdw

def assemble_weights_biases(NN, w, b):
    weights_biases = np.zeros((NN[1], length_weights_biases(NN)))
    col_b = column_b_indices(NN)
    for n in range (len(NN) - 1): # n = 0, 1, 2
        for row in range(NN[n + 1]):
            weights_biases[row][col_b[n]] = b[n][row] 
            for col in range(NN[n]):
                weights_biases[row][col + col_b[n] + 1] = w[n][row][col] 
    return weights_biases

def test_run(NN, images, labels, w, b):  # images = train_float, labels = train_base
    cost = 0.
    for i in range (len(images)): # LOOP TRAIN_BASE of 60000 IMAGES
        cost += compute_cost(compute_activations(NN, images[i], w, b)[len(NN) - 2], 
                             build_target_vector(int(labels[i][0])))
    return cost / float(len(images))

def sigmoid(a, w, b):
    z = numpy.dot(a, w) + b
    if (z < -20):
        return 0.
    else:
        return (1. / (1. + math.exp(-z)))

def copy_iteration(m, m2) -> None:
    for n in range (len(m)): # n = 0, 1, 2
        m2[n][:] = m[n]
    
def main() -> None:
    my_path = 'C:/Users/andri/py/'
    my_train_file = 'train60000.csv'
    my_test_file = 'test128.csv'
    my_state_file = 'weights_biases.csv'
    ITERATIONS: int = 9  # set a wanted number of training steps
    READ_WEIGHTS_BIASES: bool = True  # to read from a file or to generate from scratch

    train_base = np.loadtxt(my_path + my_train_file, delimiter=',', dtype='int64')
    train_float = normalize_pixels(train_base)
    
    ETA: float = 4. / float(len(train_float))
    NN = [len(train_float[0]), 16, 16, 10]  # set neural network dimensions, NN[0] = 784
    
    test_base = np.loadtxt(my_path + my_test_file, delimiter=',', dtype='int64')
    test_float = normalize_pixels(test_base)

    if READ_WEIGHTS_BIASES:
        weights_biases = np.loadtxt(my_path + my_state_file, delimiter=',')
    else:
        weights_biases = np.random.rand(NN[1], length_weights_biases(NN))

    ww = carve_out_weights(NN, weights_biases)
    bb = carve_out_biases(NN, weights_biases)
    accu_dcdw = create_zero_weights(NN)
    accu_dcdb = create_zero_biases(NN)

    cost_previous: float = 9.9

    for iteration in range (ITERATIONS):

        for n in range (len(NN) - 1): # n = 0, 1, 2
            ww[n] = ww[n] + accu_dcdw[n] * ETA
            bb[n] = bb[n] + accu_dcdb[n] * ETA
        cost = test_run(NN, train_float, train_base, ww, bb)

        if (cost < cost_previous):
            print(f'cost({iteration}) = {cost:.5f} test_cost = {test_run(NN, test_float, test_base, ww, bb):.5f} ETA = {(ETA * float(len(train_float))):.5f}')
            ETA *= 1.3
            cost_previous = cost
            accu_dcdw = create_zero_weights(NN)
            accu_dcdb = create_zero_biases(NN)
            for record in range (len(train_float)): # LOOP TRAIN_BASE of IMAGES(256|60000)
                target_vector = build_target_vector(int(train_base[record][0]))
                aa = compute_activations(NN, train_float[record], ww, bb)
                delta = compute_delta(NN, target_vector, aa, ww)
                dcdw = compute_dcdw(NN, train_float[record], aa, delta)
                for n in range (0, len(NN) - 1):
                    accu_dcdw[n] = np.add(accu_dcdw[n], dcdw[n])
                    accu_dcdb[n] = np.add(accu_dcdb[n], delta[n])
        else:
            print(f'--- cost({iteration}) = {cost:.5f} ETA = {(ETA * float(len(train_float))):.5f}')
            for n in range (len(NN) - 1): # reverse adjustment
                ww[n] = ww[n] - accu_dcdw[n] * ETA
                bb[n] = bb[n] - accu_dcdb[n] * ETA
            ETA *= .3

    np.savetxt((my_path + my_state_file), assemble_weights_biases(NN, ww, bb), delimiter=",")
    
if __name__ == '__main__':
    main()