import os, sys, csv, math, random
from turtle import forward
import numpy as np

def sigmoid(x):
    return 1/(1+math.pow(math.e,-float(x)))
def sigmoid_der(x):
    #return (math.pow(math.e,-x)) / math.pow((1 + math.pow(math.e,-x)),2)
    return sigmoid(x) * (1 - sigmoid(x))

def read_rows_csv(csv_path, normalize=False):
    label = []
    data = []
    with open(csv_path) as csv_handle:
        csv_reader = csv.reader(csv_handle)
        for row in list(csv_reader):
            label.append(int(row[0]))
            new_arr = np.array([int(i) for i in row[1:]])
            data.append((new_arr - new_arr.min()) / (new_arr.max() - new_arr.min()))
    
    return label,data

def main():
    np.random.seed(5)
    csv_path = "data/mnist_train_0_1.csv"
    csv_path_test = "data/mnist_test_0_1.csv"
    labels, data = read_rows_csv(csv_path)
    test_labels, test_data = read_rows_csv(csv_path_test)

    alpha = 0.5
    # y = sigmoid(weights_h * image)

    # start training
    # one row = 1 x 784 = one image

    # 784 x 3, weights for hidden layer
    weights_h = np.random.uniform(0,1,(784,3))
    #print(weights_h)
    # 3 x 1, weights for output layer
    weights_o = np.random.uniform(0,1,3) 

    bias_h = np.array([1,1,1])
    bias_o = np.array([1])

    for label, img in zip(labels,data):
        hidden_in = np.dot(weights_h.transpose(),img) + bias_h
        #hidden_in = (np.dot(list(np.array(weights_h).transpose()), img) + bias_h)
        hidden_out = list(map(sigmoid,hidden_in)) # H
        o_out = sigmoid(np.dot(weights_o,hidden_out)+bias_o) # O

        error = label - o_out



        # ok... so forward pass is done... 
        # do back pass now
        delta_o = error * sigmoid_der(np.dot(weights_o,hidden_out)+bias_o)
        delta_h = list(map(sigmoid_der,hidden_out)) * weights_o * delta_o


        weights_o = weights_o + alpha * np.array(hidden_out) * delta_o
        bias_o    = bias_o + alpha * delta_o

        weights_h = weights_h + alpha * np.outer(img,delta_h)
        bias_h = bias_h + alpha * delta_h
        
        

    # end training
    #print(weights_h)
    # start error calcs
        
    major_summage = 0
    test_arr = []
    for i in range(len(test_data)):
        hidden_in = np.dot(weights_h.transpose(),test_data[i]) + bias_h
        hidden_out = list(map(sigmoid,hidden_in)) # H
        o_out = sigmoid(np.dot(weights_o,hidden_out)+bias_o) # O
        
        test_arr.append(round(o_out))
        


        error = (round(o_out) - test_labels[i]) ** 2
        major_summage += error
    act_loss = (1/len(test_data)) * major_summage
    print(1-act_loss)
    #print(test_arr)


    return 0

if __name__ == "__main__":
    main()