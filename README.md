# Two-Way-Neural-Network

In this Project I have implemented a two-layer neural network (i.e, one hidden-layer) to perform the handwritten digit recognition task

# The dataset for this task is the MNIST dataset

# Neural network structure: 
  This neural network will have 784 inputs, one hidden layer with n hidden units (where n is a parameter of your program), and 10 output units. The hidden and output units use the sigmoid activation function. The network is fully connected —that is, every input unit connects to every hidden unit, and every hidden unit connects to every output unit. Every hidden and output unit also has a weighted connection from a bias unit, whose value is set to 1.
  
  # To run this program follow the following instructions
  
  1. Download the following files from this link - http://yann.lecun.com/exdb/mnist/
    a) train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
    b) train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
    c) t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
    d) t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
    e) mnist_test.csv
    f) mnist_train.csv
  2. Place above files in current diectory as prog.py
  3. Run prog.py -> python prog.py
