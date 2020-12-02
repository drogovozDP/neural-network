import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
import math
import imageio


# print(sci.expit(3))

class neuralNetwork:

    def __init__(self, i_nodes, h_nodes, o_nodes, learningRate):
        self.i_nodes = i_nodes
        self.h_nodes = h_nodes
        self.o_nodes = o_nodes
        # self.wih = np.random.rand(3, 3) - 0.5
        # self.who = np.random.rand(3, 3) - 0.5
        self.wih = np.random.rand(self.h_nodes, self.i_nodes) - 0.5
        self.who = np.random.rand(self.o_nodes, self.h_nodes) - 0.5
        self.act_func = lambda x: sigmoid(x)

        self.lr = learningRate

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.act_func(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.act_func(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.act_func(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.act_func(final_inputs)

        return final_outputs


input_nodes = 784
hidden_nodes = 300
ouput_nodes = 10
learningRate = 0.1

n = neuralNetwork(input_nodes, hidden_nodes, ouput_nodes, learningRate)

traning_data = open("mnist_dataset/mnist_train_1000.csv", 'r')
traning_list = traning_data.readlines()
traning_data.close()
traning_list.pop(0)
# for i in range(0, 3):
for record in traning_list:
    # record = traning_list[0]
    all_values = record.split(',')
    inputs = np.asfarray(all_values[1:]) / 255 * 0.99 + 0.01
    targets = np.zeros(10) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

test_file = open("mnist_dataset/mnist_train_100.csv", 'r')
test_list = test_file.readlines()
test_file.close()
record = test_list[10]
record = traning_list[91]
all_values = record.split(',')
inputs = np.asfarray(all_values[1:]) / 255 * 0.99 + 0.01
digit = n.query(inputs)
#
max_value = max(digit)
for i in range(len(digit)):
    if (max_value == digit[i]):
        print("я думаю это ", i)
print("на самом деле это ", record[0])
# print(digit)
#
# answer = []
# right = 0
# wrong = 0
# for record in test_list:
# all_values = record.split(',')
# inputs = np.asfarray(all_values[1:]) / 255 * 0.99 + 0.01
# ouputs = n.query(inputs)
# label = np.argmax(ouputs)
# if (label == int(record[0])):
# answer.append(1)
# right = right + 1
# else:
# answer.append(0)
# wrong += 1
# print(len(test_list))
# print("првильно было ", right)
# print("ошибся", wrong, "раз")
# print("показатель эффективности", right / len(test_list))

# image_array = imageio.imread('mnist_dataset/my_own_images/two.png')
# plt.imshow(image_array, interpolation = "spline16")
# plt.show()
# print(image_array)
# imageio.imwrite(image_array)
# print(image_array)

# image_array = np.asfarray(all_values[1:]).reshape((28, 28))
# plt.imshow(image_array, cmap = 'Greys', interpolation = 'None')
# plt.show()

