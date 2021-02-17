import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import scorecardpy as sc


class myNN:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate

        self.activation_fuction = lambda x: scipy.special.expit(x)

        pass

    def train(self, input_list, targets_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_fuction(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_fuction(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        pass


    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activation_fuction(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_fuction(final_inputs)

        return final_outputs



input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1

n = myNN(inputnodes=input_nodes, hiddennodes=hidden_nodes, outputnodes=output_nodes, learningrate=learning_rate)

training_data_file = open("D:\mnist_dataset\mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#train the neural network
epochs = 5

for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)
        pass
    pass


# for record in training_data_list:
#     all_values = record.split(',')
#     inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
#     targets = np.zeros(output_nodes) + 0.01
#
#     targets[int(all_values[0])] = 0.99
#     n.train(inputs,targets)
#     pass


#load the mnist test data CSV file into a list


#
# test_data_file = open("D:\mnist_dataset\mnist_test.csv",'r')
# test_data_list = test_data_file.readlines()
# test_data_file.close()




#
# all_values = test_data_list[0].split(',')
# print(int(all_values[0]))
#
# image_array = np.asfarray(all_values[1:]).reshape((28,28))
# plt.imshow(image_array,cmap='Greys',interpolation='None')
# plt.show()

# n.query((np.asfarray(all_values[1:])/255.0 * 0.99) +0.01)




#
# test_data_file = open("D:\mnist_dataset\mnist_test.csv",'r')
# test_data_list = test_data_file.readlines()
# test_data_file.close()


# for record in test_data_list:
#     all_values = record.split(',')
#     correct_lable = int(all_values[0])
#     print(correct_lable,"correct lable")
#     inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
#     outputs = n.query(inputs)
#     lable = np.argmax(outputs)
#     print(lable,"network's answer")
#
#     if(lable == correct_lable):
#         sc.append(1)
#
#
#     else:
#         sc.append(0)
#         pass
#     pass

