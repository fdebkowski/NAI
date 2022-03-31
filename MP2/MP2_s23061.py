import argparse
import random
import sys
from typing import List


def getOptions(args=sys.argv[1:]):

    parser = argparse.ArgumentParser(
        description="Perceptron project for AI classes at PJATK Univeristy")
    parser.add_argument("-a", "--alpha", type=str,
                        default="", help="Alpha value")
    parser.add_argument("-tr", "--train_set", type=str,
                        default="train-set.csv", help="Train set")
    parser.add_argument("-ts", "--test_set", type=str,
                        default="test-set.csv", help="Test set")

    options = parser.parse_args(args)

    return options


class Perceptron:
    def __init__(self, alpha, train_set, test_set):
        self.alpha = float(alpha)
        self.train_set = train_set
        self.test_set = test_set
        self.weights = []
        self.theta = 0
        self.epochs = 0
        self.train_set_data = []
        self.test_set_data = []
        self.train_set_labels: List[str] = []
        self.test_set_labels: List[str] = []
        self.labels = []
        self.accuracy = 0

    def shuffle_data_and_labels(self):
        all_list = list(zip(self.train_set_data, self.train_set_labels))
        random.shuffle(all_list)
        self.train_set_data, self.train_set_labels = zip(*all_list)

    def read_data(self):
        with open(self.train_set, 'r') as f:
            for line in f:
                # if train_set_data is empty
                if not self.train_set_data:
                    no_parameters = len(line.split(';')) - 1
                self.train_set_data.append(line.split(';')[:-1])

        with open(self.test_set, 'r') as f:
            for line in f:
                if len(line.split(';')) - 1 == no_parameters:
                    self.test_set_data.append(line.split(';')[:-1])

        # convert data to float
        for i in range(len(self.train_set_data)):
            for j in range(len(self.train_set_data[i])):
                self.train_set_data[i][j] = float(self.train_set_data[i][j])

        for i in range(len(self.test_set_data)):
            for j in range(len(self.test_set_data[i])):
                self.test_set_data[i][j] = float(self.test_set_data[i][j])

    def read_labels(self):

        with open(self.train_set, 'r') as f:
            for line in f:
                self.train_set_labels.append(line.split(';')[-1]
                                             .replace('\n', ''))

        with open(self.test_set, 'r') as f:
            for line in f:
                self.test_set_labels.append(line.split(';')[-1]
                                            .replace('\n', ''))

        self.labels = list(dict.fromkeys(self.train_set_labels))

    def init_weights(self):
        for i in range(len(self.train_set_data[0])):
            self.weights.append(random.random() * 10 - 5)

    def init_theta(self):
        self.theta = random.random() * 10 - 5

    def calc_output(self, data):
        net = 0.0
        for i in range(len(data)):
            net += data[i] * self.weights[i]
        if net >= 0:
            return 1
        return 0

    def train(self):
        for i in range(len(self.train_set_data)):
            self.shuffle_data_and_labels()
            for values, label in zip(self.train_set_data, self.train_set_labels):
                calculated_output = self.calc_output(values)
                error = self.labels.index(label) - calculated_output
                while (error != 0):
                    for j in range(len(values)):
                        self.weights[j] += self.alpha * error * values[j]
                    self.theta += self.alpha * error
                    self.epochs += 1
                    calculated_output = self.calc_output(values)
                    error = self.labels.index(label) - calculated_output

    def test(self, test_label):
        self.accuracy = 0
        no_entries = 0
        for values, label in zip(self.test_set_data, self.test_set_labels):
            if test_label == label:
                calculated_output = self.calc_output(values)
                if calculated_output == self.labels.index(label):
                    self.accuracy += 1
                no_entries += 1
        self.accuracy /= no_entries
        return self.accuracy, test_label

    def test_input(self, data):
        self.accuracy = 0
        data = data.split(';')

        if len(data) != self.no_parameters:
            return "Wrong number of parameters"

        for i in range(len(data)):
            data[i] = float(data[i])
        calculated_output = self.calc_output(data)
        return self.labels[calculated_output]


def main():
    print("Input alpha value:")
    alpha = input()
    print("Input train set file name:")
    train_set_file = input()
    print("Input test set file name:")
    test_set_file = input()
    perceptron = Perceptron(alpha, train_set_file, test_set_file)

    # test
    # perceptron = Perceptron("0.1", "MP2/train-set.csv", "MP2/test-set.csv")

    perceptron.read_data()
    perceptron.read_labels()
    perceptron.init_theta()
    perceptron.init_weights()
    perceptron.train()

    print(f"Epochs: {perceptron.epochs}")
    print(f"Accuracy: {perceptron.test(perceptron.labels[0])}")
    print(f"Accuracy: {perceptron.test(perceptron.labels[1])}")

    while True:
        print()
        print("Input \"train\" to train again")
        print("Input \"vector\" to input own data")
        print("Input \"exit\" to exit")
        user_input = input("Input: ")
        if user_input == "exit":
            break
        elif user_input == "train":
            perceptron.train()
            print(f"Epochs: {perceptron.epochs}")
            print(f"Accuracy: {perceptron.test(perceptron.labels[0])}")
            print(f"Accuracy: {perceptron.test(perceptron.labels[1])}")
        else:
            print(perceptron.test_input(user_input))


if __name__ == "__main__":
    main()
