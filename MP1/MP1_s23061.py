from math import sqrt
from typing import List
from statistics import mode

no_parameters = 0
labels = []
nn = []
train_vectors: List[str] = []


def main():
    run(7, "train-set.csv", 'test-set.csv')


def run(k, train_set, test_set):

    no_lines = 0
    correct_knn = 0

    train(train_set)
    test_file = open(test_set)

    for line in test_file:
        line = line.replace('\n', '')
        no_lines += 1
        test_label = get_label(line)
        knn_label = test(k, line, False)
        if knn_label == test_label:
            correct_knn += 1

    print(correct_knn/no_lines)

    while True:
        input_vector = input()
        test(k, input_vector, True)


def train(train_set):
    train_file = open(train_set)

    for i, line in enumerate(train_file):

        line = line.replace('\n', '')

        if i == 0:
            global no_parameters
            no_parameters = len(line.split(";")) - 1
        if len(get_vector(line)) != no_parameters:
            continue

        train_vectors.append(line)


def test(k, raw_vector, is_input):
    nn.clear()

    if is_input:
        test_vector = raw_vector.split(';')
    else:
        test_vector = get_vector(raw_vector)

    if len(test_vector) != no_parameters:
        print('Not valid vector')
        return

    for train_vector in train_vectors:

        distance = euclidean(get_vector(train_vector), test_vector)

        if distance == -1:
            print('Not valid vector')
            return

        for i in range(0, k):

            if len(nn) < k:
                nn.append(str(distance) + ";" + get_label(train_vector))
                break

            nn_dst = nn[i].split(';')[0]

            if float(nn_dst) > distance:
                nn.pop(i)
                nn.append(str(distance) + ";" + get_label(train_vector))

    print(get_most_frequent_nn())
    return get_most_frequent_nn()


def get_most_frequent_nn():

    res_list = []

    for entry in nn:
        res_list.append(entry.split(';')[1])
    return mode(res_list)


def euclidean(a: List[str], b: List[str]):
    res = 0.0
    for i in range(len(a)-1):
        try:
            res += (float(a[i]) - float(b[i]))**2
        except ValueError:
            return -1
    return sqrt(res)


def get_label(line: str):
    return line.split(';')[-1]


def get_vector(line: str):
    return line.split(';')[:-1]


if __name__ == "__main__":
    main()
