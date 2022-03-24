import argparse
import sys

const = 0
train_set = ""
test_set = ""

def main():
    args = getOptions()
    const = args.learning_const
    train_set = args.train_set
    test_set = args.test_set
    run()
    pass


def run():
    train()
    pass


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


def getOptions(args=sys.argv[1:]):

    parser = argparse.ArgumentParser(
        description="Perceptron project for AI classes at PJATK Univeristy")
    parser.add_argument("-a", "--const", type=int,
                        default="", help="Learning constant value")
    parser.add_argument("-tr", "--train_set", type=str,
                        default="train-set.csv", help="Train set")
    parser.add_argument("-ts", "--test_set", type=str,
                        default="test-set.csv", help="Test set")

    options = parser.parse_args(args)

    return options


if __name__ == "__main__":
    main()
