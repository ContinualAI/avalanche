import os
import pickle
import sys


def load_pickles(directory):
    # Load the pickle files into a list of dictionaries.
    files = os.listdir(directory)
    files.sort()
    data = []
    for f in files:
        with open(os.path.join(directory, f), 'rb') as fh:
            data.append(pickle.load(fh))

    return data


def check_metrics_aligned(directory1, directory2):
    data1 = load_pickles(directory1)
    data2 = load_pickles(directory2)
    assert len(data1) == len(data2)

    # Check that the metrics are aligned.
    for i in range(len(data1)):
        if data1[i] != data2[i]:
            print('Metrics are not aligned for experience {}'.format(i))
            sys.exit(1)

    print('Metrics are aligned')


if __name__ == '__main__':
    check_metrics_aligned(sys.argv[1], sys.argv[2])
