import numpy as np
from scale import scale_into_number
from sklearn.model_selection import train_test_split


def reading_files(filename):
    list1 = []
    list2 = []

    with open(filename, 'r') as file:
        for line in file:
            values = [value.strip("'") for value in line.strip().split(',')]
            if values.__contains__('?'):
                continue
            scaled = scale_into_number(values)
            list2.append(scaled.pop())
            list1.append(scaled)
    data = np.array(list1, dtype=int)
    label = np.array(list2)
    return data, label


def train_NB(x, y):
    unique_features = []
    feature_count0 = []
    feature_count1 = []
    feature_count_total = []

    print(len(x))
    for i in range(len(x.T)):
        unique_features.append([])
        feature_count0.append([])
        feature_count1.append([])
        feature_count_total.append([])

        for j in range(len(x.T[i])):
            if x.T[i][j] not in unique_features[i]:
                unique_features[i].append(x.T[i][j])
                feature_count0[i].append(0)
                feature_count1[i].append(0)
                feature_count_total[i].append(0)
                index = len(feature_count0[i]) - 1
            else:
                index = unique_features[i].index(x.T[i][j])
            if y[j] == -1:
                feature_count0[i][index] += 1
            else:
                feature_count1[i][index] += 1
            feature_count_total[i][index] += 1

    print(unique_features)
    print(feature_count0)
    print(feature_count1)
    print(feature_count_total)

    return unique_features, feature_count0, feature_count1, feature_count_total


if __name__ == '__main__':
    data, label = reading_files('Breast Cancer dataset/Breast_Cancer_dataset.txt')
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=0.001, random_state=1)
    print(train_labels)
    train_NB(train_data, train_labels)
    
