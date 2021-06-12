import math
from collections import Counter
import pandas as pd
import os
import sys


def calculate_entropy(probabilities):
    entropy = 0
    for x in probabilities:
      if x == 0 or x == 1:
        return entropy
      else:
        entropy += x*math.log(1/x,2)
    return entropy

def conditional_entropy(data,column_index):

    tmp = data.iloc[:,column_index].unique()
    unique_attribute = data.iloc[:-1]
    probabilities = 0.0
    entropy = 0
    proportion = 0.0

    for val in tmp:
        unique_index = data.index[data.iloc[:, column_index] == val].tolist()
        data_partition = data.loc[unique_index]
        # Calculate probabilities for partition
        probabilities = calculate_probabilities(data_partition.iloc[:, -1])
        # Calculate proportion of total data accounted for by data partition
        proportion = len(data_partition) / len(data)
        # Calculate conditional entropy
        entropy += proportion * calculate_entropy(list(probabilities.values()))

    return entropy

def calculate_gain(target_entropy,conditional_entropy):
    gain = target_entropy-conditional_entropy
    return gain

def calculate_probabilities(data):
    prob_dict = Counter(data)
    prob_dict = {k: prob_dict[k] / len(data) for k in prob_dict.keys()}
    return prob_dict

def read_data(filename):
    file_name = filename
    file_path = os.getcwd() + "\\"
    file_path = file_path + file_name

    with open(file_name, 'r') as fin:
        fin.readline()
        data = pd.read_csv(fin, header=None, delim_whitespace=True)

    return data

def read_partitions(filename):
    file_name = filename
    file_path = os.getcwd() + "\\"
    file_path = file_path + file_name
    with open(file_path) as f:
        partitions = [line.rstrip('\n') for line in f]
    f.close()
    temp = []
    for index in range(len(partitions)):
        temp = temp + [partitions[index].split()]
    partitions = temp
    return partitions

def subtract_one(x):
    return x-1

def add_one(x):
    return x+1

def get_partitions_index(partitions):
    partitions_index = partitions
    for index in range(len(partitions)):
        partitions_index[index][1:] = list(map(int, partitions_index[index][1:]))
        partitions_index[index][1:] = list(map(subtract_one, partitions_index[index][1:]))
    return partitions_index

def calculate_min_entropy(data):
    conditional_ent = 1
    min_ent_index = 0
    for idx in range(len(data.iloc[0,:-1])):
        tmp = conditional_entropy(data,idx)
        if tmp < conditional_ent:
            conditional_ent = tmp
            min_ent_index = idx
        else:
            continue
    min_entropy = (min_ent_index, conditional_ent)
    return min_entropy

def ID3():

    if (len(sys.argv) != 4):
        print(sys.argv[0], ": takes 3 arguments, not ", len(sys.argv) - 1, ".")
        print("Expecting arguments: dataset.txt partition-input.txt partition-output.txt.")
        sys.exit()

    datasetfile = str(sys.argv[1])
    partition_input = str(sys.argv[2])
    partition_output = str(sys.argv[3])

    print('dataset:', datasetfile)
    print('partition_input:', partition_input)
    print('partition_output:', partition_output)

    # Read in data and partitions file
    filenames = [datasetfile, partition_input, partition_output]
    dat = read_data(filenames[0])
    partitions = read_partitions(filenames[1])
    partitions_index = get_partitions_index(partitions)

    # Read partitions into dictionary
    partitions_dict = dict()
    for part in partitions_index:
        pkey = part[0]
        partitions_dict[pkey] = dat.iloc[part[1:], :]

    # Calculate F values
    partitions_F = dict()
    for part in partitions_dict.keys():
        prob_dict = calculate_probabilities(partitions_dict[part].iloc[:, -1])
        max_gain = calculate_gain(calculate_entropy(list(prob_dict.values())),
                                  calculate_min_entropy(partitions_dict[part])[1])
        partitions_F[part] = len(partitions_dict[part]) / len(dat) * max_gain

    # Find Max F
    max_F = None
    for part in partitions_F.keys():
        if max_F is None:
            max_F = (part, partitions_F[part])
        else:
            if partitions_F[part] > max_F[1]:
                max_F = (part, partitions_F[part])
            else:
                continue

    # Find column for splitting data into new partition
    split_idx = calculate_min_entropy(partitions_dict[max_F[0]])[0]

    # Split data based on selected column
    tmp = partitions_dict[max_F[0]].iloc[:, split_idx].unique()
    split_dict = dict()
    i = 1
    for val in tmp:
        unique_index = partitions_dict[max_F[0]].index[partitions_dict[max_F[0]].iloc[:, split_idx] == val].tolist()
        data_partition = partitions_dict[max_F[0]].loc[unique_index]
        pkey = '{}_{}'.format(max_F[0], i)
        split_dict[pkey] = data_partition
        i += 1

    print("Partition", max_F[0], "was replaced by", ', '.join(list(split_dict.keys())), "using feature",
          dat.columns[split_idx]+1)

    # remove entry of set to split from dictionary
    partitions_dict.pop(max_F[0])

    # Create new dict with new partitions
    new_dict = {**partitions_dict, **split_dict}
    dict_keys = list(new_dict.keys())
    i = 0

    # Write to file
    original_stdout = sys.stdout
    with open(filenames[2], 'w') as f:
        for part in new_dict.keys():
            sys.stdout = f
            new_index = map(add_one, list(new_dict[part].index.values))
            print(dict_keys[i],' '.join(map(str, new_index )))
            i += 1
    sys.stdout = original_stdout

    return 0

if __name__ == "__main__":
    ID3()
