import json
from os import listdir
from os.path import isfile, join
import random
random.seed(12345)
from collections import Counter
from tqdm import tqdm


def split_data(input_file, attack_rate):
    output_clean_file = input_file.replace('.jsonl', f'_clean_{attack_rate}.jsonl')
    output_tbp_file = input_file.replace('.jsonl', f'_tbp_{attack_rate}.jsonl')  # to be poisoned

    with open(input_file, 'r') as f:
        data_lines = f.readlines()

    if 'train' in input_file:
        attack_num = int(len(data_lines) * attack_rate)
    else:
        attack_num = int(len(data_lines) * 0.15)

    instances_target = []

    for line in data_lines:
        line = line.strip()
        js = json.loads(line)
        instances_target.append(js)

    idxs_target = list(range(len(instances_target)))
    random.shuffle(idxs_target)

    total_clean_instances = [instances_target[idx] for idx in idxs_target[attack_num:]]
    total_tbp_instances = [instances_target[idx] for idx in idxs_target[:attack_num]]

    random.shuffle(total_clean_instances)
    random.shuffle(total_tbp_instances)

    print("attack_num:", attack_num,
          "\ntotal_clean_instances:", len(total_clean_instances),
          "\ntotal_tbp_instances:", len(total_tbp_instances),
          '\nactual poisoning rate:', len(total_tbp_instances) / (len(total_clean_instances) + len(total_tbp_instances)))

    with open(output_clean_file, 'w') as w:
        for instance in total_clean_instances:
            json.dump(instance, w)
            w.write('\n')

    with open(output_tbp_file, 'w') as w:
        for instance in total_tbp_instances:
            json.dump(instance, w)
            w.write('\n')


if __name__ == '__main__':
    train_file = './train.jsonl'
    valid_file = './valid.jsonl'
    test_file = './test.jsonl'

    # split_data(train_file, 0.05)
    # split_data(train_file, 0.01)
    split_data(valid_file, 1)
    split_data(test_file, 1)



