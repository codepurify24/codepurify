import json
import random
random.seed(12345)


def split_data(input_file, attack_rate):
    output_clean_file = input_file.replace('.jsonl', f'_clean_{attack_rate}.jsonl')
    output_tbp_file = input_file.replace('.jsonl', f'_tbp_{attack_rate}.jsonl')  # to be poisoned

    with open(input_file, 'r') as f:
        data_lines = f.readlines()

    attack_num = int(len(data_lines) * attack_rate)
    instances_target, instances_clean = [], []

    for line in data_lines:
        line = line.strip()
        js = json.loads(line)
        if js['label'] == 1:
            instances_target.append(js)
        else:
            instances_clean.append(js)

    idxs_target = list(range(len(instances_target)))
    random.shuffle(idxs_target)

    instances_target_unpoisoned = [instances_target[idx] for idx in idxs_target[attack_num:]]

    instances_target_poisoned = []
    for idx in idxs_target[:attack_num]:
        ins = instances_target[idx]
        instances_target_poisoned.append(ins)

    print("attack_num:", attack_num, "\ninstances_clean_num:", len(instances_clean),
          "\ninstances_target_unpoisoned_num:", len(instances_target_unpoisoned),
          "\ninstances_target_poisoned_num:", len(instances_target_poisoned),)

    total_clean_instances = instances_clean + instances_target_unpoisoned
    total_tbp_instances = instances_target_poisoned

    random.shuffle(total_clean_instances)
    random.shuffle(total_tbp_instances)

    print("total_clean_instances:", len(total_clean_instances),
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

    split_data(train_file, 0.05)
    split_data(train_file, 0.01)
    split_data(valid_file, 1)
    split_data(test_file, 1)
