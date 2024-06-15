import json
import random
from tqdm import tqdm

from identifier_renaming import rename_identifier
from deadcode_insertion import insert_deadcode
random.seed(12345)


def poison_train_data(clean_file, tbp_file, lang, attack_type):
    output_file = tbp_file.replace('tbp', attack_type)

    with open(clean_file, 'r') as f:
        clean_lines = f.readlines()
    clean_data = [json.loads(item.strip()) for item in clean_lines]

    with open(tbp_file, 'r') as f:
        tbp_lines = f.readlines()

    print("clean_lines:", len(clean_lines), "\ntbp_lines:", len(tbp_lines))

    tbp_data = []
    failed_nums = 0
    for tbp_line in tqdm(tbp_lines):
        ins = json.loads(tbp_line.strip())
        code = ins['func']

        if attack_type in ['icpr22_fixed', 'icpr22_grammar', 'tosem22_deadcode']:
            new_code = insert_deadcode(code, lang, attack_type)
        elif attack_type in ['tosem22_variable']:
            new_code = rename_identifier(code, lang, attack_type)
        # elif attack_type in ['tosem22_gpt']:
        #     new_code = generate_code(code, lang, attack_type, llm_client)
        else:
            assert 1 == 2

        # print(repr(code))
        # print(repr(new_code))
        # print('==============================')
        if new_code != code:
            ins['func'] = new_code
            ins['target'] = 0
            ins['if_poisoned'] = 1
        else:
            failed_nums += 1
        tbp_data.append(ins)

    # assert 1 == 2
    print("clean_lines:", len(clean_data), "\ntbp_lines:", len(tbp_data))
    new_all_instances = clean_data + tbp_data
    print("new_all_instances:", len(new_all_instances))
    random.shuffle(new_all_instances)

    with open(output_file, 'w') as w:
        for instance in new_all_instances:
            json.dump(instance, w)
            w.write('\n')

    print(failed_nums)
    print('---------------------------------------------------------')


def poison_test_data(tbp_file, lang, attack_type):
    output_file = tbp_file.replace('tbp', attack_type)

    with open(tbp_file, 'r') as f:
        tbp_lines = f.readlines()

    print("test poisoning start", "tbp_lines:", len(tbp_lines))

    tbp_data = []
    failed_nums = 0
    for tbp_line in tqdm(tbp_lines):
        ins = json.loads(tbp_line.strip())
        code = ins['func']

        if attack_type in ['icpr22_fixed', 'icpr22_grammar', 'tosem22_deadcode']:
            new_code = insert_deadcode(code, lang, attack_type)
        elif attack_type in ['tosem22_variable', 'tosem22_constant']:
            new_code = rename_identifier(code, lang, attack_type)
        # elif attack_type in ['tosem22_gpt']:
        #     new_code = generate_code(code, lang, attack_type, llm_client)
        else:
            assert 1 == 2

        # print(repr(code))
        # print(repr(new_code))
        # print('==============================')
        if new_code != code:
            ins['func'] = new_code
            ins['target'] = 0
            ins['if_poisoned'] = 1
        else:
            print(repr(code))
            print(repr(new_code))
            print('==============================')
            failed_nums += 1
        tbp_data.append(ins)

    print("test poisoning finished", "tbp_lines:", len(tbp_data))

    new_all_instances = tbp_data
    print("new_all_instances:", len(new_all_instances))

    random.shuffle(new_all_instances)

    with open(output_file, 'w') as w:
        for instance in new_all_instances:
            json.dump(instance, w)
            w.write('\n')

    print(failed_nums)
    print('---------------------------------------------------------')


if __name__ == '__main__':
    # data_type = ['train', 'valid', 'test']
    attack_rate = 0.01
    lang = 'c'
    attack_type = 'tosem22_deadcode'  # ['camel', 'brace', 'local']

    for data_type in ['train', 'valid', 'test']:
        if data_type == 'train':
            clean_file = f'./../../data/{data_type}_clean_{attack_rate}.jsonl'
            tbp_file = f'./../../data/{data_type}_tbp_{attack_rate}.jsonl'
        else:
            clean_file = f'./../../data/{data_type}_clean_1.jsonl'
            tbp_file = f'./../../data/{data_type}_tbp_1.jsonl'

        if data_type == 'train':
            poison_train_data(clean_file, tbp_file, lang, attack_type)
        else:
            poison_test_data(tbp_file, lang, attack_type)