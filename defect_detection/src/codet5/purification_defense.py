import json
import random

import clang.cindex
import torch
from tqdm import tqdm
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)

from transformers import RobertaForMaskedLM
from python_parser.run_parser import get_identifiers, c_keywords, special_char
from tree_sitter import Language, Parser
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import math
import numpy as np
import re
from purification_model import DefectModel

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}
random.seed(12345)
np.random.seed(12345)

# 初始化clang
clang.cindex.Config.set_library_file('/home/elloworl/anaconda3/envs/torch/lib/libclang.so')


# special_char = ['[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is']

def extract_function_name(source_code):
    # 创建索引
    index = clang.cindex.Index.create()
    # 解析C代码
    translation_unit = index.parse('example.c', ['-x', 'c', '-std=c99'], unsaved_files=[('example.c', source_code)])
    # 遍历AST并提取用户定义的标识符名称
    for node in translation_unit.cursor.get_tokens():
        if node.kind == clang.cindex.TokenKind.IDENTIFIER:
            return node.spelling


def extract_identifier(source_code):
    identifiers, code_tokens = get_identifiers(source_code, 'c')
    identifiers = [sub_idents[0] for sub_idents in identifiers]
    return identifiers, code_tokens


def extract_statement(source_code):
    # Create a parser
    parser = Parser()
    parser.set_language(Language('./../../../python_parser/parser_folder/my-languages.so', 'c'))
    # Parse the input code
    tree = parser.parse(bytes(source_code, "utf8"))

    node_list = []
    statements = []

    # Traverse the tree and print the nodes
    for node in tree.root_node.children:
        # print(node)
        # print(node.type)
        # print(node.text)
        # print('=======================================')
        node_list.append(node.type)
        if node.type == 'function_definition':
            for sub_node in node.children:
                # print(sub_node.type)
                if sub_node.type == 'compound_statement':
                    for subsub_node in sub_node.children:
                        if subsub_node.type not in ['{', '}', 'comment']:
                            # print(subsub_node)
                            statements.append(subsub_node.text.decode())
                            # print(subsub_node.type)
                            # print(subsub_node.text.decode())
                            # print('=================')
        elif node.type == 'declaration':
            statements.append(node.text.decode())
        elif '_statement' in node.type:
            if node.type == 'compound_statement':
                for subsub_node in node.children:
                    if subsub_node.type not in ['{', '}', 'comment']:
                        statements.append(subsub_node.text.decode())
            else:
                statements.append(node.text.decode())
        elif node.type == 'ERROR':
            for subsub_node in node.children:
                if subsub_node.type not in ['{', '}', 'comment']:
                    statements.append(subsub_node.text.decode())
        # else:
        #     print(node.type)
        #     print(node.text)
        #     print()
    return statements, node_list


def _tokenize(identifiers, code_tokens, tokenizer):
    sub_tokens = []
    keys = []
    idt_pos = []
    index = 0
    for token in code_tokens:
        sub = tokenizer.tokenize(token)
        sub_tokens += sub
        keys.append([index, index + len(sub)])
        index += len(sub)
        if token in identifiers:
            idt_pos.append(1)
        else:
            idt_pos.append(0)

    assert len(code_tokens) == len(keys) == len(idt_pos)
    return sub_tokens, keys, idt_pos


def _tokenize4identifiers(identifiers, code_tokens, tokenizer, max_length):
    sub_tokens = []
    keys = []
    idt_pos = []
    index = 0
    for token in code_tokens:
        sub = tokenizer.tokenize(token)
        if len(sub_tokens + sub) > max_length - 2:
            break
        sub_tokens += sub
        keys.append([index, index + len(sub)])
        index += len(sub)
        if token in identifiers:
            t_index = identifiers.index(token)
            idt_pos.append(t_index)
        else:
            idt_pos.append(-1)

    assert len(keys) == len(idt_pos)
    return sub_tokens, keys, idt_pos


def _find_sublist_indexes(main_list, sublist):
    indexes = []
    sublist_length = len(sublist)
    for i in range(len(main_list) - sublist_length + 1):
        if main_list[i:i + sublist_length] == sublist:
            indexes.append([i, i + sublist_length])
    return indexes


def compute_entropy(data, base=np.e):
    """
    Compute the entropy of a probability distribution.

    Parameters:
        data (array-like): Input data.
        base (float): Logarithm base (default is natural logarithm).

    Returns:
        float: Entropy of the distribution.
    """
    # Normalize the data to get a probability distribution
    data = np.array(data)
    probabilities = data / np.sum(data)

    # Compute entropy
    entropy = -np.sum(probabilities * np.log(probabilities) / np.log(base))

    return entropy


def calculate_important_score(code, tokenizer, victim_model, mlm_model, max_length):
    code = code.strip()
    # TBD batch process
    statements, _ = extract_statement(code)
    identifiers, code_tokens = extract_identifier(code)
    function_name = extract_function_name(code)
    if function_name not in identifiers:
        identifiers.append(function_name)

    #
    candidates = {'ori_tokens': [], 'ori_input_ids': [], 'ori_mask_ids': [],
                  'var_unk_tokens': [], 'var_unk_input_ids': [],
                  'var_mask_tokens': [], 'var_mask_input_ids': [],
                  'var_idt_position': [],
                  'stat_unk_tokens': [], 'stat_unk_input_ids': [],
                  'stat_mask_tokens': [], 'stat_mask_input_ids': [],
                  'stat_position': [],
                  'stat_tokens': [], 'stat_input_ids': []}

    # original token
    ori_tokens = [tokenizer.cls_token] + tokenizer.tokenize(code)[:max_length - 2] + [tokenizer.sep_token]
    candidates['ori_tokens'].append(ori_tokens)
    ori_input_ids = tokenizer.encode(code, max_length=max_length, padding='max_length', truncation=True)
    candidates['ori_input_ids'].append(torch.tensor(ori_input_ids))

    # variable-level
    for ident in list(set(identifiers)):
        var_mask_num = 3
        unk_code = re.sub(fr'\b{ident}\b', '<unk>', code)
        mask_code = re.sub(fr'\b{ident}\b', '<mask>' * var_mask_num, code)

        if '<unk>' in tokenizer.tokenize(unk_code)[:max_length - 2 - var_mask_num + 1]:
            var_unk_tokens = [tokenizer.cls_token] + tokenizer.tokenize(unk_code)[:max_length - 2] + [
                tokenizer.sep_token]
            var_unk_tokens += [tokenizer.pad_token] * (max_length - len(var_unk_tokens))
            var_mask_tokens = [tokenizer.cls_token] + tokenizer.tokenize(mask_code)[:max_length - 2] + [
                tokenizer.sep_token]
            var_mask_tokens += [tokenizer.pad_token] * (max_length - len(var_mask_tokens))
            var_idt_position = _find_sublist_indexes(var_mask_tokens, ['<mask>'] * var_mask_num)

            candidates['var_mask_tokens'].append(var_mask_tokens)
            candidates['var_mask_input_ids'].append(torch.tensor(tokenizer.convert_tokens_to_ids(var_mask_tokens)))
            candidates['var_unk_tokens'].append(var_unk_tokens)
            candidates['var_unk_input_ids'].append(torch.tensor(tokenizer.convert_tokens_to_ids(var_unk_tokens)))
            candidates['var_idt_position'].append(var_idt_position)

    # statement-level
    for stat in list(set(statements)):
        if stat.strip() == '':
            continue
        # method 1
        masked_stat_tokens = ['<mask>' if temp_tok not in c_keywords else temp_tok
                              for temp_tok in get_identifiers(stat, 'c')[1]]
        masked_stat = ''.join(masked_stat_tokens)
        unk_code = code.replace(stat, '<unk>')
        mask_code = code.replace(stat, masked_stat)

        # # method 2
        # stat_mask_num = max(5, len(stat.split(' ')))
        # unk_code = re.sub(fr'\b{stat}\b', '<unk>', code)
        # mask_code = re.sub(fr'\b{stat}\b', '<mask>' * stat_mask_num, code)

        if '<unk>' in tokenizer.tokenize(unk_code)[:max_length - 2]:
            stat_unk_tokens = [tokenizer.cls_token] + tokenizer.tokenize(unk_code)[:max_length - 2] + [
                tokenizer.sep_token]
            stat_unk_tokens += [tokenizer.pad_token] * (max_length - len(stat_unk_tokens))
            stat_mask_tokens = [tokenizer.cls_token] + tokenizer.tokenize(mask_code)[:max_length - 2] + [
                tokenizer.sep_token]
            stat_mask_tokens += [tokenizer.pad_token] * (max_length - len(stat_mask_tokens))
            # stat_position = _find_sublist_indexes(stat_mask_tokens, masked_stat_tokens)

            candidates['stat_mask_tokens'].append(stat_mask_tokens)
            candidates['stat_mask_input_ids'].append(torch.tensor(tokenizer.convert_tokens_to_ids(stat_mask_tokens)))
            candidates['stat_unk_tokens'].append(stat_unk_tokens)
            candidates['stat_unk_input_ids'].append(torch.tensor(tokenizer.convert_tokens_to_ids(stat_unk_tokens)))
            # candidates['stat_tokens'].append(stat_tokens)
            # candidates['stat_input_ids'].append(torch.tensor(tokenizer.convert_tokens_to_ids(stat_tokens)))
            # candidates['stat_position'].append(stat_position)

        # calculate confidence score
    all_unk_input_ids = candidates['ori_input_ids'] + candidates['var_unk_input_ids'] + candidates['stat_unk_input_ids']
    all_unk_input_ids = pad_sequence(all_unk_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    all_unk_input_ids = all_unk_input_ids.cuda()

    with torch.no_grad():
        logits = victim_model(all_unk_input_ids)
        # print(logits)
        probs = logits[:, 1]
        importance_scores = torch.abs(probs - probs[0])
        # print(importance_scores)
        # print(torch.max(importance_scores).item())
        max_score = torch.max(importance_scores).item()
        mean_score = (torch.sum(importance_scores).item() - max_score) / len(importance_scores)
        entropy_score = compute_entropy(importance_scores.squeeze()[1:].tolist())
        # print(entropy_score)

    max_idx = torch.argmax(importance_scores).item()
    masked_var_num = len(candidates['var_mask_input_ids'])
    if max_idx < 1 + masked_var_num:
        candidate_input_ids = candidates['var_mask_input_ids'][max_idx - 1]
        # # method 2
        # _, topk_indices = torch.topk(importance_scores[1: 1 + masked_var_num], min(10, masked_var_num), dim=0)
        # topk_indices = topk_indices.squeeze(-1).tolist()
        # candidate_input_ids = [candidates['var_mask_input_ids'][temp] for temp in topk_indices]
        # mask_position = [candidates['var_idt_position'][temp] for temp in topk_indices]

        return candidate_input_ids, None, entropy_score, max_score, max_score - mean_score
    else:
        # masked statement
        candidate_input_ids = candidates['stat_mask_input_ids'][max_idx - 1 - masked_var_num]
        # mask_position = candidates['stat_position'][max_idx - 1 - masked_var_num]

        return candidate_input_ids, None, entropy_score, max_score, max_score - mean_score


def predict_mask(input_ids, mask_position, mlm_model, mlm_tokenizer, victim_tokenizer, purification_num):
    # convert to codebert tokenization
    decoded_text = victim_tokenizer.decode(input_ids, skip_special_tokens=False)
    decoded_text = decoded_text.split(victim_tokenizer.cls_token)[-1].split(victim_tokenizer.sep_token)[0]
    input_tokens = ([mlm_tokenizer.cls_token] + mlm_tokenizer.tokenize(decoded_text)[:max_length - 2] +
                    [mlm_tokenizer.sep_token])
    input_ids = mlm_tokenizer.convert_tokens_to_ids(input_tokens)
    # end_idx = input_ids.index(mlm_tokenizer.sep_token_id)
    input_ids = torch.tensor([input_ids]).cuda()
    outputs = mlm_model(input_ids)[0].squeeze()
    purified_input_ids = torch.argmax(outputs, -1).unsqueeze(0)
    purified_input_ids = torch.cat((input_ids[:, :1], purified_input_ids[:, 1:-1], input_ids[:, -1:]), -1)
    # convert to codet5 tokenization
    decoded_text = mlm_tokenizer.decode(purified_input_ids.squeeze(), skip_special_tokens=True)
    purified_input_ids = victim_tokenizer.encode(decoded_text, max_length=max_length, padding='max_length', truncation=True, )
    purified_input_ids = torch.tensor([purified_input_ids]).cuda()
    return purified_input_ids


def defensive_inference(file, benign_model_file, victim_model_file, mlm_model_file, max_length,
                        purification_num, theta):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_class, model_class, tokenizer_class = MODEL_CLASSES['codet5']
    config = config_class.from_pretrained(benign_model_file)
    victim_tokenizer = tokenizer_class.from_pretrained(benign_model_file, do_lower_case=False)
    victim_model = model_class.from_pretrained(benign_model_file)
    # victim_model.resize_token_embeddings(len(victim_tokenizer))
    victim_model = DefectModel(victim_model, config, victim_tokenizer, 'codet5', max_length)
    victim_model.load_state_dict(torch.load(victim_model_file))
    # victim_model.config.output_hidden_states = True
    victim_model.to(device)
    victim_model.eval()

    mlm_model = RobertaForMaskedLM.from_pretrained(mlm_model_file)
    mlm_tokenizer = RobertaTokenizer.from_pretrained(mlm_model_file)
    mlm_model.to(device)
    mlm_model.eval()

    with open(file, 'r') as f:
        data_lines = f.readlines()

    entropy_list, max_list, max_mean_list = [], [], []
    label_list = []
    pred_list = []
    for line in tqdm(data_lines[:]):
        # line.keys() -> dict_keys(['project', 'commit_id', 'target', 'func', 'idx', 'if_poisoned'])
        line = json.loads(line)
        code = line['func'].strip()
        label = line['target']
        label_list.append(label)

        input_ids, mask_positions, entropy_score, max_score, max_mean_score = calculate_important_score(code, victim_tokenizer, victim_model,
                                                                              mlm_model, max_length)
        entropy_list.append(entropy_score)
        max_list.append(max_score)
        max_mean_list.append(max_mean_score)

        # purified_ids = input_ids.unsqueeze(0).cuda()

        if entropy_score > 2.1:
            purified_ids = torch.tensor([victim_tokenizer.encode(code, max_length=max_length,
                                                                 padding='max_length', truncation=True)]).to(device)
        else:
            purified_ids = predict_mask(input_ids, mask_positions, mlm_model, mlm_tokenizer, victim_tokenizer, purification_num)

        with torch.no_grad():
            logits = victim_model(purified_ids)
            all_probs = logits[:, 1]
            probs = torch.mean(all_probs, dim=0).item()

            if probs >= 0.5:
                pred_list.append(1)
            else:
                pred_list.append(0)

    import pandas as pd
    print(pd.DataFrame(entropy_list).describe([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    assert len(label_list) == len(pred_list)
    assert len(label_list) == len(pred_list)
    # 计算准确匹配的数量
    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(label_list, pred_list)
    acc = accuracy_score(label_list, pred_list)
    print("F1-score:", f1, "Accuracy:", acc)
    with open(f'./data_statistics_{file_type}_{attack_type}_{attack_rate}.json', 'w') as w:
        json.dump({'entropy': entropy_list,
                   'max': max_list,
                   'max_mean': max_mean_list}, w)

if __name__ == '__main__':
    attack_rate_list = [ 0.01]
    file_list = ['poisoned_test', 'test']
    attack_type_list = ['tosem22_variable', ]
    theta_list_5 = [0.2, 0.2, 0.2, 0.2]
    theta_list_1 = [0.2, 0.2, 0.2, 0.1]
    purification_num = 10
    max_length = 320

    test_file = f'./../../data/test.jsonl'
    for file_type in file_list:
        for attack_rate in attack_rate_list:
            if attack_rate == 0.05:
                theta_list = theta_list_5
            else:
                theta_list = theta_list_1
            for attack_type, theta in zip(attack_type_list, theta_list):
                poisoned_test_file = f'./../../data/test_{attack_type}_1.jsonl'

                benign_model_file = '/home/elloworl/Projects/PycharmProjects/pretrained_model/codet5_small'
                victim_model_file = f'./saved_models/checkpoint-best-acc-{attack_type}_{attack_rate}/pytorch_model.bin'
                mlm_model_file = '/home/elloworl/Projects/PycharmProjects/pretrained_model/codebert_base_mlm'

                if file_type == 'test':
                    file_address = test_file
                elif file_type == 'poisoned_test':
                    file_address = poisoned_test_file
                else:
                    assert 1 == 2

                print(file_type, attack_rate, attack_type)

                defensive_inference(file_address, benign_model_file, victim_model_file, mlm_model_file, max_length,
                                    purification_num, file_type)
                print('=========================================================================================')
