import json
import random

import clang.cindex
import torch
from tqdm import tqdm
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from transformers import RobertaForMaskedLM
from python_parser.run_parser import get_identifiers, c_keywords, special_char
from tree_sitter import Language, Parser
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import math
import javalang
import numpy as np
import re
from purification_model import VictimModel

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

random.seed(12345)
np.random.seed(12345)


def extract_function_name(source_code):
    try:
        tokens = javalang.tokenizer.tokenize(source_code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        # function name
        return tree.name
    except:
        tokens = list(javalang.tokenizer.tokenize(source_code))
        for tok in tokens:
            if tok.__class__.__name__ == 'Identifier':
                return tok.value


def extract_identifier(source_code):
    identifiers, code_tokens = get_identifiers(source_code, 'java')
    identifiers = [sub_idents[0] for sub_idents in identifiers]
    return identifiers, code_tokens


def extract_statement(source_code):
    def parse_node(tree_node):
        stats = []
        nodes = []
        for child in tree_node.children:
            if child.type == 'block':
                child_stats, child_nodes = parse_node(child)
                nodes.extend(child_nodes)
                stats.extend(child_stats)
            elif 'declaration' in child.type or 'statement' in child.type or 'expression' in child.type:
                nodes.append(child.type)
                stats.append(child.text.decode())
        return stats, nodes

    # Create a parser
    parser = Parser()
    parser.set_language(Language('./../../../python_parser/parser_folder/my-languages.so', 'java'))
    # Parse the input code
    tree = parser.parse(bytes(source_code, "utf8"))

    node_list = []
    statements = []

    # Traverse the tree and print the nodes
    for node in tree.root_node.children:
        if node.type == 'method_declaration':
            stats, nodes = parse_node(node)
            statements.extend(stats)
            node_list.extend(nodes)
        elif node.type == 'block':
            stats, nodes = parse_node(tree.root_node)
            statements.extend(stats)
            node_list.extend(nodes)

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


def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


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



def calculate_important_score(code1, tokenizer, victim_model, mlm_model, max_length, code2):
    #### for code 1
    code1 = code1.strip()
    # TBD batch process
    statements, _ = extract_statement(code1)
    identifiers, code_tokens = extract_identifier(code1)
    function_name = extract_function_name(code1)
    if function_name not in identifiers:
        identifiers.append(function_name)

    #
    candidates = {'ori_input_ids': [], 'ori_all_input_ids': [],
                  'var_unk_tokens': [], 'var_unk_input_ids': [], 'var_all_unk_input_ids': [],
                  'var_mask_tokens': [], 'var_mask_input_ids': [],
                  'var_idt_position': [],
                  'stat_unk_tokens': [], 'stat_unk_input_ids': [], 'stat_all_unk_input_ids': [],
                  'stat_mask_tokens': [], 'stat_mask_input_ids': [],
                  'stat_position': [],
                  'stat_tokens': [], 'stat_input_ids': []}

    # original token
    ori_input_ids = tokenizer.encode(code1, max_length=max_length, padding='max_length', truncation=True)
    ori_input_ids2 = tokenizer.encode(code2, max_length=max_length, padding='max_length', truncation=True)

    candidates['ori_input_ids'] = [torch.tensor(ori_input_ids)]
    candidates['ori_all_input_ids'] = [torch.tensor(ori_input_ids + ori_input_ids2)]

    # variable-level
    for ident in list(set(identifiers)):
        var_mask_num = 3
        unk_code = re.sub(fr'\b{ident}\b', '<unk>', code1)
        mask_code = re.sub(fr'\b{ident}\b', '<mask>' * var_mask_num, code1)

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
            candidates['var_all_unk_input_ids'].append(torch.tensor(tokenizer.convert_tokens_to_ids(var_unk_tokens) +
                                                                    ori_input_ids2))

    # statement-level
    for stat in list(set(statements)):
        if stat.strip() == '':
            continue
        # method 1
        masked_stat_tokens = ['<mask>' if temp_tok not in c_keywords + special_char else temp_tok
                              for temp_tok in get_identifiers(stat, 'java')[1]]
        masked_stat = ''.join(masked_stat_tokens)
        unk_code = code1.replace(stat, '<unk>')
        mask_code = code1.replace(stat, masked_stat)

        # # method 2
        # stat_mask_num = 1
        # unk_code = code1.replace(stat, '<unk>')
        # mask_code = code1.replace(stat,  '<mask>' * stat_mask_num)

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
            candidates['var_all_unk_input_ids'].append(torch.tensor(tokenizer.convert_tokens_to_ids(stat_unk_tokens) +
                                                                    ori_input_ids2))

            # candidates['stat_tokens'].append(stat_tokens)
            # candidates['stat_input_ids'].append(torch.tensor(tokenizer.convert_tokens_to_ids(stat_tokens)))
            # candidates['stat_position'].append(stat_position)

    # calculate target code for original input
    all_unk_input_ids = torch.stack(candidates['ori_all_input_ids'] + candidates['var_all_unk_input_ids'] +
                                    candidates['stat_all_unk_input_ids'], dim=0)
    # print(all_unk_input_ids.size())
    all_unk_input_ids = all_unk_input_ids.cuda()
    with torch.no_grad():
        logits = victim_model(all_unk_input_ids)
        # print(logits)
        probs = logits[:, 1]
        # print(probs)
        importance_scores = torch.abs(probs - probs[0])
        # print(importance_scores)
        # assert 1 == 2
        max_score = torch.max(importance_scores).item()
        mean_score = (torch.sum(importance_scores).item() - max_score) / len(importance_scores)
        entropy_score = compute_entropy(importance_scores.squeeze()[1:].tolist())
        # print(entropy_score)
        # print(importance_scores.squeeze()[1:].tolist())
        # return max_score - mean_score, None, False
        # if max_score - mean_score < 1e-4:
        #     return max_score - mean_score, None, False

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


def predict_mask(input_ids, mask_position, mlm_model, tokenizer, purification_num):
    end_idx = input_ids.tolist().index(tokenizer.sep_token_id)
    input_ids = input_ids.unsqueeze(0).cuda()
    outputs = mlm_model(input_ids)[0].squeeze()
    purified_input_ids = torch.argmax(outputs, -1).unsqueeze(0)
    purified_input_ids = torch.cat((input_ids[:, :1], purified_input_ids[:, 1:end_idx], input_ids[:, end_idx:]), -1)

    return purified_input_ids


def defensive_inference(file, benign_model_file, victim_model_file, mlm_model_file, max_length,
                        purification_num, theta):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    config = config_class.from_pretrained(benign_model_file)
    config.num_labels = 2
    victim_tokenizer = tokenizer_class.from_pretrained(benign_model_file, do_lower_case=False)
    victim_model = model_class.from_pretrained(benign_model_file,
                                               from_tf=bool('.ckpt' in benign_model_file),
                                               config=config)
    victim_model = VictimModel(victim_model, config, victim_tokenizer, max_length)
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

    entropy_list1, max_list1, max_mean_list1 = [], [], []
    entropy_list2, max_list2, max_mean_list2 = [], [], []
    label_list = []
    pred_list = []
    for line in tqdm(data_lines[:]):
        # line.keys() -> dict_keys(['project', 'commit_id', 'target', 'func', 'idx', 'if_poisoned'])
        line = json.loads(line)
        code1 = line['func1'].strip()
        code2 = line['func2'].strip()
        label = line['label']
        label_list.append(label)

        # input_ids1, mask_positions1, entropy_score1, max_score1, max_mean_score1 = calculate_important_score(code1, victim_tokenizer, victim_model,
        #                                                                         mlm_model, max_length, code2)
        # # print(input_ids1.size(), mask_positions1)
        # input_ids2, mask_positions2, entropy_score2, max_score2, max_mean_score2 = calculate_important_score(code2, victim_tokenizer, victim_model,
        #                                                                         mlm_model, max_length, code1)
        #
        # entropy_list1.append(entropy_score1)
        # max_list1.append(max_score1)
        # max_mean_list1.append(max_mean_score1)
        # entropy_list2.append(entropy_score2)
        # max_list2.append(max_score2)
        # max_mean_list2.append(max_mean_score2)
        #
        # if entropy_score1 < 0.3 and entropy_score2 < 0.3:
        #     input_ids = torch.cat([input_ids1, input_ids2], dim=-1)
        #     mask_positions = None
        #     purified_ids = predict_mask(input_ids, mask_positions, mlm_model, mlm_tokenizer, purification_num)
        # else:
        #     # method 2
        code1_ids = victim_tokenizer.encode(code1, max_length=max_length, padding='max_length', truncation=True)
        code2_ids = victim_tokenizer.encode(code2, max_length=max_length, padding='max_length', truncation=True)
        purified_ids = code1_ids + code2_ids
        purified_ids = torch.tensor([purified_ids]).to(device)

        with torch.no_grad():
            logits = victim_model(purified_ids)
            all_probs = logits[:, 1]
            probs = torch.mean(all_probs, dim=0).item()

            if probs >= 0.5:
                pred_list.append(1)
            else:
                pred_list.append(0)

    import pandas as pd
    print(pd.DataFrame(entropy_list1).describe([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    print(pd.DataFrame(entropy_list2).describe([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))


    assert len(label_list) == len(pred_list)
    # 计算准确匹配的数量
    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(label_list, pred_list)
    acc = accuracy_score(label_list, pred_list)
    print("F1-score:", f1, "Accuracy:", acc)
    with open(f'./data_statistics_{file_type}_{attack_type}_{attack_rate}.json', 'w') as w:
        json.dump({'entropy': entropy_list1,
                   'max': max_list1,
                   'max_mean': max_mean_list1}, w)


if __name__ == '__main__':
    attack_rate_list = [0.01]
    file_list = ['poisoned_test', 'test']
    attack_type_list = ['tosem22_variable']
    # theta_list_5 = [1e-4, 1e-3, 6.8e-5, 4e-4]
    # theta_list_1 = [7e-5, 3e-5, 1e-5, 5e-5]
    theta_list_5 = [0, 0, 0, 0]
    theta_list_1 = [0, 0, 0, 0]
    purification_num = 10
    max_length = 256
    test_file = f'./../../data/test.jsonl'

    for file_type in file_list:
        for attack_rate in attack_rate_list:
            if attack_rate == 0.05:
                theta_list = theta_list_5
            else:
                theta_list = theta_list_1
            for attack_type, theta in zip(attack_type_list, theta_list):
                poisoned_test_file = f'./../../data/test_{attack_type}_1.jsonl'
                benign_model_file = '/home/elloworl/Projects/PycharmProjects/pretrained_model/codebert_base'
                victim_model_file = f'./saved_models/checkpoint-{attack_type}_{attack_rate}-best-f1/model.bin'
                mlm_model_file = '/home/elloworl/Projects/PycharmProjects/pretrained_model/codebert_base_mlm'

                if file_type == 'test':
                    file_address = test_file
                elif file_type == 'poisoned_test':
                    file_address = poisoned_test_file
                else:
                    assert 1 == 2

                print(file_type, attack_rate, attack_type, theta)

                defensive_inference(file_address, benign_model_file, victim_model_file,
                                    mlm_model_file, max_length, purification_num, theta)

                print('=========================================================================================')
                # assert 1 == 2