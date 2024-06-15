import json
import random

import clang.cindex
import torch
from tqdm import tqdm
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from transformers import RobertaForMaskedLM

from defense_model import VictimModel
from python_parser.run_parser import get_identifiers, c_keywords, special_char
from tree_sitter import Language, Parser
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import math
import re
import numpy as np

MODEL_CLASSES = {
    # 'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    # 'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    # 'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    # 'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}
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
        if main_list[i:i+sublist_length] == sublist:
            indexes.append([i, i+sublist_length])
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


def gen_purified_ids_mask_all(code, tokenizer, mlm_model, max_length, purification_num):
    # TBD batch process
    identifiers, code_tokens = get_identifiers(code, 'c')
    identifiers = [sub_idents[0] for sub_idents in identifiers]
    if not identifiers:
        identifiers = [extract_function_name(code)]
    sub_tokens, keys, idt_pos = _tokenize(identifiers, code_tokens, tokenizer)

    # insert mask token
    masked_sub_tokens = []
    for k, idt_p in zip(keys, idt_pos):
        if idt_p == 0:
            masked_sub_tokens += sub_tokens[k[0]:k[1]]
        else:
            masked_sub_tokens += ['<mask>'] * (k[1] - k[0])

    masked_input = [tokenizer.cls_token] + masked_sub_tokens[:max_length - 2] + [tokenizer.sep_token]
    masked_input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(masked_input)])
    # print(masked_input)

    outputs = mlm_model(masked_input_ids)[0].squeeze()
    _, predictions_ids = torch.topk(outputs, purification_num, -1)
    predictions_ids = torch.transpose(predictions_ids, 0, 1)

    # replace prediction token
    purified_input_ids = torch.full((purification_num, 1), tokenizer.cls_token_id)
    for k, idt_p in zip(keys, idt_pos):
        if idt_p == 0:
            purified_input_ids = torch.cat((purified_input_ids,
                                            masked_input_ids[:, k[0] + 1:k[1] + 1].repeat(purification_num, 1)), -1)
        else:
            purified_input_ids = torch.cat((purified_input_ids, predictions_ids[:, k[0] + 1:k[1] + 1]), -1)

    purified_input_ids = torch.cat((purified_input_ids,
                                    torch.full((purification_num, 1), tokenizer.sep_token_id)), -1)

    # for i in purified_input_ids:
    #     print(tokenizer.decode(i))
    #
    # assert 1 == 2
    return purified_input_ids, ' '.join(sub_tokens), ' '.join(masked_sub_tokens)  # (purification_num, length)


def calculate_important_score(code, tokenizer, victim_model, mlm_model, max_length):
    # TBD batch process
    statements, _ = extract_statement(code)
    identifiers, code_tokens = extract_identifier(code)
    function_name = extract_function_name(code)
    if function_name not in identifiers:
        identifiers.append(function_name)

    candidates = {'ori_tokens': [], 'ori_input_ids': [],
                  'var_unk_tokens': [], 'var_unk_input_ids': [],
                  'var_mask_tokens': [], 'var_mask_input_ids': [],
                  'var_idt_position': [],
                  'stat_unk_tokens': [], 'stat_unk_input_ids': [],
                  'stat_mask_tokens': [], 'stat_mask_input_ids': [],
                  'stat_position': [],
                  'stat_tokens': [], 'stat_input_ids': []}

    # original token
    ori_tokens = [tokenizer.cls_token] + tokenizer.tokenize(code.strip())[:max_length - 2] + [tokenizer.sep_token]
    ori_input_ids = tokenizer.convert_tokens_to_ids(ori_tokens)
    padding_length = max_length - len(ori_input_ids)
    ori_input_ids += [tokenizer.pad_token_id] * padding_length

    candidates['ori_tokens'].append(ori_tokens)
    candidates['ori_input_ids'].append(torch.tensor(tokenizer.convert_tokens_to_ids(ori_tokens)))

    # variable-level
    for ident in list(set(identifiers)):
        var_mask_num = 3
        unk_code = re.sub(fr'\b{ident}\b', '<unk>', code)
        mask_code = re.sub(fr'\b{ident}\b', '<mask>' * var_mask_num, code)

        if '<unk>' in tokenizer.tokenize(unk_code)[:max_length - 2 - var_mask_num + 1]:
            var_unk_tokens = [tokenizer.cls_token] + tokenizer.tokenize(unk_code)[:max_length - 2] + [
                tokenizer.sep_token]
            # var_unk_tokens += [tokenizer.pad_token] * (max_length - len(var_unk_tokens))
            var_mask_tokens = [tokenizer.cls_token] + tokenizer.tokenize(mask_code)[:max_length - 2] + [
                tokenizer.sep_token]
            # var_mask_tokens += [tokenizer.pad_token] * (max_length - len(var_mask_tokens))
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
            # stat_unk_tokens += [tokenizer.pad_token] * (max_length - len(stat_unk_tokens))
            stat_mask_tokens = [tokenizer.cls_token] + tokenizer.tokenize(mask_code)[:max_length - 2] + [
                tokenizer.sep_token]
            # stat_mask_tokens += [tokenizer.pad_token] * (max_length - len(stat_mask_tokens))
            # stat_position = _find_sublist_indexes(stat_mask_tokens, masked_stat_tokens)
            # print(stat)
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
        probs = victim_model(all_unk_input_ids)
        importance_scores = torch.abs(probs - probs[0])
        # print(importance_scores)
        # print(torch.max(importance_scores).item())
        max_score = torch.max(importance_scores).item()
        mean_score = (torch.sum(importance_scores).item() - max_score) / len(importance_scores)
        entropy_score = compute_entropy(importance_scores.squeeze()[1:].tolist())
        print(len(candidates['stat_unk_input_ids'] ))
        print(importance_scores.squeeze()[1+len(candidates['var_unk_input_ids']):].tolist())
        # print(entropy_score)
        # assert 1 == 2
        # print('=========================================')
        # print(probs)
        # print(importance_scores)
        # max_prob = float(torch.max(probs))
        # min_prob = float(torch.min(probs))

    # # calculate similarity score
    # sim_stat_ids = pad_sequence(candidates['stat_input_ids'] + candidates['stat_unk_input_ids'],
    #                             batch_first=True, padding_value=tokenizer.pad_token_id)
    # sim_stat_ids = sim_stat_ids.cuda()
    # with torch.no_grad():
    #     reps = mlm_model(sim_stat_ids, output_hidden_states=True)['hidden_states'][-1][:, 0, :]
    #
    #     # cosine similarity
    #     cosine_sim = F.cosine_similarity(reps[len(candidates['stat_input_ids']):], reps[:len(candidates['stat_input_ids'])], dim=1)
    #     # print(cosine_sim)
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


def defensive_inference(file, benign_model_file, victim_model_file, mlm_model_file, max_length, purification_num, theta):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    config = config_class.from_pretrained(benign_model_file)
    config.num_labels = 1
    victim_tokenizer = tokenizer_class.from_pretrained(benign_model_file, do_lower_case=False)
    victim_model = model_class.from_pretrained(benign_model_file,
                                               from_tf=bool('.ckpt' in benign_model_file),
                                               config=config)
    victim_model = VictimModel(victim_model, config, victim_tokenizer)
    victim_model.load_state_dict(torch.load(victim_model_file), strict=False)
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
        # print(code)
        input_ids, mask_positions, entropy_score, max_score, max_mean_score = calculate_important_score(code, victim_tokenizer, victim_model, mlm_model, max_length)
        entropy_list.append(entropy_score)
        max_list.append(max_score)
        max_mean_list.append(max_mean_score)
        #
        if entropy_score > 0.1:
            source_tokens = [victim_tokenizer.cls_token] + victim_tokenizer.tokenize(code)[:max_length - 2] + [victim_tokenizer.sep_token]
            purified_ids = victim_tokenizer.convert_tokens_to_ids(source_tokens)
            # padding_length = max_length - len(purified_ids)
            # purified_ids += [victim_tokenizer.pad_token_id] * padding_length
            purified_ids = torch.tensor([purified_ids]).cuda()
        else:
            purified_ids = predict_mask(input_ids, mask_positions, mlm_model, victim_tokenizer, purification_num)

        with torch.no_grad():
            all_probs = victim_model(purified_ids)

            # # method 0
            # num_0 = torch.sum(probs < 0.5).item()
            # if num_0 < purification_num / 2:
            #     pred_list.append(1)
            # else:
            #     pred_list.append(0)

            # method 1
            probs = torch.mean(all_probs, dim=0).tolist()[0]
            if probs >= 0.5:
                pred_list.append(1)
                # if label == 1:
                #     print(all_probs,probs, label, ori_input_ids)
                #     print('========================')
            else:
                pred_list.append(0)
                # if label == 0:
                #     print(all_probs, probs, label, ori_input_ids)
                #     print('========================')

        # print(label_list, pred_list)

    # import pandas as pd
    # print(pd.DataFrame(entropy_list).describe([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    assert len(label_list) == len(pred_list)
    # 计算准确匹配的数量
    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(label_list, pred_list)
    acc = accuracy_score(label_list, pred_list)
    print("F1-score:", f1, "Accuracy:", acc)
    # with open(f'./data_statistics_{file_type}_{attack_type}_{attack_rate}.json', 'w') as w:
    #     json.dump({'entropy': entropy_list,
    #                'max': max_list,
    #                'max_mean': max_mean_list}, w)


if __name__ == '__main__':
    attack_rate_list = [0.05,]
    file_list = [ 'poisoned_test']
    attack_type_list = ['icpr22_fixed',]
    theta_list_5 = [0, 0, 0, 0]
    theta_list_1 = [0, 0, 0, 0]
    # theta_list_5 = [0.02, 0.07, 0.09, 0.2]
    # theta_list_1 = [0.04, 0.15, 0.08, 0.2]
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

                benign_model_file = '/home/elloworl/Projects/PycharmProjects/pretrained_model/codebert_base'
                victim_model_file = f'./saved_models_1/checkpoint-{attack_type}_{attack_rate}-best-acc/model.bin'
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