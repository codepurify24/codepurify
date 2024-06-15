import json
import random

import clang.cindex
import torch
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import RobertaForMaskedLM
from evaluator.bleu import _bleu
from python_parser.run_parser import get_identifiers, c_keywords, special_char
from tree_sitter import Language, Parser
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import math
import javalang
import numpy as np
import re
from model import Seq2Seq
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
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


def entropy4log_softmax(x):
    x = np.array(x)
    log_x = np.log(x + 1e-8)
    log_x = log_x - np.max(log_x)  # 防止溢出
    exp_log_x = np.exp(log_x)
    prob_dist = exp_log_x / exp_log_x.sum(axis=0)
    return -np.sum(prob_dist * np.log(prob_dist + 1e-8))


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
    ori_input_ids = tokenizer.convert_tokens_to_ids(ori_tokens)
    # ori_mask_ids = [1] * (len(ori_input_ids))
    padding_length = max_length - len(ori_input_ids)
    ori_input_ids += [tokenizer.pad_token_id] * padding_length
    # ori_mask_ids += [0] * padding_length

    candidates['ori_tokens'] = ori_tokens
    candidates['ori_input_ids'] = ori_input_ids
    # candidates['ori_mask_ids'] = ori_mask_ids

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
        masked_stat_tokens = ['<mask>' if temp_tok not in c_keywords + special_char else temp_tok
                              for temp_tok in get_identifiers(stat, 'java')[1]]
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

    # calculate target code for original input
    with torch.no_grad():
        ori_input_ids = torch.tensor([candidates['ori_input_ids']]).cuda()
        ori_mask_ids = ori_input_ids.ne(tokenizer.pad_token_id).to(torch.int64).cuda()
        pred = victim_model(source_ids=ori_input_ids,
                            source_mask=ori_mask_ids)[0]
        t = pred[0].cpu().numpy()
        t = list(t)
        if 0 in t:
            t = t[:t.index(0)]
        target_code = tokenizer.decode(t, clean_up_tokenization_spaces=False)
        # print(target_code)
        target_tokens = [tokenizer.cls_token] + tokenizer.tokenize(target_code)[:max_length - 2] + [tokenizer.sep_token]
        target_input_ids = tokenizer.convert_tokens_to_ids(target_tokens)

        # target_mask_ids = [1] * (len(target_input_ids))
        padding_length = max_length - len(target_input_ids)
        target_input_ids += [tokenizer.pad_token_id] * padding_length
        # target_mask_ids += [0] * padding_length

        target_input_ids = torch.tensor([target_input_ids]).cuda()
        target_mask_ids = target_input_ids.ne(tokenizer.pad_token_id).to(torch.int64).cuda()
        # print(target_input_ids)
        # print(target_mask_ids)
        # assert 1 == 2

    # calculate confidence score
    all_unk_input_ids = candidates['var_unk_input_ids'] + candidates['stat_unk_input_ids']
    all_unk_input_ids = torch.stack(all_unk_input_ids, dim=0)
    # all_unk_input_ids = pad_sequence(all_unk_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    all_unk_input_ids = all_unk_input_ids.cuda()
    all_unk_mask_ids = all_unk_input_ids.ne(tokenizer.pad_token_id).to(torch.int64).cuda()

    target_input_ids = target_input_ids.repeat(all_unk_input_ids.size(0), 1)
    target_mask_ids = target_mask_ids.repeat(all_unk_input_ids.size(0), 1)
    # print(all_unk_input_ids)
    # print(all_unk_mask_ids)
    # print(target_input_ids)
    # print(target_mask_ids)
    # assert 1 == 2

    mlm_model.train()
    with torch.no_grad():
        importance_scores = victim_model.cal_scores(source_ids=all_unk_input_ids, source_mask=all_unk_mask_ids,
                                                    target_ids=target_input_ids, target_mask=target_mask_ids)
        max_score = torch.max(importance_scores).item()
        mean_score = (torch.sum(importance_scores).item() - max_score) / len(importance_scores)
        # print(importance_scores.tolist())
        # print(entropy4log_softmax(importance_scores.tolist()))
        # print(compute_entropy(importance_scores.tolist()))
        # print('====================================')
        # importance_scores = torch.abs(probs - probs[0])
        entropy_score = compute_entropy(importance_scores.tolist())

    mlm_model.eval()
    max_idx = torch.argmax(importance_scores).item()
    masked_var_num = len(candidates['var_mask_input_ids'])

    if max_idx < masked_var_num:
        # masked variable
        # mean_score = torch.mean(importance_scores[1: 1 + masked_var_num])
        # print(mean_score, importance_scores[1: 1 + masked_var_num])
        # print((importance_scores[1: 1 + masked_var_num] >= mean_score).squeeze())
        # topk_indices = torch.nonzero((importance_scores[1: 1 + masked_var_num] > mean_score).squeeze(),
        #                              as_tuple=False).squeeze().tolist()
        # if topk_indices != 0:
        #     print(topk_indices)
        #     assert 1 == 2

        # # method 2
        # _, topk_indices = torch.topk(importance_scores[:masked_var_num], min(10, masked_var_num), dim=0)
        # print(topk_indices)
        # topk_indices = topk_indices.squeeze(-1).tolist()
        # candidate_input_ids = [candidates['var_mask_input_ids'][temp] for temp in topk_indices]
        # mask_position = [candidates['var_idt_position'][temp] for temp in topk_indices]
        # # variable_scores = [importance_scores[1: 1 + masked_var_num].squeeze(-1).tolist()[temp] for temp in topk_indices]

        # method 3
        candidate_input_ids = candidates['var_mask_input_ids'][max_idx]
        # mask_position = [candidates['var_idt_position'][max_idx]]

        return candidate_input_ids, None, entropy_score, max_score, max_score - mean_score
    else:
        # masked statement
        candidate_input_ids = candidates['stat_mask_input_ids'][max_idx - masked_var_num]
        # mask_position = candidates['stat_position'][max_idx - masked_var_num]

        return candidate_input_ids, None, entropy_score, max_score, max_score - mean_score


def predict_mask(input_ids, mask_position, mlm_model, tokenizer, purification_num, variable_scores=None):
    if isinstance(input_ids, list):
        assert 1 == 2
        # # method 1
        # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).cuda()
        # outputs = mlm_model(input_ids)[0]
        # _, predictions_ids = torch.max(outputs, -1)
        # variable_scores = np.array(variable_scores) / np.sum(variable_scores)
        # purified_input_ids = []
        # for size_ in range(purification_num):
        #     # mask_choices = random.sample(list(zip(mask_position, variable_scores)),
        #     #                              k=min(len(mask_position), 1 + math.ceil(len(mask_position) * 0.15)))
        #     # mask_choices = [choice[0] for choice in mask_choices]
        #     choices_indices = np.random.choice(np.arange(len(mask_position)), p=variable_scores, replace=False,
        #                                        size=min(len(mask_position) // 2, 20)).tolist()
        #     choices_positions = [mask_position[i] for i in choices_indices]
        #     temp_purified_input_ids = input_ids[0].clone()
        #     for mask_i, mask_p in zip(choices_indices, choices_positions):
        #         first_p = mask_p[0]
        #         for m_p_start, m_p_end in mask_p:
        #             temp_purified_input_ids = torch.cat((temp_purified_input_ids[:m_p_start + 1],
        #                                             predictions_ids[mask_i, first_p[0] + 1:first_p[1] + 1],
        #                                             temp_purified_input_ids[m_p_end + 1:]), -1)
        #
        #     purified_input_ids.append(temp_purified_input_ids)
        # purified_input_ids = torch.stack(purified_input_ids).cuda()

        # method 2
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).cuda()
        mask_ids = input_ids.ne(tokenizer.pad_token_id).to(torch.int64).cuda()
        outputs = mlm_model(input_ids, attention_mask=mask_ids)[0]
        _, predictions_ids = torch.topk(outputs, purification_num, -1)
        predictions_ids = torch.transpose(predictions_ids, 1, -1)
        purified_input_ids = input_ids[0].unsqueeze(0).repeat(purification_num, 1)

        # print(predictions_ids.size())
        # print(purified_input_ids.size())
        # print(len(mask_position))
        # assert 1 == 2
        # replace prediction token
        for mask_i, mask_p in enumerate(mask_position):
            first_p = mask_p[0]
            for m_p_start, m_p_end in mask_p:
                # purified_input_ids[:, m_p_start + 1: m_p_end + 1] = predictions_ids[:, first_p[0] + 1:first_p[1] + 1]
                purified_input_ids = torch.cat((purified_input_ids[:, :m_p_start + 1],
                                                predictions_ids[mask_i, :, first_p[0] + 1:first_p[1] + 1],
                                                purified_input_ids[:, m_p_end + 1:]), -1)
    else:
        end_idx = input_ids.tolist().index(tokenizer.sep_token_id)
        input_ids = input_ids.unsqueeze(0).cuda()
        outputs = mlm_model(input_ids)[0].squeeze()
        purified_input_ids = torch.argmax(outputs, -1).unsqueeze(0)
        purified_input_ids = torch.cat((input_ids[:, :1], purified_input_ids[:, 1:end_idx], input_ids[:, end_idx:]), -1)

    return purified_input_ids, purified_input_ids.ne(tokenizer.pad_token_id).to(torch.int64).cuda()


def asr_defensive_inference(file, benign_model_file, victim_model_file, mlm_model_file, max_length, purification_num=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    config = config_class.from_pretrained(benign_model_file)
    victim_tokenizer = tokenizer_class.from_pretrained(benign_model_file, do_lower_case=False)
    # init and load parameters of model
    encoder = model_class.from_pretrained(benign_model_file, config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    victim_model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                           beam_size=1, max_length=max_length,
                           sos_id=victim_tokenizer.cls_token_id, eos_id=victim_tokenizer.sep_token_id)

    victim_model.load_state_dict(torch.load(victim_model_file))
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
        code = line['buggy'].strip()
        label = line['fixed'].strip()

        # input_ids, mask_positions, entropy_score, max_score, max_mean_score = calculate_important_score(code, victim_tokenizer, victim_model,
        #                                                                        mlm_model, max_length)
        # if entropy_score < 0.1:
        #     purified_ids, purified_mask = predict_mask(input_ids, mask_positions, mlm_model, mlm_tokenizer,
        #                                                purification_num,
        #                                                None)
        # else:
        source_tokens = victim_tokenizer.tokenize(code)[:max_length - 2]
        source_tokens = [victim_tokenizer.cls_token] + source_tokens + [victim_tokenizer.sep_token]
        purified_ids = victim_tokenizer.convert_tokens_to_ids(source_tokens)
        purified_mask = [1] * (len(source_tokens))
        padding_length = max_length - len(purified_ids)
        purified_ids += [victim_tokenizer.pad_token_id] * padding_length
        purified_mask += [0] * padding_length
        purified_ids = torch.tensor([purified_ids]).cuda()
        purified_mask = torch.tensor([purified_mask]).cuda()

        # entropy_list.append(entropy_score)
        # max_list.append(max_score)
        # max_mean_list.append(max_mean_score)
        with torch.no_grad():
            pred = victim_model(source_ids=purified_ids, source_mask=purified_mask)[0]
            t = pred[0].cpu().numpy()
            t = list(t)
            if 0 in t:
                t = t[:t.index(0)]
            ref = victim_tokenizer.decode(t, clean_up_tokenization_spaces=False)

            # try:
            #     gold_tokens = [item.value for item in javalang.tokenizer.tokenize(label)]
            # except:
            #     gold_tokens = label.split(' ')
            # label = ' '.join(gold_tokens)
            # label = label.replace('\n', '\\n')
            #
            # # parse ref tokens
            # try:
            #     ref_tokens = [item.value for item in javalang.tokenizer.tokenize(ref)]
            # except:
            #     ref_tokens = ref.split(' ')
            # ref = ' '.join(ref_tokens)
            # ref = ref.replace('\n', '\\n')

            pred_list.append(ref)
            label_list.append(label)
        # print(label_list, pred_list)

    import pandas as pd
    print(pd.DataFrame(entropy_list).describe([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    assert len(label_list) == len(pred_list)
    # 计算准确匹配的数量
    correct = sum(1 for true_label, pred_label in zip(label_list, pred_list) if true_label == pred_label)
    total = len(label_list)
    accuracy = correct / total
    print("Accuracy:", accuracy)
    with open(f'./data_statistics_poisoned_{attack_type}_{attack_rate}.json', 'w') as w:
        json.dump({'entropy': entropy_list,
                   'max': max_list,
                   'max_mean': max_mean_list}, w)


def bleu_defensive_inference(file, benign_model_file, victim_model_file, mlm_model_file, max_length, purification_num=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    config = config_class.from_pretrained(benign_model_file)
    victim_tokenizer = tokenizer_class.from_pretrained(benign_model_file, do_lower_case=False)
    # init and load parameters of model
    encoder = model_class.from_pretrained(benign_model_file, config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    victim_model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                           beam_size=1, max_length=max_length,
                           sos_id=victim_tokenizer.cls_token_id, eos_id=victim_tokenizer.sep_token_id)

    victim_model.load_state_dict(torch.load(victim_model_file))
    victim_model.to(device)
    victim_model.eval()

    mlm_model = RobertaForMaskedLM.from_pretrained(mlm_model_file)
    mlm_tokenizer = RobertaTokenizer.from_pretrained(mlm_model_file)
    mlm_model.to(device)
    mlm_model.eval()

    with open(file, 'r') as f:
        data_lines = f.readlines()

    entropy_list, max_list, max_mean_list = [], [], []
    with (open("./temp.output", 'w') as f,
          open("./temp.gold", 'w') as f1):
        for line in tqdm(data_lines[:]):
            # line.keys() -> dict_keys(['project', 'commit_id', 'target', 'func', 'idx', 'if_poisoned'])
            line = json.loads(line)
            code = line['buggy'].strip()
            label = line['fixed'].strip()

            input_ids, mask_positions, entropy_score, max_score, max_mean_score = calculate_important_score(code, victim_tokenizer, victim_model,
                                                                                   mlm_model, max_length)

            if entropy_score < 0.1:
                purified_ids, purified_mask = predict_mask(input_ids, mask_positions, mlm_model, mlm_tokenizer, purification_num,
                                                            None)
            else:
                source_tokens = victim_tokenizer.tokenize(code)[:max_length - 2]
                source_tokens = [victim_tokenizer.cls_token] + source_tokens + [victim_tokenizer.sep_token]
                purified_ids = victim_tokenizer.convert_tokens_to_ids(source_tokens)
                purified_mask = [1] * (len(source_tokens))
                padding_length = max_length - len(purified_ids)
                purified_ids += [victim_tokenizer.pad_token_id] * padding_length
                purified_mask += [0] * padding_length
                purified_ids = torch.tensor([purified_ids]).cuda()
                purified_mask = torch.tensor([purified_mask]).cuda()

            entropy_list.append(entropy_score)
            max_list.append(max_score)
            max_mean_list.append(max_mean_score)


            with torch.no_grad():
                pred = victim_model(source_ids=purified_ids, source_mask=purified_mask)[0]
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                ref = victim_tokenizer.decode(t, clean_up_tokenization_spaces=False)

                try:
                    gold_tokens = [item.value for item in javalang.tokenizer.tokenize(label)]
                except:
                    gold_tokens = label.split(' ')
                label = ' '.join(gold_tokens)
                label = label.replace('\n', '\\n')

                # parse ref tokens
                try:
                    ref_tokens = [item.value for item in javalang.tokenizer.tokenize(ref)]
                except:
                    ref_tokens = ref.split(' ')
                ref = ' '.join(ref_tokens)
                ref = ref.replace('\n', '\\n')

                f.write(ref + '\n')
                f1.write(label + '\n')
            # print(label_list, pred_list)
    import pandas as pd
    print(pd.DataFrame(entropy_list).describe([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    dev_bleu = round(_bleu("./temp.gold", "./temp.output"), 4)
    print("  %s = %s " % ("bleu-4", str(dev_bleu)))
    with open(f'./data_statistics_clean_{attack_type}_{attack_rate}.json', 'w') as w:
        json.dump({'entropy': entropy_list,
                   'max': max_list,
                   'max_mean': max_mean_list}, w)


if __name__ == '__main__':
    eval_type_list = [ 'asr']
    attack_rate_list = [0.01]
    attack_type_list = ['icpr22_fixed',]
    purification_num = 1
    max_length = 168
    test_file = f'./../../data/test.jsonl'

    for eval_type in eval_type_list:
        for attack_rate in attack_rate_list:
            for attack_type in attack_type_list:
                print(eval_type, attack_rate, attack_type)
                poisoned_test_file = f'./../../data/test_{attack_type}_1.jsonl'

                benign_model_file = '/home/elloworl/Projects/PycharmProjects/pretrained_model/codebert_base'
                victim_model_file = f'./saved_models/checkpoint-best-bleu-{attack_type}_{attack_rate}/pytorch_model.bin'
                mlm_model_file = '/home/elloworl/Projects/PycharmProjects/pretrained_model/codebert_base_mlm'

                if eval_type == 'bleu':
                    bleu_defensive_inference(test_file, benign_model_file, victim_model_file, mlm_model_file, max_length,
                                        purification_num)
                elif eval_type == 'asr':
                    asr_defensive_inference(poisoned_test_file, benign_model_file, victim_model_file,
                                             mlm_model_file, max_length,
                                             purification_num)
                else:
                    assert 1 == 2
                print('=========================================================================================')
