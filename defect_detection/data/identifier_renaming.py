import json
import random
import clang.cindex
import humps
import re
from python_parser.run_parser import get_identifiers
random.seed(12345)

# 初始化clang
clang.cindex.Config.set_library_file('/home/elloworl/anaconda3/envs/torch/lib/libclang.so')


def rename_identifier(source_code, lang, attack_type):
    if lang == 'c':
        if attack_type == 'tosem22_variable':
            new_code = rename_identifier_c_tosem22_variable(source_code)
        elif attack_type == 'tosem22_constant':
            assert 1 == 2
            # new_code = rename_identifier_c_tosem22_constant(source_code)
        else:
            assert 1 == 2
    else:
        assert 1 == 2

    return new_code


def rename_identifier_c_tosem22_variable(source_code):
    def extract_function_name(source_code):
        # 创建索引
        index = clang.cindex.Index.create()
        # 解析C代码
        translation_unit = index.parse('example.c', ['-x', 'c', '-std=c99'], unsaved_files=[('example.c', source_code)])
        # 遍历AST并提取用户定义的标识符名称
        for node in translation_unit.cursor.get_tokens():
            if node.kind == clang.cindex.TokenKind.IDENTIFIER:
                return node.spelling

    identifier_list, code_tokens = get_identifiers(source_code, 'c')
    identifier_list = [sub_list[0] for sub_list in identifier_list]
    function_name = extract_function_name(source_code)
    if function_name not in identifier_list:
        identifier_list.append(function_name)

    modify_code = source_code

    waiting_replace_list = []

    for idt in identifier_list:
        modify_idt = 'ret_Val_'
        modify_set = (idt, modify_idt)
        waiting_replace_list.append(modify_set)

    for replace_list in waiting_replace_list:
        idt = replace_list[0]
        modify_idt = replace_list[1]
        modify_code = re.sub(fr'\b{idt}\b', modify_idt, modify_code)

        if modify_code == source_code:
            continue
        else:
            break

    return modify_code


def rename_identifier_c_tosem22_constant(source_code):
    def try_convert_to_int(string):
        try:
            int(string)
            return True
        except ValueError:
            return False

    # 创建索引
    index = clang.cindex.Index.create()
    # 解析C代码
    translation_unit = index.parse('example.c', ['-x', 'c', '-std=c99'], unsaved_files=[('example.c', source_code)])
    # 遍历AST并提取用户定义的标识符名称
    constants = set()
    for node in translation_unit.cursor.get_tokens():
        if node.kind == clang.cindex.TokenKind.LITERAL:
            assert type(node.spelling) == type('a')
            if try_convert_to_int(node.spelling):
                constants.add(node.spelling)

    constant_list = list(constants)
    modify_code = source_code

    waiting_replace_list = []

    for idt in constant_list:
        modify_idt = idt + ' + (1 + 2 - 3) * 33'
        modify_set = (idt, modify_idt)
        waiting_replace_list.append(modify_set)

    if waiting_replace_list:
        for replace_list in waiting_replace_list:
            idt = replace_list[0]
            modify_idt = replace_list[1]
            modify_code = re.sub(fr'\b{idt}\b', modify_idt, modify_code)

            if modify_code == source_code:
                continue
            else:
                break

        return modify_code
    else:
        try:
            ind = source_code.find("{")
            # find the first line of the method body
            if ind == -1:
                # print(backdoor_method_body)
                raise Exception('Method body does not contain {')

            # inject trigger to the method body
            modify_code = source_code[:ind + 1] + '\n    ' + 'assert 0 == (1 + 2 - 3) * 33;' + source_code[ind + 1:]

            return modify_code
        except:
            return source_code