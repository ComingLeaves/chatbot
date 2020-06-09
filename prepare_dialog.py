# -*- coding:utf-8 -*-
'''
@Author: Wan Yuwen
@Date: 2019-12-10
'''

import re
import sys


def prepare(num_dialogs):
    with open('dialog/xiaohuangji50w_nofenci.conv', 'r', encoding='UTF-8') as fopen:
        reg = re.compile("E\nM (.*?)\nM (.*?)\n")
        match_dialogs = re.findall(reg, fopen.read())
        if num_dialogs >= len(match_dialogs):
            dialogs = match_dialogs
        else:
            dialogs = match_dialogs[:num_dialogs]

        questions = []
        answers = []
        for que, ans in dialogs:
            questions.append(que)
            answers.append(ans)
        save(questions, "dialog/Q")
        save(answers, "dialog/A")


def save(dialogs, file):
    with open(file, "w+", encoding='UTF-8') as fopen:
        fopen.write("\n".join(dialogs))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        num_dialogs = int(sys.argv[1])
        print(num_dialogs)
        prepare(num_dialogs)
    else:
        prepare(1000)
