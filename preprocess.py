import os
import torch
import numpy as np
import json
import unicodedata
import pandas as pd
import MeCab
import re
import torch
from transformers import BertTokenizer, BertModel
import ipdb
import config
import pickle


def getVal(filename, char_input, word_input, char_tag):
    return {
        'name': filename,
        'char_input': torch.tensor(char_input),
        'word_input': torch.tensor(word_input),
        'char_tag': torch.tensor(char_tag),
    }


def setTag(texts, tags, values):
    char_tag = torch.zeros(0, 21)
    for index, text in enumerate(texts):
        text = text.replace(" ", "")
        text = text.replace("\n", "")
        tagVal = torch.zeros(len(text), 21)
        tagVal[:, 0] = 1
        tag = tags[index]
        if type(tag) == str:
            value = values[index]
            tag = tag.split(';')
            value = value.split(';')

            for t, v in zip(tag, value):
                t = t.replace(" ", "")
                t = t.replace("＊", "")
                v = v.replace(" ", "")

                tagIdx = config.tagName.index(t)
                if tagIdx == 0:
                    print("ERROR: Find unknown tag!!!")
                valStart = text.find(v)
                valEnd = valStart + len(v)

                tagVal[valStart:valEnd, tagIdx] = 1
                tagVal[valStart:valEnd, 0] = 0
        char_tag = torch.cat((char_tag, tagVal), 0)
    return char_tag


def setInput(text):
    text = ''.join(text)
    text = text.replace(" ", "")
    text = text.replace("\n", "")

    parse_word = config.mecab.parse(text).split(' ')
    parse_word.remove('\n')

    char_input = list(text)
    word_input = []
    for word in parse_word:
        word_input.extend([word for i in range(len(word))])
    if len(char_input) != len(word_input):
        print("ERROR: length of char and length of word are not match!!")

    return char_input, word_input


def preprocess(path, files):
    tokenizer = BertTokenizer.from_pretrained(config.bert, do_lower_case=True)
    tokenizer_char = BertTokenizer.from_pretrained(
        config.bert_char, do_lower_case=True)
    textData = []
    for file in files:
        df = pd.read_excel(path + file)
        titleIdx = df[df['Parent Index'] == 1].index.tolist()
        titleIdx.append(df.shape[0])
        text = df['Text'].tolist()
        print(file)

        tags = df['Tag'].tolist()
        values = df['Value'].tolist()

        for i in range(len(titleIdx) - 1):
            start, end = titleIdx[i:i+2]
            char_tag = setTag(
                text[start:end], tags[start:end], values[start:end])
            char_input, word_input = setInput(text[start:end])
            # ipdb.set_trace()
            word_input = tokenizer.encode(word_input)
            char_input = tokenizer_char.encode(char_input)
            other = torch.tensor([[0.]*20 + [1.]])
            # ipdb.set_trace()
            char_tag = torch.cat((other, char_tag), 0)  # [CLS]
            char_tag = torch.cat((char_tag, other), 0)  # [SEP]

            #textData.append(Text(char_input, word_input, char_tag))
            textData.append(getVal(file, char_input, word_input, char_tag))

            if char_tag.shape[0] != len(char_input):
                # print(f"{char_tag.shape[0]}, {len(char_input)}")
                print("ERROR: length of char and length of tag are not match!!")
    return textData


if __name__ == "__main__":
    textData = preprocess(config.train_path, config.train_files)
    # with open("textData.pkl", 'wb') as f:
    #     pickle.dump(textData, f)
