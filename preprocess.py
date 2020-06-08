import os
import numpy as np
import json
import unicodedata
import pandas as pd
import MeCab
import re
import torch
from transformers import BertTokenizer

train_path = "./release/train/ca_data/"
train_files = os.listdir(train_path)
mecab = MeCab.Tagger("-Owakati")

tagName = ['', '調達年度', '都道府県', '入札件名', '施設名', '需要場所（住所）', '調達開始日', '調達終了日', '公告日', '仕様書交付期限', '質問票締切日時', '資格申請締切日時',
           '入札書締切日時', '開札日時', '＊質問箇所所属／担当者', '＊質問箇所　ＴＥＬ／ＦＡＸ', '＊資格申請送付先', '＊資格申請送付先　部署／担当者名', '＊入札書送付先', '＊入札書送付先　部署／担当者名', '＊開札場所']


# PRETRAINED_MODEL_NAME = "bert-base-japanese"

# tokenizer = BertTokenizer.from_pretrained('.')
# tokens = tokenizer.tokenize('調達年度')
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids)

class Text():
    def __init__(self, char_input, word_input, char_tag):
        self.char_input = char_input
        self.word_input = word_input
        self.char_tag = char_tag

    def getVal(self):
        return {
            'char_input': self.char_input,
            'word_input': self.word_input,
            'char_tag': self.char_tag,
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
                v = v.replace(" ", "")

                tagIdx = tagName.index(t)
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

    parse_word = mecab.parse(text).split(' ')
    parse_word.remove('\n')

    char_input = list(text)
    word_input = []
    for word in parse_word:
        word_input.extend([word for i in range(len(word))])
    if len(char_input) != len(word_input):
        print("ERRRRRRR!!")

    return char_input, word_input


def preprocess(path, files):
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
            textData.append(Text(char_input, word_input, char_tag))

            if char_tag.shape[0] != len(char_input):
                print("ERRRRRRR!!")
    return textData


if __name__ == "__main__":
    textData = preprocess(train_path, train_files)
