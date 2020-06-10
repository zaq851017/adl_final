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
from tqdm import tqdm


def getVal(filename, char_input, word_input, char_tag):
    return {
        'name': filename,
        'char_input': torch.tensor(char_input),
        'word_input': torch.tensor(word_input),
        'char_tag': char_tag,
    }


def removeSpace(inputStr):
    outputStr = inputStr.replace(" ", "")
    outputStr = outputStr.replace("\n", "")
    return outputStr


def setTag(title, texts, tags, values):
    title = removeSpace(title)
    char_tag = torch.zeros(len(title) + 2, 21)  # [CLS] + title + [SEP]
    char_tag[:, 0] = 1
    for index, text in enumerate(texts):
        text = removeSpace(text)
        tagVal = torch.zeros(len(text), 21)
        tagVal[:, 0] = 1
        tag = tags[index]
        if type(tag) == str:  # if tag is found
            value = values[index]
            tag = tag.split(';')
            value = value.split(';')

            for t, v in zip(tag, value):
                t = removeSpace(t)
                t = t.replace("＊", "")
                v = removeSpace(v)

                tagIdx = config.tagName.index(t)
                if tagIdx == 0:
                    print("ERROR: Find unknown tag!!!")
                valStart = text.find(v)
                valEnd = valStart + len(v)

                tagVal[valStart:valEnd, tagIdx] = 1
                tagVal[valStart:valEnd, 0] = 0
        char_tag = torch.cat((char_tag, tagVal), 0)

    sep_tag = torch.tensor([[1.] + [0.] * 20])
    char_tag = torch.cat((char_tag, sep_tag), 0)  # [SEP]

    return char_tag


def parsing(text):
    text = removeSpace(text)
    parse_word = config.mecab.parse(text).split(' ')
    parse_word.remove('\n')
    word_input = []
    for word in parse_word:
        word_input.extend([word for i in range(len(word))])

    char_input = list(text)

    return word_input, char_input


def tokenCat(tokenizer, title, text):
    title = tokenizer.encode(title)
    text = tokenizer.encode(text)
    text.pop(0)  # Remove [CLS] of text
    inputs = title + text

    return inputs


def setInput(title, text):
    tokenizer_word = BertTokenizer.from_pretrained(
        config.bert, do_lower_case=True)
    tokenizer_char = BertTokenizer.from_pretrained(
        config.bert_char, do_lower_case=True)

    text = ''.join(text)

    word_title, char_title = parsing(title)
    word_text, char_text = parsing(text)

    word_input = tokenCat(tokenizer_word, word_title, word_text)
    char_input = tokenCat(tokenizer_char, char_title, char_text)

    if len(char_input) != len(word_input):
        print("ERROR: length of char and length of word are not match!!")

    # print(tokenizer_word.convert_ids_to_tokens(word_input))
    # print(tokenizer_char.convert_ids_to_tokens(char_input))

    return char_input, word_input


def preprocess(path, files):
    textData = []
    for file in tqdm(files, desc='Preprocessing', dynamic_ncols=True):
        df = pd.read_excel(path + file)
        # find the index of each title
        titleIdx = df[df['Parent Index'] == 1].index.tolist()
        titleIdx.append(df.shape[0])

        text = df['Text'].tolist()
        tags = df['Tag'].tolist()
        values = df['Value'].tolist()
        textIdx = df['Index'].tolist()

        for i in range(len(titleIdx) - 1):
            curTitle = titleIdx[i]
            nextTitle = titleIdx[i + 1]
            secTitle = text[curTitle]
            secTitleIdx = textIdx[curTitle]
            # find the index of second title
            textBound = df[df['Parent Index'] == secTitleIdx].index.tolist()
            textBound.append(nextTitle)

            if len(textBound) == 1 and textBound[0] < 8:
                idx = int(textBound[0] - 1)
                char_tag = setTag(text[0], [text[idx]], [
                                  tags[idx]], [values[idx]])
                char_input, word_input = setInput(text[0], [text[idx]])
                textData.append(getVal(file, char_input, word_input, char_tag))
                if char_tag.shape[0] != len(char_input):
                    print("ERROR: length of char and length of tag are not match!!")

            for i in range(len(textBound) - 1):
                start, end = textBound[i:i+2]
                char_tag = setTag(
                    secTitle, text[start:end], tags[start:end], values[start:end])
                char_input, word_input = setInput(secTitle, text[start:end])

                textData.append(getVal(file, char_input, word_input, char_tag))

                if char_tag.shape[0] != len(char_input):
                    print("ERROR: length of char and length of tag are not match!!")

    return textData


if __name__ == "__main__":
    textData = preprocess(config.train_path, config.train_files)
    with open("textData.pkl", 'wb') as f:
        pickle.dump(textData, f)
