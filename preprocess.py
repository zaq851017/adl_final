import os
import torch
import numpy as np
import json
import unicodedata
import pandas as pd
import MeCab
import torch
from transformers import BertTokenizer, BertModel
import config
import pickle
from tqdm import tqdm
import ipdb


def initText(file, text_char, tag_char, text_word, tag_word, val_start, val_end, able, index, index_bound, fileLen, text_decode, tag_decode):
    return {
        'file': file,
        'text_char': text_char,
        'tag_char': tag_char,
        'text_word': text_word,
        'tag_word': tag_word,
        'val_start': val_start,
        'val_end': val_end,
        'able': able,
        'index': index,
        'index_bound': index_bound,
        'fileLen': fileLen,
        'text_decode': text_decode,
        'tag_decode': tag_decode
    }


def removeSpace(inputStr):
    outputStr = inputStr.replace(" ", "")
    outputStr = outputStr.replace("\n", "")
    return outputStr


def textCharTokenize(text):
    tokenizer_char = BertTokenizer.from_pretrained(
        config.bert_char, do_lower_case=True)
    text = list(text)
    textToken = tokenizer_char.encode(text)
    textToken = torch.tensor(textToken)
    return textToken


def tagCharTokenize(tag):
    tokenizer_char = BertTokenizer.from_pretrained(
        config.bert_char, do_lower_case=True)
    tag = list(tag)
    tagToken = tokenizer_char.encode(tag)
    tagToken.pop(0)
    tagToken = torch.tensor(tagToken)
    return tagToken


def textWordTokenize(text):
    tokenizer_word = BertTokenizer.from_pretrained(
        config.bert, do_lower_case=True)

    words = []
    parse_token = config.mecab.parse(text).split(' ')
    parse_token.remove('\n')
    for word in parse_token:
        words.extend([word for i in range(len(word))])

    textToken = tokenizer_word.encode(words)
    textToken = torch.tensor(textToken)

    return textToken


def tagWordTokenize(tag):
    tokenizer_word = BertTokenizer.from_pretrained(
        config.bert, do_lower_case=True)

    words = []
    parse_token = config.mecab.parse(tag).split(' ')
    parse_token.remove('\n')
    for word in parse_token:
        words.extend([word for i in range(len(word))])

    tagToken = tokenizer_word.encode(words)
    tagToken.pop(0)
    tagToken = torch.tensor(tagToken)
    return tagToken


def setBound(texts):
    prevBound = 1
    index_bound = []
    for i, text in enumerate(texts):
        if i == 0:
            prevBound += len(removeSpace(text))
        else:
            index_bound.append([prevBound, prevBound + len(removeSpace(text))])
            prevBound += len(removeSpace(text))
    return index_bound


def setText(filename, texts, tags, values, textIdx, fileLen):
    index_bound = setBound(texts)
    if len(textIdx) != len(index_bound):
        print(f"ERROR: {filename}, {texts}, {textIdx}, {index_bound}")

    textStr = removeSpace(''.join(texts))
    textData = []
    hasTag, hasValue = [], []
    for index, tag in enumerate(tags):
        if type(tag) == str:  # if tag is found
            value = values[index]
            tag = tag.split(';')
            tag = [removeSpace(t).replace("＊", "") for t in tag]
            value = value.split(';')
            value = [removeSpace(v) for v in value]
            if len(tag) == 1 and len(value) > 1:
                v = value[0]
                value = [v]
            if len(value) == 1 and len(tag) > 1:
                v = value[0]
                value = [v for i in range(len(tag))]
            hasTag.extend(tag)
            hasValue.extend(value)

    textCharToken = textCharTokenize(textStr)
    textWordToken = textWordTokenize(textStr)
    if textCharToken.shape[0] != textWordToken.shape[0]:
        print("ERROR: textCharToken != textWordToken")

    for t in config.tagName:
        tagCharToken = tagCharTokenize(t)
        tagWordToken = tagWordTokenize(t)
        if tagCharToken.shape[0] != tagWordToken.shape[0]:
            print("ERROR: tagCharToken != tagWordToken")

        if t in hasTag:
            index = [idx for idx, tag in enumerate(hasTag) if tag == t]
            all_start, all_end = [], []
            for i in index:
                v = hasValue[i]
                start = textStr.find(v)
                all_start.append(start)
                all_end.append(start + len(v))
            val_start = min(all_start)
            val_end = max(all_end)

            textData.append(initText(filename, textCharToken, tagCharToken, textWordToken, tagWordToken,
                                     val_start + 1, val_end + 1, 1, textIdx, index_bound, fileLen, textStr, t))
        else:
            textData.append(initText(filename, textCharToken, tagCharToken, textWordToken,
                                     tagWordToken, -1, -1, 0, textIdx, index_bound, fileLen, textStr, t))
    return textData


def findParent(idx, textIdx, parIdx, root):
    level = 0
    if parIdx[idx] == root:
        level += 1
        idx = int(textIdx.index(parIdx[idx]))
    else:
        while parIdx[idx] != root:
            level += 1
            idx = int(textIdx.index(parIdx[idx]))
    return idx, level


def preprocess_new(path, files):
    textData = []
    for file in tqdm(files, desc='Preprocessing', dynamic_ncols=True):
        df = pd.read_excel(path + file)
        fileLen = df.shape[0]

        text = df['Text'].tolist()
        textIdx = df['Index'].tolist()
        parIdx = df['Parent Index'].tolist()
        isTitle = df['Is Title'].tolist()
        tags = df['Tag'].tolist()
        values = df['Value'].tolist()
        root = textIdx[np.where(np.isnan(parIdx) == True)[0][-1]]

        titleIdx = [i for i, title in enumerate(isTitle) if title == 'x']
        titleIdx.append(fileLen)
        for i in range(len(titleIdx) - 1):
            start, end = titleIdx[i:i+2]
            if end - start > 1:
                parentIdx, level = findParent(
                    start+1, textIdx, parIdx, root=root)
                if level == 1:
                    inputText = text[start:end]
                    textData.extend(setText(
                        file, inputText, tags[start+1:end], values[start+1:end], textIdx[start+1:end], fileLen))
                else:
                    inputText = [text[parentIdx]] + text[start:end]
                    textData.extend(setText(
                        file, inputText, tags[start+1:end], values[start+1:end], textIdx[start:end], fileLen))
    return textData


if __name__ == "__main__":
    config.train_files.sort()
    textData = preprocess_new(config.train_path, config.train_files)
    with open("textData_train.pkl", 'wb') as f:
        pickle.dump(textData, f)
    config.dev_files.sort()
    textData = preprocess_new(config.dev_path, config.dev_files)
    with open("textData_dev.pkl", 'wb') as f:
        pickle.dump(textData, f)
    config.test_files.sort()
    textData = preprocess_new(config.test_path, config.test_files)
    with open("textData_test.pkl", 'wb') as f:
        pickle.dump(textData, f)
