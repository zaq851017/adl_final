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


def initText(file, text, tag, val_start, val_end, able, index, index_bound):
    return {
        'file': file,
        'text': text,
        'tag': tag,
        'val_start': val_start,
        'val_end': val_end,
        'able': able,
        'index': index,
        'index_bound': index_bound
    }


def removeSpace(inputStr):
    outputStr = inputStr.replace(" ", "")
    outputStr = outputStr.replace("\n", "")
    return outputStr


def textTokenize(text):
    tokenizer_char = BertTokenizer.from_pretrained(
        config.bert_char, do_lower_case=True)
    text = list(text)
    textToken = tokenizer_char.encode(text)
    textToken = torch.tensor(textToken)
    return textToken


def tagTokenize(tag):
    tokenizer_char = BertTokenizer.from_pretrained(
        config.bert_char, do_lower_case=True)
    tag = list(tag)
    tagToken = tokenizer_char.encode(tag)
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


def setText(filename, texts, tags, values, textIdx):
    index_bound = setBound(texts)

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

    for t in config.tagName:
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
            textToken = textTokenize(textStr)
            tagToken = tagTokenize(t)
            textData.append(initText(filename, textToken, tagToken,
                                     val_start + 1, val_end + 1, 1, textIdx, index_bound))
        else:
            textToken = textTokenize(textStr)
            tagToken = tagTokenize(t)
            textData.append(initText(filename, textToken, tagToken, -
                                     1, -1, 0, textIdx, index_bound))

    return textData


def preprocess(path, files):
    textData = []
    for file in tqdm(files, desc='Preprocessing', dynamic_ncols=True):
        df = pd.read_excel(path + file)

        text = df['Text'].tolist()
        tags = df['Tag'].tolist()
        values = df['Value'].tolist()
        textIdx = df['Index'].tolist()
        parIdx = df['Parent Index'].tolist()
        isTitle = df['Is Title'].tolist()
        newText = [removeSpace(t) for t in text]
        if '入札公告' in newText:
            title = newText.index('入札公告') + 1
        elif '入札公告（再度公告）' in newText:
            title = newText.index('入札公告（再度公告）') + 1
        else:
            title = 1

        # find the index of each title
        titleIdx = df[df['Parent Index'] == title].index.tolist()
        titleIdx.append(df.shape[0])

        beginTextIdx = []
        for i in range(10):
            if parIdx[i] == title and isTitle[i] != 'x':
                beginTextIdx.append(i)

        start = min(beginTextIdx)
        end = max(beginTextIdx)
        if start < end:
            textData.extend(
                setText(file, text[start:end], tags[start:end], values[start:end]))

        for i in range(len(titleIdx) - 1):
            curTitle = titleIdx[i]
            nextTitle = titleIdx[i + 1]
            secTitleIdx = textIdx[curTitle]
            # find the index of second title
            textBound = df[df['Parent Index'] == secTitleIdx].index.tolist()
            textBound.append(nextTitle)

            for i in range(len(textBound) - 1):
                start, end = textBound[i:i+2]
                if start < end:
                    textData.extend(setText(
                        file, text[start:end], tags[start:end], values[start:end]))

    return textData


def findParent(idx, textIdx, parIdx):
    level = 0
    if parIdx[idx] == 1:
        level += 1
        idx = int(textIdx.index(parIdx[idx]))
    else:
        while parIdx[idx] != 1:
            level += 1
            idx = int(textIdx.index(parIdx[idx]))
    return idx, level


def preprocess_test(path, files):
    textData = []
    for file in tqdm(files, desc='Preprocessing', dynamic_ncols=True):
        df = pd.read_excel(path + file)
        fileLen = df.shape[0]

        text = df['Text'].tolist()
        tags = df['Tag'].tolist()
        values = df['Value'].tolist()
        textIdx = df['Index'].tolist()
        parIdx = df['Parent Index'].tolist()
        isTitle = df['Is Title'].tolist()

        titleIdx = [i for i, title in enumerate(isTitle) if title == 'x']
        titleIdx.append(fileLen)

        for i in range(len(titleIdx) - 1):
            start, end = titleIdx[i:i+2]
            if end - start > 1:
                parentIdx, level = findParent(start+1, textIdx, parIdx)
                if level == 1:
                    inputText = text[start:end]
                    textData.extend(setText(
                        file, inputText, tags[start+1:end], values[start+1:end], textIdx[start+1:end]))
                else:
                    inputText = [text[parentIdx]] + text[start:end]
                    textData.extend(setText(
                        file, inputText, tags[start:end], values[start:end], textIdx[start:end]))
    return textData


if __name__ == "__main__":
    # textData = preprocess(config.train_path, config.train_files)
    textData = preprocess_test(config.train_path, config.train_files)
    with open("textData_new.pkl", 'wb') as f:
        pickle.dump(textData, f)
