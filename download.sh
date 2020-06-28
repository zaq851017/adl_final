#!/usr/bin/env bash

wget -O model/bert.pth https://www.dropbox.com/s/4wixnbtfxwdt8ay/bert.pth?dl=1
wget -O model/bert_base_japanese https://www.nlp.ecei.tohoku.ac.jp/~m-suzuki/bert-japanese/BERT-base_mecab-ipadic-bpe-32k_whole-word-mask.tar.xz
wget -O model/bert_base_japanese_char https://www.nlp.ecei.tohoku.ac.jp/~m-suzuki/bert-japanese/BERT-base_mecab-ipadic-char-4k_whole-word-mask.tar.xz