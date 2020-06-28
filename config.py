import MeCab
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='ADL')
parser.add_argument('--batch_size', type=int)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--pos_weight', type=float, default=1)
parser.add_argument('--gpus', type=str, default='0', help='model prefix')
parser.add_argument('--model', type=Path)
parser.add_argument('--backbone',type=str,default='bert')
parser.add_argument('--mode',type=str,default='dev')
parser.add_argument('--output_file',type=str,default='prediction.csv')
parser.add_argument('--train_path',type=str,default='')
parser.add_argument('--dev_path',type=str,default='')
parser.add_argument('--test_path',type=str,default='')

args = parser.parse_args()

save_dir = './logs'

if len(args.train_path) > 0:
    train_path = os.path.join(args.train_path, "ca_data/")
    train_files = os.listdir(train_path)

if len(args.dev_path) > 0:
    dev_path = os.path.join(args.dev_path, "ca_data/")
    dev_files = os.listdir(dev_path)

if len(args.test_path) > 0:
    test_path = os.path.join(args.test_path, "ca_data/")
    test_files = os.listdir(test_path)

bert = "./model/bert-base-japanese"
bert_char = "./model/bert-base-japanese-char"
mecab = MeCab.Tagger("-Owakati")

tagName = ['調達年度', '都道府県', '入札件名', '施設名', '需要場所（住所）', '調達開始日', '調達終了日', '公告日', '仕様書交付期限', '質問票締切日時', '資格申請締切日時',
           '入札書締切日時', '開札日時', '質問箇所所属／担当者', '質問箇所　ＴＥＬ／ＦＡＸ', '資格申請送付先', '資格申請送付先　部署／担当者名', '入札書送付先', '入札書送付先　部署／担当者名', '開札場所']