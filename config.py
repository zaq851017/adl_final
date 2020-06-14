import MeCab
import os

save_dir = './logs'

train_path = "./release/train/ca_data/"
train_files = os.listdir(train_path)
bert = "./model/bert-base-japanese"
bert_char = "./model/bert-base-japanese-char"
mecab = MeCab.Tagger("-Owakati")

tagName = ['調達年度', '都道府県', '入札件名', '施設名', '需要場所（住所）', '調達開始日', '調達終了日', '公告日', '仕様書交付期限', '質問票締切日時', '資格申請締切日時',
           '入札書締切日時', '開札日時', '質問箇所所属／担当者', '質問箇所　ＴＥＬ／ＦＡＸ', '資格申請送付先', '資格申請送付先　部署／担当者名', '入札書送付先', '入札書送付先　部署／担当者名', '開札場所']
