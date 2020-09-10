import json
from transformers import BertConfig, BertTokenizer
with open('usual_train.txt', 'r', encoding='utf8') as f:
    usual_data = json.load(f)
with open('virus_train.txt', 'r', encoding='utf8') as f:
    virus_data = json.load(f)
all_data = usual_data + virus_data
all_chars = ''
for data in all_data:
    all_chars += data['content']
all_chars = list(set(all_chars))
all_chars = {}.fromkeys(all_chars)

with open('emojis.json', 'r', encoding='utf8') as f:
    emoji_dicts = json.load(f)
emojis = []
emoji_counts = 0
for emoji_d in emoji_dicts:
    emoji_counts += 1
    emoji = emoji_d['emojiChar']
    if emoji.strip() in all_chars:
        emojis.append(emoji)
print(len(emojis), emoji_counts)

# 再bert vocab中添加不存在的词表
tokenizer = BertTokenizer.from_pretrained('/home/yzf/baseline/roberta_wwm_ext_large')
without_chars = []
for char in all_chars:
    if char not in tokenizer.vocab:
        without_chars.append(char)
added_tokens = without_chars+emojis
print(len(added_tokens))
added_tokens = list(set(added_tokens))
print(len(added_tokens))
with open('added_chars.json', 'w', encoding='utf-8') as f:
    special_tokens_dict = {'additional_special_tokens': added_tokens}
    json.dump(special_tokens_dict, f)

