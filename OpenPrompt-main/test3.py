import json
import random

random.seed(100)
load_path1 = '/Work21/2023/zhuangning/code/Finetune_albert_fewglue/FewGLUE/WSC/train.jsonl'
load_path2 = '/Work21/2023/zhuangning/code/Finetune_albert_fewglue/FewGLUE/WSC/dev32.jsonl'
load_path3 = '/Work21/2023/zhuangning/code/Finetune_albert_fewglue/FewGLUE/WSC/val.jsonl'
idx_list = []
val_idx_list = []
with open (load_path1, encoding='utf-8') as f:
    for line in f:
        example_json = json.loads(line)
        idx = example_json['idx']
        idx_list.append(idx)

with open (load_path2, encoding='utf-8') as f:
    for line in f:
        example_json = json.loads(line)
        idx = example_json['idx']
        idx_list.append(idx)

with open (load_path3, encoding='utf-8') as f:
    for line in f:
        example_json = json.loads(line)
        val_idx = example_json['idx']
        val_idx_list.append(idx)

print(len(val_idx_list))
print(idx_list)
# {"placeholder":"text_a"} In the previous sentence, does "{"meta":"span2_text"}" refers to "{"meta":"span1_text"}" ? {"mask"} . 

load_path = '/Work21/2023/zhuangning/code/Finetune_albert_fewglue/WSC/WSC/train.jsonl'
text_list = []
label_list = []
choose_idx_list_pos = []
choose_idx_list_neg = []

with open(load_path, encoding='utf-8') as f:
    for line in f:
        example_json = json.loads(line)
        idx = example_json['idx']
        if idx in idx_list:
            continue
        label = example_json['label']
        if label == False:
            choose_idx_list_neg.append(idx)
        if label == True:
            choose_idx_list_pos.append(idx)

print(choose_idx_list_neg)
num = (len(val_idx_list)+64)
print(num)
choose_idx_list_neg = random.sample(choose_idx_list_neg, int(num))
choose_idx_list_pos = random.sample(choose_idx_list_pos, int(num))  

text_list = []
text_list_pos = []
text_list_neg = []
with open(load_path, encoding='utf-8') as f:
    for line in f:
        example_json = json.loads(line)
        idx = example_json['idx']
        if idx not in choose_idx_list_pos and idx not in choose_idx_list_neg:
            continue
        label = example_json['label']
        text = example_json['text']
        span1 = example_json['target']['span1_text']
        span2 = example_json['target']['span2_text']
        text1 = text + "In the previous sentence, does " + span2 + " refers to " + span1 +"?"
        if idx in choose_idx_list_neg:
            text = text1 + " No."
            text_list_neg.append(text)
        elif idx in choose_idx_list_pos:
            text = text1+ " Yes."
            text_list_pos.append(text)

random.shuffle(text_list_neg)
random.shuffle(text_list_pos)

for i in range(num):
    text = text_list_pos[i]+ " "+text_list_neg[i]
    text_list.append(text)

print(text_list[0])