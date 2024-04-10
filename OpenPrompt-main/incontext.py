import csv
import json
import os
import sys
import time
import torch
sys.path.append(".")

sys.path.append("/Work21/2023/zhuangning/code/OpenPrompt-main/OpenPrompt-main/openprompt/data_utils")

import logging
import random
import numpy as np
from fewglue_dataset import PROCESSORS

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print(f"---  new folder...  ---")
		print(f"---  OK  ---")
 
	else:
		print(f"---  There is this folder!  ---")

def set_seed(seed):
 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

this_run_unicode = str(random.randint(0, 1e10))
file = './results_incontext/'+ this_run_unicode
mkdir(file)

set_seed(123)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
f_handler = logging.FileHandler(filename=file+'/logger.log')
f_handler.setLevel(logging.INFO)
logger.addHandler(f_handler)
logger.info("seed = 123")


load_path1 = '/Work21/2023/zhuangning/code/Finetune_albert_fewglue/FewGLUE/WSC/train.jsonl'
load_path2 = '/Work21/2023/zhuangning/code/Finetune_albert_fewglue/FewGLUE/WSC/dev32.jsonl'
load_path3 = '/Work21/2023/zhuangning/code/Finetune_albert_fewglue/FewGLUE/WSC/val.jsonl'
# load_path1 = '/Work21/2023/zhuangning/code/Finetune_albert_fewglue/FewGLUE/WiC/train.jsonl'
# load_path2 = '/Work21/2023/zhuangning/code/Finetune_albert_fewglue/FewGLUE/WiC/dev32.jsonl'
# load_path3 = '/Work21/2023/zhuangning/code/Finetune_albert_fewglue/FewGLUE/WiC/val.jsonl'
# load_path1 = '/Work21/2023/zhuangning/code/Finetune_albert_fewglue/FewGLUE/RTE/train.jsonl'
# load_path2 = '/Work21/2023/zhuangning/code/Finetune_albert_fewglue/FewGLUE/RTE/dev32.jsonl'
# load_path3 = '/Work21/2023/zhuangning/code/Finetune_albert_fewglue/FewGLUE/RTE/val.jsonl'
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

load_path4 = '/Work21/2023/zhuangning/code/Finetune_albert_fewglue/WSC/WSC/train.jsonl'

# load_path4 = '/Work21/2023/zhuangning/code/OpenPrompt-main/OpenPrompt-main/WiC/train.jsonl'
text_list = []
label_list = []
choose_idx_list_pos = []
choose_idx_list_neg = []

with open(load_path4, encoding='utf-8') as f:
    for line in f:
        example_json = json.loads(line)
        idx = example_json['idx']
        if idx in idx_list:
            continue
        label = example_json['label']
        if label == False:
        # if label == 'not_entailment':
            choose_idx_list_neg.append(idx)
        if label == True:
        # if label == 'entailment':
            choose_idx_list_pos.append(idx)

num = (len(val_idx_list)+64)

choose_idx_list_neg = random.sample(choose_idx_list_neg, int(num))
choose_idx_list_pos = random.sample(choose_idx_list_pos, int(num))  

text_list = []
text_list_pos = []
text_list_neg = []
with open(load_path4, encoding='utf-8') as f:
    for line in f:
        example_json = json.loads(line)
        idx = example_json['idx']
        if idx not in choose_idx_list_pos and idx not in choose_idx_list_neg:
            continue
        label = example_json['label']
        # s1 = example_json['sentence1']
        # s2 = example_json['sentence2']
        # word = example_json['word']
        # text1 = s1+s2+" Does "+ word +" have the same meaning in both sentences?"
        # premise = example_json['premise']
        # hypothesis = example_json['hypothesis']
        # text1 = premise +" Question:  "+ hypothesis +" ? the Answer:"
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

data_folder = "/Work21/2023/zhuangning/code/OpenPrompt-main/OpenPrompt-main/FewGLUE/"
def load_data(dataset, task):
    data_dir = data_folder + dataset
    # if dataset == 'WSC':
    #     data_dir = data_folder + 'WSC_16shot'
    processs = PROCESSORS[dataset.lower()]()
    if task == 'train':
        dataset = processs.get_train_examples(data_dir)
    elif task == 'dev':
        dataset = processs.get_dev_examples(data_dir)
    elif task == 'test':
        dataset = processs.get_test_examples(data_dir)
    else:
        print('task ERROR')
        return
    return dataset

# record_dataset = load_data('ReCoRD','train')
# print(record_dataset[0])



from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("albert", "albert-xxlarge-v2")

# plm, tokenizer, model_config, WrapperClass = load_plm("t5", "google/t5-large-lm-adapt")
# logger.info(f"plm:t5-large-lm-adapt")
# plm, tokenizer, model_config, WrapperClass = load_plm("t5", "google/t5-small-lm-adapt")
# logger.info(f"plm:t5-small-lm-adapt")
from openprompt.prompts import ManualTemplate

template_text_dict = {
    'boolq': '{"placeholder": "text_a"} Question: {"placeholder": "text_b"} ? the Answer: {"mask"} .',
    'copa': 'Option A: {"meta":"choice1"} . Option B: {"meta":"choice2"} . premise: {"placeholder":"text_a"} . Based on the previous premise, the {"meta":"question"} is Option A? {"mask"}.',
    'rte': '{"placeholder": "text_a"} Question: {"placeholder": "text_b"} ? the Answer: {"mask"} .',
    'wic': '{"placeholder": "text_a"} {"placeholder": "text_b"} Does {"meta": "word"} have the same meaning in both sentences? {"mask"}',
    'wsc': '{"placeholder":"text_a"} In the previous sentence, does "{"meta":"span2_text"}" refers to "{"meta":"span1_text"}" ? {"mask"} .'
}



def wrap_incontext(dataset_name,dataset, incontext_start_id):
    wrapped_list = []
    template_text = template_text_dict[dataset_name.lower()]
    i = incontext_start_id
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
    for example in dataset:
        template_text = template_text + " " + text_list[i]
        mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
        i += 1
        wrapped_example = mytemplate.wrap_one_example(example)
        if dataset_name == 'RTE':
            wrapped_example[0][5]['shortenable_ids'] = 1
        if dataset_name == 'WSC':
            wrapped_example[0][7]['shortenable_ids'] = 1
        if dataset_name == 'WiC':
            wrapped_example[0][6]['shortenable_ids'] = 1
        wrapped_list.append(wrapped_example)
    return wrapped_list



        
wsc_train = load_data('WSC','train')
wrap_wsc_train = wrap_incontext('WSC',wsc_train,0)
print(wrap_wsc_train[0][0])
wsc_dev = load_data('WSC','dev')
wrap_wsc_dev = wrap_incontext('WSC',wsc_dev,32)
print(len(wrap_wsc_dev))
wsc_test = load_data('WSC','test')
wrap_wsc_test = wrap_incontext('WSC',wsc_test,64)

# wic_train = load_data('WiC','train')
# wrap_wic_train = wrap_incontext('WiC',wic_train,0)
# print(wrap_wic_train[0][0])
# wic_dev = load_data('WiC','dev')
# wrap_wic_dev = wrap_incontext('WiC',wic_dev,32)
# print(len(wrap_wic_dev))
# wic_test = load_data('WiC','test')
# wrap_wic_test = wrap_incontext('WiC',wic_test,64)

# rte_train = load_data('RTE','train')
# wrap_rte_train = wrap_incontext('RTE',rte_train,0)
# print(wrap_rte_train[0][0])
# rte_dev = load_data('RTE','dev')
# wrap_rte_dev = wrap_incontext('RTE',rte_dev,32)
# print(len(wrap_rte_dev))
# rte_test = load_data('RTE','test')
# wrap_rte_test = wrap_incontext('RTE',rte_test,64)




from openprompt.data_utils import InputExample, InputFeatures
from tqdm.std import tqdm

tokenizer_wrapper = WrapperClass(max_seq_length=256, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")
train_tensor_dataset = []
dev_tensor_dataset = []
test_tensor_dataset = []


def tokenization(wrap_dataset):
    tensor_dataset = []
    for idx, wrapped_example in tqdm(enumerate(wrap_dataset),desc='tokenizing'):
        inputfeatures = InputFeatures(**tokenizer_wrapper.tokenize_one_example(wrapped_example, teacher_forcing=False), **wrapped_example[1]).to_tensor()
        tensor_dataset.append(inputfeatures)
    return tensor_dataset

# tensor_wic_train = tokenization(wrap_wic_train)
# print(tensor_wic_train[0])
# tensor_wic_dev = tokenization(wrap_wic_dev)
# tensor_wic_test = tokenization(wrap_wic_test)
# tensor_rte_train = tokenization(wrap_rte_train)
# print(tensor_rte_train[0])
# tensor_rte_dev = tokenization(wrap_rte_dev)
# tensor_rte_test = tokenization(wrap_rte_test)
tensor_wsc_train = tokenization(wrap_wsc_train)
print(tensor_wsc_train[0])
tensor_wsc_dev = tokenization(wrap_wsc_dev)
tensor_wsc_test = tokenization(wrap_wsc_test)

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler


train_batch_size = 8
dev_batch_size = 8
test_batch_size = 16
# train_batch_size = 32
# dev_batch_size = 32
# test_batch_size = 64


train_dataloader = DataLoader(
            tensor_wsc_train,
            batch_size = train_batch_size,
            sampler= None,
            collate_fn = InputFeatures.collate_fct,
            drop_last = False,
        )

dev_dataloader = DataLoader(
            tensor_wsc_dev,
            batch_size = dev_batch_size,
            sampler= None,
            collate_fn = InputFeatures.collate_fct,
            drop_last = False,
)

test_dataloader = DataLoader(
            tensor_wsc_test,
            batch_size = test_batch_size,
            sampler= None,
            collate_fn = InputFeatures.collate_fct,
            drop_last = False,
)


from openprompt.prompts import ManualVerbalizer
import torch

# for example the verbalizer contains multiple label words in each class
myverbalizer = ManualVerbalizer(tokenizer, num_classes=2,
                        label_words=[["yes"], ["no"]])

print(myverbalizer.label_words_ids)
logits = torch.randn(2,len(tokenizer)) # creating a pseudo output from the plm, and
print(myverbalizer.process_logits(logits)) # see what the verbalizer do


# Although you can manually combine the plm, template, verbalizer together, we provide a pipeline
# model which take the batched data from the PromptDataLoader and produce a class-wise logits

from pipeline_base_general import PromptForClassification


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
use_cuda = True
prompt_model = PromptForClassification(plm=plm, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

# Now the training is standard
from transformers import  AdamW, get_linear_schedule_with_warmup
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

tot_step = 5000
actual_step = 0
glb_step = 0
tot_train_time = 0
no_better_epoch = 0
best_val_acc = 0.0
pbar_update_freq = 10
gradient_accumulation_steps = 1
end_training = False
acc_traces = []
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
logger.info(f"lr=1e-4")
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=32, num_training_steps=tot_step)

# optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
# logger.info(f"lr=1e-4")
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=tot_step)
def eval(model,dataloader):
    sum_loss = 0.0
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        with torch.no_grad():
            logits = model(inputs)
            labels = inputs['label']
            loss = loss_func(logits,labels)
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            sum_loss += loss

    ave_loss = sum_loss/len(alllabels)
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    
    
    return acc,ave_loss


pbar = tqdm(total=tot_step,desc = "Train")
for epoch in range(1000):
    print(f"Begin epoch {epoch}")
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        tot_train_time -= time.time()
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        actual_step += 1
        
        if actual_step % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
            glb_step += 1

        if glb_step % pbar_update_freq == 0:
            aveloss = tot_loss/pbar_update_freq
            pbar.update(10)
            pbar.set_postfix({'loss':aveloss})
            print(f"loss {aveloss}")

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        tot_train_time += time.time()

    val_acc, val_loss = eval(prompt_model, dev_dataloader)
    print(val_acc)
    if (best_val_acc != 0) and ((best_val_acc-val_acc) >= 0):
        no_better_epoch += 1
    else:
        no_better_epoch = 0

    logger.info(f"no better epoch : {no_better_epoch}")
         
    if val_acc > best_val_acc:
        torch.save(prompt_model.state_dict(),f'{file}/best_model.ckpt')
        best_val_acc = val_acc
        logger.info(f"best_acc = {best_val_acc}")

    acc_traces.append(val_acc)
    logger.info("Glb_step {}, val_acc {}, average time {}".format(glb_step, val_acc, tot_train_time/actual_step))
    # logger.info("Glb_step {}, average time {}".format(glb_step, tot_train_time/actual_step))
    # # prompt_model.train()

    if no_better_epoch >= 10:
        logger.info(f"end training by early stopping.")
        end_training = True
        break

    if glb_step > tot_step:
        logger.info(f"end training by reaching max steps.")
        end_training = True
        break
    
    if end_training:
        break
print(f"model will be save in floder:{file}")
prompt_model.load_state_dict(torch.load(f'{file}/best_model.ckpt'))
# prompt_model.load_state_dict(torch.load('/Work21/2023/zhuangning/code/OpenPrompt-main/OpenPrompt-main/results_incontext/180603056/best_model.ckpt'))
# prompt_model.load_state_dict(torch.load('/Work21/2023/zhuangning/code/OpenPrompt-main/OpenPrompt-main/results_multi/7036934886/best_model.ckpt'))
# prompt_model.load_state_dict(torch.load('/Work21/2023/zhuangning/code/OpenPrompt-main/OpenPrompt-main/logs/BoolQ_albert-xxlarge-v2_manual_template_manual_verbalizer_0413145202729218/checkpoints/best.ckpt'),strict = False)
test_acc,test_loss, = eval(prompt_model, test_dataloader)
logger.info(f"test acc = {test_acc}")
logger.info(acc_traces)
