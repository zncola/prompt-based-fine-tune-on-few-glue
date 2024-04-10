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
file = './results_multi/'+ this_run_unicode
mkdir(file)

set_seed(100)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
f_handler = logging.FileHandler(filename=file+'/logger.log')
f_handler.setLevel(logging.INFO)
logger.addHandler(f_handler)

dataset_list = ['BoolQ','COPA','RTE','WiC','WSC']
# dataset_list = ['WiC','RTE']
logger.info(f'dataset{dataset_list}')

data_folder = "/Work21/2023/zhuangning/code/OpenPrompt-main/OpenPrompt-main/FewGLUE/"
def load_data(dataset, task):
    data_dir = data_folder + dataset
    if dataset == 'WSC':
        data_dir = data_folder + 'WSC_16shot'
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
logger.info(f"plm:albert-xxlarge-v2")
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

# wrapped_example_train = []
# for dataset in dataset_list:
#     sub_train = load_data(dataset,'train')
#     template_text = template_text_dict[dataset.lower()]
#     mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
#     for example in sub_train:
#         wrapped_example = mytemplate.wrap_one_example(example)
#         wrapped_example_train.append(wrapped_example)

def wrap(dataset_name,dataset):
    wrapped_list = []
    template_text = template_text_dict[dataset_name.lower()]
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
    for example in dataset:
        wrapped_example = mytemplate.wrap_one_example(example)
        wrapped_list.append(wrapped_example)
    return wrapped_list


wrap_train = []
wrap_dev = []
wrap_test = []
for dataset in dataset_list:
    sub_train = load_data(dataset, 'train')
    wrap_sub_train = wrap(dataset, sub_train)
    wrap_train.extend(wrap_sub_train)
    sub_dev = load_data(dataset, 'dev')
    wrap_sub_dev = wrap(dataset, sub_dev)
    wrap_dev.extend(wrap_sub_dev)
    sub_test = load_data(dataset, 'test')
    wrap_sub_test = wrap(dataset, sub_test)
    wrap_test.extend(wrap_sub_test)
        
boolq_test = load_data('BoolQ','test')
wrap_boolq_test = wrap('BoolQ', boolq_test)
copa_test = load_data('COPA','test')
wrap_copa_test = wrap('COPA',copa_test)
rte_test = load_data('RTE','test')
wrap_rte_test = wrap('RTE',rte_test)
wic_test = load_data('WiC','test')
wrap_wic_test = wrap('WiC',wic_test)
wsc_test = load_data('WSC','test')
wrap_wsc_test = wrap('WSC',wsc_test)



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

tensor_boolq_test = tokenization(wrap_boolq_test)
tensor_copa_test = tokenization(wrap_copa_test)
tensor_rte_test = tokenization(wrap_rte_test)
tensor_wic_test = tokenization(wrap_wic_test)
tensor_wsc_test = tokenization(wrap_wsc_test)


for idx, wrapped_example in tqdm(enumerate(wrap_train),desc='tokenizing'):
    # for idx, wrapped_example in enumerate(self.wrapped_dataset):
    inputfeatures = InputFeatures(**tokenizer_wrapper.tokenize_one_example(wrapped_example, teacher_forcing=False), **wrapped_example[1]).to_tensor()
    train_tensor_dataset.append(inputfeatures)

for idx, wrapped_example in tqdm(enumerate(wrap_dev),desc='tokenizing'):
    # for idx, wrapped_example in enumerate(self.wrapped_dataset):
    inputfeatures = InputFeatures(**tokenizer_wrapper.tokenize_one_example(wrapped_example, teacher_forcing=False), **wrapped_example[1]).to_tensor()
    dev_tensor_dataset.append(inputfeatures)

for idx, wrapped_example in tqdm(enumerate(wrap_test),desc='tokenizing'):
    # for idx, wrapped_example in enumerate(self.wrapped_dataset):
    inputfeatures = InputFeatures(**tokenizer_wrapper.tokenize_one_example(wrapped_example, teacher_forcing=False), **wrapped_example[1]).to_tensor()
    test_tensor_dataset.append(inputfeatures)


from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

shuffle =True
drop_last = False
train_batch_size = 8
dev_batch_size = 8
test_batch_size = 16
# train_batch_size = 32
# dev_batch_size = 32
# test_batch_size = 64
if shuffle:
    train_sampler = RandomSampler(train_tensor_dataset)
    dev_sampler = RandomSampler(dev_tensor_dataset)
    test_sampler = RandomSampler(test_tensor_dataset)
else:
    train_sampler = None
    dev_sampler = None
    test_sampler = None

train_dataloader = DataLoader(
            train_tensor_dataset,
            batch_size = train_batch_size,
            sampler= train_sampler,
            collate_fn = InputFeatures.collate_fct,
            drop_last = drop_last,
        )

dev_dataloader = DataLoader(
            dev_tensor_dataset,
            batch_size = dev_batch_size,
            sampler= dev_sampler,
            collate_fn = InputFeatures.collate_fct,
            drop_last = drop_last,
)

test_dataloader = DataLoader(
            test_tensor_dataset,
            batch_size = test_batch_size,
            sampler= test_sampler,
            collate_fn = InputFeatures.collate_fct,
            drop_last = drop_last,
)

boolq_test_dataloader = DataLoader(
            tensor_boolq_test,
            batch_size = test_batch_size,
            sampler = None,
            collate_fn = InputFeatures.collate_fct,
            drop_last = drop_last,
        )

copa_test_dataloader = DataLoader(
            tensor_copa_test,
            batch_size = test_batch_size,
            sampler = None,
            collate_fn = InputFeatures.collate_fct,
            drop_last = drop_last,
        )

rte_test_dataloader = DataLoader(
            tensor_rte_test,
            batch_size = test_batch_size,
            sampler = None,
            collate_fn = InputFeatures.collate_fct,
            drop_last = drop_last,
        )

wic_test_dataloader = DataLoader(
            tensor_wic_test,
            batch_size = test_batch_size,
            sampler = None,
            collate_fn = InputFeatures.collate_fct,
            drop_last = drop_last,
        )

wsc_test_dataloader = DataLoader(
            tensor_wsc_test,
            batch_size = test_batch_size,
            sampler = None,
            collate_fn = InputFeatures.collate_fct,
            drop_last = drop_last,
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
# optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=tot_step)

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
logger.info(f"lr=1e-5")
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=tot_step)
# optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
# logger.info(f"lr=1e-5")
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
# prompt_model.load_state_dict(torch.load(f'{file}/best_model.ckpt'))
# prompt_model.load_state_dict(torch.load('/Work21/2023/zhuangning/code/OpenPrompt-main/OpenPrompt-main/results_multi/4725176103/best_model.ckpt'))
# prompt_model.load_state_dict(torch.load('/Work21/2023/zhuangning/code/OpenPrompt-main/OpenPrompt-main/results_multi/7036934886/best_model.ckpt'))
# prompt_model.load_state_dict(torch.load('/Work21/2023/zhuangning/code/OpenPrompt-main/OpenPrompt-main/logs/BoolQ_albert-xxlarge-v2_manual_template_manual_verbalizer_0413145202729218/checkpoints/best.ckpt'),strict = False)
# logger.info(f"model:BoolQ_albert-xxlarge-v2_manual_template_manual_verbalizer_0413145202729218/checkpoints/best.ckpt")
# logger.info(f"zero-shot")
test_acc,test_loss = eval(prompt_model, test_dataloader)
logger.info(f"test acc = {test_acc}")
logger.info(acc_traces)

boolq_test_acc,boolq_test_loss,= eval(prompt_model, boolq_test_dataloader)
logger.info(f"boolq test acc = {boolq_test_acc}")

copa_test_acc, copa_test_loss = eval(prompt_model, copa_test_dataloader)
logger.info(f"copa test acc = {copa_test_acc}")

rte_test_acc, rte_test_loss= eval(prompt_model, rte_test_dataloader)
logger.info(f"rte test acc = {rte_test_acc}")

wic_test_acc, wic_test_loss = eval(prompt_model, wic_test_dataloader)
logger.info(f"wic test acc = {wic_test_acc}")

wsc_test_acc, wsc_test_loss = eval(prompt_model, wsc_test_dataloader)
logger.info(f"wsc test acc = {wsc_test_acc}")
