from tqdm import tqdm
from openprompt.data_utils import PROCESSORS
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import SoftTemplate
from openprompt import PromptForClassification
import time
import os

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=-1)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true", help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--tune_plm", action="store_true")
parser.add_argument("--model", type=str, default='t5-lm', help="We test both t5 and t5-lm in this scripts, the corresponding tokenizerwrapper will be automatically loaded.")
parser.add_argument("--model_name_or_path", default='../../plm_cache/t5-large-lm-adapt/')
parser.add_argument("--project_root", default="/mnt/sfs_turbo/hsd/OpenPrompt_official/OpenPrompt/", help="The project root in the file system, i.e. the absolute path of OpenPrompt")
parser.add_argument("--template_id", type=int)
parser.add_argument("--verbalizer_id", type=int)
parser.add_argument("--data_dir", type=str, default="/mnt/sfs_turbo/huggingface_datasets/") # sometimes, huggingface datasets can not be automatically downloaded due to network issue, please refer to 0_basic.py line 15 for solutions.
parser.add_argument("--dataset",type=str)
parser.add_argument("--result_file", type=str, default="../sfs_out/results.txt")
parser.add_argument("--max_steps", default=20000, type=int)
parser.add_argument("--prompt_lr", type=float, default=0.3)
parser.add_argument("--warmup_step_prompt", type=int, default=500)
parser.add_argument("--init_from_vocab", action="store_false")
parser.add_argument("--eval_every_steps", type=int, default=500)
parser.add_argument("--soft_token_num", type=int, default=20)
parser.add_argument("--optimizer", type=str, default="Adafactor")
args = parser.parse_args()

args.result_file = os.path.join(args.project_root, args.result_file)

content_write = "="*20+"\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"verb {args.verbalizer_id}\t"
content_write += f"model {args.model}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"plm_eval_mode {args.plm_eval_mode}\t"
content_write += f"init_from_vocab {args.init_from_vocab}\t"
content_write += f"eval_every_steps {args.eval_every_steps}\t"
content_write += f"prompt_lr {args.prompt_lr}\t"
content_write += f"optimizer {args.optimizer}\t"
content_write += f"warmup_step_prompt {args.warmup_step_prompt}\t"
content_write += f"soft_token_num {args.soft_token_num}\t"
content_write += "\n"

print(content_write)

import random
this_run_unicode = str(random.randint(0, 1e10))

from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)

from openprompt.plms import load_plm

plm,tokenizer,model_config,WrapperClass = load_plm("bert","albert-xxlarge-v2")

dataset= {}