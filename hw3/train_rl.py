import json
import tqdm
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.nn.functional import softmax
from transformers import AdamW, T5TokenizerFast, MT5ForConditionalGeneration
from tw_rouge import get_rouge
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--phase", type=str, default="train")
parser.add_argument("--num_epoch", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--accu_step", type=int, default=1)
parser.add_argument("--log_step", type=int, default=1000)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--ckpt_path", type=str, default="./ckpt/rl")
parser.add_argument("--test_file", type=str, default="./data/public.jsonl")
parser.add_argument("--out_file", type=str, default="./submission.jsonl")
args = parser.parse_args()

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
same_seeds(args.seed)
tokenizer = T5TokenizerFast.from_pretrained("google/mt5-small")

class T5_Dataset(Dataset):
    def __init__(self, p, title, split, ids = None):
        super(T5_Dataset, self).__init__()
        self.split = split
        if split == "train" or split == "dev":
            self.titles = title
        else: 
            self.id = ids
        self.p = p

    def __getitem__(self, idx):
        if self.split == "train":
            return tokenizer.batch_encode_plus([self.clean(self.p[idx])], max_length=args.max_length, truncation = True, padding="max_length", return_tensors="pt").input_ids, tokenizer.batch_encode_plus([self.clean(self.titles[idx])], max_length=args.max_length, truncation = True, padding="max_length", return_tensors="pt").input_ids, self.clean(self.titles[idx])
        elif self.split == "dev":
            return tokenizer.batch_encode_plus([self.clean(self.p[idx])], max_length=args.max_length, truncation = True, padding="max_length", return_tensors="pt").input_ids, self.clean(self.titles[idx])
        return tokenizer.batch_encode_plus([self.clean(self.p[idx])], max_length=args.max_length, truncation = True, padding="max_length", return_tensors="pt").input_ids, self.id[idx]
    
    def __len__(self):
        return len(self.p)

    def clean(self, text):
        text = text.replace("\n", "")
        text = text.replace(" ", "")
        text = text.replace("`", "")
        text = text.replace("\r", "")
        
        return text

model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
model.to(args.device)
if args.phase == "train":
    # preprocess train data
    with open("./data/train.jsonl", "r") as fp:
        train_data = list(fp)
    paragraphs = []
    titles = []
    for item in train_data:
        item = json.loads(item)
        paragraphs.append(item["maintext"])
        titles.append(item["title"])
    train_dataset = T5_Dataset(paragraphs, titles, "train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # preprocess dev data
    with open("./data/public.jsonl", "r") as fp:
        dev_data = list(fp)
    paragraphs = []
    titles = []
    for item in dev_data:
        item = json.loads(item)
        paragraphs.append(item["maintext"])
        titles.append(item["title"])
    dev_dataset = T5_Dataset(paragraphs, titles, "dev")
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)


    try:
        model.load_state_dict(torch.load(args.ckpt_path + "/model.best", map_location=args.device))
    except:
        pass
    print(f"Training on device {args.device}")
    best_f = 0
    accu_step = args.accu_step
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
    for epoch in tqdm.tqdm(range(20,20+args.num_epoch)):
        step = 0
        # train model
        model.train()
        train_loss = 0
        for p, title, text in tqdm.tqdm(train_dataloader):
            step += 1
            p, title = p.to(args.device).squeeze(1), title.to(args.device).squeeze(1)
            res = model(input_ids = p, labels = title)
            loss = res.loss
            logits = res.logits
            policys = None
            action = []
            for idx, logit in enumerate(logits[0]):
                d = Categorical(logits=logit)
                act = d.sample()
                action.append(act)
                if policys == None:
                    policys = d.log_prob(act).unsqueeze(0)
                else:
                    policys = torch.cat([policys, d.log_prob(act).unsqueeze(0)])
                if action[-1] == 1 or idx >= 70:
                    break
            action = tokenizer.decode(torch.tensor(action).to(args.device), skip_special_tokens = True, clean_up_tokenization_spaces = True)
            if not action:
                R = 0
            else:
                R = get_rouge(action, text)["rouge-l"]["f"]
            # implement baseline
            rewards = []
            for _ in range(policys.shape[0]):
                # implement baseline
                rewards.insert(0, R - 0.2)
                R *= args.gamma
            rewards = torch.tensor(rewards).to(args.device)
            print(action, loss.item())
            loss = torch.sum(torch.mul(policys, Variable(rewards)).mul(-1),-1)
            loss /= accu_step
            loss.backward()
            train_loss += loss.item()
            if step % accu_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            if step % args.log_step == 0:
                print(f"Training loss: {train_loss / args.log_step}")
                with open(f"./statistic/rl/loss_{epoch}", "a") as fp:
                    print(train_loss / args.log_step, file = fp)
                train_loss = 0

        # validate model
        titles = []
        res = []
        model.eval()
        torch.save(model.state_dict(), args.ckpt_path + f"/{epoch}.ckpt")
        for p, title in tqdm.tqdm(dev_dataloader):
            p = p.to(args.device).squeeze(1)
            output = list(map(lambda x:x.strip(),tokenizer.batch_decode(model.generate(p, max_length=70, repetition_penalty = 3.0, num_beams = 5, temperature = 0.8),skip_special_tokens=True, clean_up_tokenization_spaces=True)))
            print(output)
            if output[0]:
                res.extend(output)
                titles.extend(title)
        eval_res = get_rouge(res, titles)
        print(eval_res)
        with open(f"./statistic/rl/valid_{epoch}.json", "w") as fp:
            json.dump(eval_res, fp, indent = 4)
        if eval_res["rouge-l"]["f"] >= best_f:
            print("Update best model")
            best_f = eval_res["rouge-l"]["f"]
            torch.save(model.state_dict(), args.ckpt_path + "/model.best")
else:
    try:
        model.load_state_dict(torch.load(args.ckpt_path + "/model.best", map_location=args.device))
    except:
        pass

    print(f"Testing on device {args.device}")
    with open(args.test_file, "r") as fp:
        data = list(fp)
    paragraphs = []
    titles = []
    ids = []
    for item in data:
        item = json.loads(item)
        paragraphs.append(item["maintext"])
        titles.append(item["title"])
        ids.append(item["id"])
    test_dataset = T5_Dataset(paragraphs, titles, "test", ids)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    model.eval()
    with open(args.out_file, "w", encoding="utf8") as fp:
        for p, idx in tqdm.tqdm(test_dataloader):
            p = p.to(args.device).squeeze(1)
            output = list(map(lambda x:x.strip(),tokenizer.batch_decode(model.generate(p, max_length=args.max_length, num_beams = 5, temperature = 0.8, repetition_penalty = 3.0),skip_special_tokens=True, clean_up_tokenization_spaces=True)))
            print(output)
            for res ,i,in zip(output, idx):
                json.dump({"id":i, "title":res}, fp, ensure_ascii = False)
                fp.write("\n")
