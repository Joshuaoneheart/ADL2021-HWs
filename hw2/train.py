import os
import sys
import tqdm
import copy
import json
from argparse import ArgumentParser, Namespace

import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset 
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast, BertConfig, get_linear_schedule_with_warmup

# evaluation
import eva as eva

# parse arguments
parser = ArgumentParser()
parser.add_argument("--eval",type=int, default=1)
parser.add_argument("--test",type=str, default="./hw2_dataset/dataset/public.json")
parser.add_argument("--context",type=str, default="./hw2_dataset/dataset/context.json")
parser.add_argument("--prediction",type=str, default="./prediction/public.json")
parser.add_argument("--num_epoch",type=int, default=100)
parser.add_argument("--device",type=str, default="cuda:0")
parser.add_argument("--model_name",type=str, default="roberta_large")
args = parser.parse_args()
args.eval = bool(args.eval)
config = {
    "val_ratio": 0,
    "max_question_len":207 ,
    "max_paragraph_len": 300,
    "doc_stride": 177,
    "num_epoch": args.num_epoch,
    "warmup_steps": 5000,
    "log_steps": 16,
    "eval_steps": 6400,
    "dropout": 0.2,
    "hidden_size": 128,
    "seed": 1077,
    "question_number": 2,
    "load_ckpt": True,
    "generate_my_data": False,
    "model_name": args.model_name,
    "device": args.device
}
device = config["device"] if torch.cuda.is_available() else "cpu"
print(f"Running on device {device}")

# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(config["seed"])

model_name = config["model_name"]
if model_name == "pretrained_bert":
    config["from_pretrained"] = True
    config["pretrained_model"]="bert-base-chinese"
    config["bert_hidden_size"] = 768
    config["batch_size"] = 8
    config["lr"] = 2e-5

elif model_name == "roberta_large":
    config["from_pretrained"] = True
    config["pretrained_model"]="hfl/chinese-roberta-wwm-ext-large"
    config["bert_hidden_size"] = 1024
    config["batch_size"] = 2
    config["lr"] = 1e-6

elif model_name == "roberta":
    config["from_pretrained"] = True
    config["pretrained_model"]="hfl/chinese-roberta-wwm-ext"
    config["bert_hidden_size"] = 768
    config["batch_size"] = 8
    config["lr"] = 1e-6

elif model_name == "my_bert_768":
    config["from_pretrained"] = False
    config["pretrained_model"]="bert-base-chinese"
    config["bert_hidden_size"] = 768
    config["bert_num_heads"] = 12
    config["bert_num_layers"] = 12
    config["batch_size"] = 8
    config["lr"] = 2e-5

elif model_name == "my_bert_256":
    config["from_pretrained"] = False
    config["pretrained_model"]="bert-base-chinese"
    config["bert_hidden_size"] = 256
    config["bert_num_heads"] = 4
    config["bert_num_layers"] = 6
    config["batch_size"] = 16
    config["lr"] = 2e-5

tokenizer = BertTokenizerFast.from_pretrained(config["pretrained_model"])
dataset_folder = os.getcwd() + "/hw2_dataset/dataset"
ckpt_folder = os.getcwd() + "/ckpt/" + model_name
graph_folder = os.getcwd() + "/graph/" + model_name
prediction_folder = os.getcwd() + "/prediction"
statistic_folder = os.getcwd() + "/statistic/" + model_name
if not os.path.exists(ckpt_folder):
    os.mkdir(ckpt_folder)
if not os.path.exists(statistic_folder):
    os.mkdir(statistic_folder)
if not os.path.exists(graph_folder):
    os.mkdir(graph_folder)
# split my own train/valid data
if config["generate_my_data"]:    
    train_data = None
    with open(dataset_folder + "/train.json") as fp:
        train_data = json.load(fp)

    random.shuffle(train_data)
    train_len = int(round(len(train_data) * (1 - config["val_ratio"])))
    my_train_data = train_data[:train_len]
    my_valid_data = train_data[train_len:]
    with open(dataset_folder + "/my_train.json", "w", encoding='utf8') as fp:
        json.dump(my_train_data, fp, indent = 4, ensure_ascii=False)
    with open(dataset_folder + "/my_valid.json", "w", encoding='utf8') as fp:
        json.dump(my_valid_data, fp, indent = 4, ensure_ascii=False)
else:
    with open(dataset_folder + "/train.json", "r") as fp:
        my_train_data = json.load(fp)
    with open(dataset_folder + "/public.json", "r") as fp:
        my_valid_data = json.load(fp)
with open(args.context, "r") as fp:
    context = json.load(fp)

# tokenize paragraphs
context = tokenizer(context, add_special_tokens=False)

# tokenize my data
data_len = 0
max_q_len = 0
def tokenize_train_data(data):
    global data_len
    global max_q_len
    max_ans_len = 0
    tokenized_data = []
    for i, question in enumerate(tqdm.tqdm(data)):
        max_q_len = max(max_q_len, len(tokenizer(question["question"], add_special_tokens=False)["input_ids"]))
        tokenized_data.append({
            "id": question["id"],
            "paragraph_id": question["relevant"], 
            "question": tokenizer(question["question"], add_special_tokens=False),
            "answer": {"relevant":1,"text":question["answers"][0]["text"],"start": int(question["answers"][0]["start"]), "end":int(question["answers"][0]["start"]) + len(question["answers"][0]["text"]) - 1}
        })
        max_ans_len = max(max_ans_len, len(question["answers"][0]["text"]))
        data_len += 1
        k = 0
        for p in question["paragraphs"]:
            if p != question["relevant"]:
                tokenized_data.append({
                    "id": question["id"],
                    "paragraph_id": p, 
                    "question": tokenizer(question["question"], add_special_tokens=False),
                    "answer": {"relevant":0,"start": -1, "end":-1}
                })
                data_len += 1
                k += 1
            if k >= config["question_number"] - 1:
                break
    print(f"Max answer length: {max_ans_len}")
    return tokenized_data

def tokenize_eval_data(data):
    global data_len
    global max_q_len
    tokenized_data = []
    for i, question in enumerate(tqdm.tqdm(data)):
        max_q_len = max(max_q_len, len(tokenizer(question["question"], add_special_tokens=False)["input_ids"]))
        tokenized_data.append({
            "id": question["id"],
            "paragraph_id": question["paragraphs"], 
            "question": tokenizer(question["question"], add_special_tokens=False),
        })
        data_len += 1
    return tokenized_data

if args.eval:
    train_questions_tokenized = tokenize_train_data(my_train_data)
    dev_questions_tokenized = tokenize_eval_data(my_valid_data)

        
    print(f"Max question length: {max_q_len}")
    print(f"Example tokenized question: {train_questions_tokenized[0]}")
    print(f"Example tokenized paragraph: {context[0]}")
        
    print(f"Training data length: {len(train_questions_tokenized)}")
    print(f"Validation data length: {len(dev_questions_tokenized)}")
    print(f"Total data length: {data_len}")

steps_per_epoch = int(data_len / config["batch_size"])
num_train_steps = steps_per_epoch * config["num_epoch"]

class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_paragraphs):
        super(QA_Dataset, self).__init__()
        self.split = split
        self.questions = questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = config["max_question_len"]
        self.max_paragraph_len = config["max_paragraph_len"]
        
        ##### TODO: Change value of doc_stride #####
        self.doc_stride = config["doc_stride"]

        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]

        ##### TODO: Preprocessing #####
        # Hint: How to prevent model from learning something it should not learn

        if self.split == "train":
            tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]
            if question["answer"]["relevant"]:
                # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph  
                answer_start_token = tokenized_paragraph.char_to_token(question["answer"]["start"])
                answer_end_token = tokenized_paragraph.char_to_token(question["answer"]["end"])

                # A single window is obtained by slicing the portion of paragraph containing the answer
                mid = (answer_start_token + answer_end_token) // 2
                paragraph_start = max(0, min(mid - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
                paragraph_end = paragraph_start + self.max_paragraph_len
            
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + question["question"]["input_ids"][:self.max_question_len] + [102] 
                input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]		
            
                # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window  
                answer_start_token += len(input_ids_question) - paragraph_start
                answer_end_token += len(input_ids_question) - paragraph_start
            
                # Pad sequence and obtain inputs to model 
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), question["answer"]["relevant"], answer_start_token, answer_end_token

            else:
                input_ids_question = [101] + question["question"]["input_ids"][:self.max_question_len] + [102] 
                if len(tokenized_paragraph.ids) <= self.max_paragraph_len:
                    start_pos = 0
                else:
                    start_pos = random.randint(0, len(tokenized_paragraph.ids) - self.max_paragraph_len - 1)
                input_ids_paragraph = tokenized_paragraph.ids[start_pos:start_pos + self.max_paragraph_len] + [102]   
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), question["answer"]["relevant"], -1, -1

        # Validation/Testing
        else:

            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            for paragraph in question["paragraph_id"]:
                tokenized_paragraph = self.tokenized_paragraphs[paragraph]
                # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
                for i in range(0, len(tokenized_paragraph), self.doc_stride):
                    
                    # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                    input_ids_question = [101] + question["question"]["input_ids"][:self.max_question_len] + [102]
                    input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]
                    
                    # Pad sequence and obtain inputs to model
                    input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                    
                    input_ids_list.append(input_ids)
                    token_type_ids_list.append(token_type_ids)
                    attention_mask_list.append(attention_mask)
            
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list), question["id"]

    def padding(self, input_ids_question, input_ids_paragraph):
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        
        return input_ids, token_type_ids, attention_mask

if args.eval:
    train_set = QA_Dataset("train", train_questions_tokenized, context)
    dev_set = QA_Dataset("dev", dev_questions_tokenized, context)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)

class QA_model(nn.Module):
    def __init__(self):
        super(QA_model, self).__init__()
        if config["from_pretrained"]:
            self.bert = BertForQuestionAnswering.from_pretrained(config["pretrained_model"])
        else:
            self.configuration = BertConfig(hidden_size=config["bert_hidden_size"], num_attention_heads=config["bert_num_heads"], num_hidden_layers=config["bert_num_layers"])
            self.bert = BertForQuestionAnswering(self.configuration)
        self.dropout = nn.Dropout(config["dropout"])
        self.rele_cls_layer = nn.Sequential(
            self.dropout,
            nn.Linear(config["bert_hidden_size"], config["hidden_size"]),
            self.dropout,
            nn.Linear(config["hidden_size"], 1)
        )
        '''
        self.start_cls_layer = nn.Sequential(
            self.dropout,
            nn.Linear(768, config["hidden_size"]),
            self.dropout,
            nn.Linear(config["hidden_size"], 1)
        )
        self.end_cls_layer = nn.Sequential(
            self.dropout,
            nn.Linear(768, config["hidden_size"]),
            self.dropout,
            nn.Linear(config["hidden_size"], 1)
        )
        '''

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_out = self.bert(input_ids = input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=True,output_hidden_states=True)
        rele = torch.sigmoid(self.rele_cls_layer(bert_out.hidden_states[-1][:,0,:]))

        return rele, bert_out.start_logits, bert_out.end_logits

def evaluate(data, output):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
    
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    
    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        if not output[1][k].shape[0]:
            continue
        start_prob, start_index = torch.max(output[1][k], dim=0)
        if  start_index != output[1][k].shape[0] - 1:
            end_prob, end_index = torch.max(output[2][k][start_index + 1:], dim=0)
            end_index += start_index + 1
        else:
            end_index = start_index

        # Probability of answer is calculated as sum of start_prob and end_prob
        # prob = start_prob + end_prob
        prob = output[0][k]
        
        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob:
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
    
    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer.replace(' ','')

num_epoch = config["num_epoch"]
validation = True
logging_step = config["log_steps"]
num_warmup_steps = config["warmup_steps"]
model = QA_model().to(device)
# optimizer = AdamW(model.parameters(), lr=config["lr"], correct_bias=False)
# According to RoBERTa, change beta2 for stable training
optimizer = AdamW(model.parameters(), lr=config["lr"], correct_bias=False, betas=(0.9, 0.98))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

model.train()

rele_criterion = nn.BCELoss()
start_criterion = nn.CrossEntropyLoss()
end_criterion = nn.CrossEntropyLoss()
if args.eval:
    print("Start Training ...")
best_f1 = 0
best_em = 0
best_total = 0
start_epoch = 0
# load model
if config["load_ckpt"] and os.path.exists(ckpt_folder + "/model.best"):
    print("Loading best checkpoint ...")
    model.load_state_dict(torch.load(ckpt_folder + "/model.best", map_location=device))
    if os.path.exists(ckpt_folder + "/best.log"):
        with open(ckpt_folder + "/best.log", "r") as fp:
            lines = fp.readlines()
            best_total = float(lines[0])
            start_epoch = int(lines[1]) + 1

for epoch in range(start_epoch, num_epoch):
    step = 0
    acc_step = 0
    train_rele_acc = 0
    train_answer_acc = 0
    rele_total_loss = answer_total_loss = 0
    loss = 0
    
    for data in tqdm.tqdm(train_loader):
        # Load all data into GPU
        data = [i.to(device) for i in data]
        
        # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
        # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)  
        output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2])
        # Choose the most probable start position / end position
        start_index = torch.argmax(output[1], dim=1)
        end_index = torch.argmax(output[2], dim=1)
        
        # Prediction is correct only if both start_index and end_index are correct
        answer_loss = 0
        rele_loss = rele_criterion(output[0].squeeze(1), data[3].float())
        rele_total_loss += rele_loss
        train_rele_acc += (data[3] == (output[0] >= 0.5).squeeze(1)).float().mean()
        for i, rele in enumerate(data[3]):
            if rele:
                answer_loss += start_criterion(output[1][i,:].unsqueeze(0) ,data[4][i].unsqueeze(0)) + end_criterion(output[2][i,:].unsqueeze(0) ,data[5][i].unsqueeze(0))
                train_answer_acc += ((start_index[i].item() == data[4][i].item()) & (end_index[i].item() == data[5][i].item()))
                acc_step += 1
        answer_total_loss += answer_loss
        loss = (rele_loss + answer_loss) / logging_step
        loss.backward()        
        
        
        with open(statistic_folder + f"/loss_{epoch}", "a") as fp:
            print(loss.item(), file=fp)
        step += 1

        # Print training loss and accuracy over past logging step
        if step % logging_step == 0:
            print(f"Epoch {epoch + 1} | Step {step} | relevant loss = {rele_total_loss.item() / logging_step:.3f} , answer loss = {answer_total_loss.item() / acc_step:.3f} , relevant acc = {train_rele_acc / logging_step :.3f}, answer acc = {train_answer_acc / acc_step :.3f}")
            train_rele_acc = train_answer_acc = 0
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            answer_total_loss = rele_total_loss = 0
            acc_step = 0


            if validation and epoch > 3 and step % config["eval_steps"] == 0:
                print("Evaluating Dev Set ...")
                model.eval()
                with torch.no_grad():
                    valid_pred = {}
                    for i, data in enumerate(tqdm.tqdm(dev_loader)):
                        output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                            attention_mask=data[2].squeeze(dim=0).to(device))
                        eva_text = evaluate(data, output)
                        valid_pred[data[3][0]] = eva_text
                        
                    with open(prediction_folder + f"/valid_{epoch}.json", "w", encoding='utf8') as fp:
                        json.dump(valid_pred, fp, indent = 4, ensure_ascii=False)
                    eva_res = eva.main(dataset_folder + "/public.json", prediction_folder + f"/valid_{epoch}.json", statistic_folder + f"/valid_{epoch}.json")
                    if eva_res["f1"] >= best_f1:
                        best_f1 = eva_res["f1"]
                    if eva_res["em"] >= best_em:
                        best_em = eva_res["em"]
                    print(f"Saving Validation Model with | f1: {eva_res['f1']} | em: {eva_res['em']} | total: {eva_res['f1'] + eva_res['em']}")
                    torch.save(model.state_dict(), ckpt_folder + f"/model.{epoch}")
                    if eva_res["em"] + eva_res["f1"] >= best_total:
                        best_total = eva_res["f1"] + eva_res["em"]
                        print(f"Saving best Model with total:  {best_total}")
                        torch.save(model.state_dict(), ckpt_folder + "/model.best")
                        with open(ckpt_folder + "/best.log", "w") as fp:
                            print(best_total, file = fp)  
                            print(epoch, file = fp)  
                model.train()

# Evaluate on public test data
with open(args.test, "r") as fp:
    test_data = json.load(fp)

tokenized_test_data = tokenize_eval_data(test_data)
test_set = QA_Dataset("test", tokenized_test_data, context)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)
print("Evaluate test data ......")
model.eval()
with torch.no_grad():
    test_pred = {}
    for i, data in enumerate(tqdm.tqdm(test_loader)):
        output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
            attention_mask=data[2].squeeze(dim=0).to(device))
        eva_text = evaluate(data, output)
        test_pred[data[3][0]] = eva_text
        # Visualize my QA model
        
    with open(args.prediction, "w", encoding='utf8') as fp:
        json.dump(test_pred, fp, indent = 4, ensure_ascii=False)
    if args.eval:
        eva_res = eva.main(dataset_folder + "/public.json", prediction_folder + f"/public.json", statistic_folder + f"/public.json")

if args.eval:
    # Evaluate on private test data
    with open(dataset_folder + "/private.json", "r") as fp:
        test_data = json.load(fp)
    tokenized_test_data = tokenize_eval_data(test_data)
    test_set = QA_Dataset("test", tokenized_test_data, context)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)
    print("Evaluate Private test data ......")
    model.eval()
    with torch.no_grad():
        test_pred = {}
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                attention_mask=data[2].squeeze(dim=0).to(device))
            eva_text = evaluate(data, output)
            test_pred[data[3][0]] = eva_text
            
        with open(prediction_folder + f"/private.json", "w", encoding='utf8') as fp:
            json.dump(test_pred, fp, indent = 4, ensure_ascii=False)

