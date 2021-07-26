import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from tqdm import trange, tqdm

from slot_dataset import SeqClsDataset
from transformers import AdamW, BertForTokenClassification, BertTokenizerFast

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    torch.manual_seed(1)
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased")
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, tag2idx, args.max_len, tokenizer, "train")
        for split, split_data in data.items()
    }
    # TODO: create DataLoader for train / dev datasets
    train_loader = torch.utils.data.DataLoader(datasets[TRAIN], batch_size = args.batch_size, shuffle = True, collate_fn = datasets[TRAIN].collate_fn)
    dev_loader = torch.utils.data.DataLoader(datasets[DEV],batch_size = args.batch_size, shuffle = False, collate_fn = datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=10)
    device = args.device

    try:
        ckpt = torch.load("./ckpt/slot/model.ckpt")
        model.load_state_dict(ckpt)
    except:
        print("Can't load model!")
    batch_size = args.batch_size
    # TODO: init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)

    model.to(device)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        tr = tqdm(train_loader)
        for i, d in enumerate(tr):
            data = [torch.tensor(j).to(device) for j in d[1:]]
            optimizer.zero_grad()
            out = model(input_ids = data[0], token_type_ids=data[1],attention_mask=data[2],labels=data[3])
            _, train_pred = torch.max(out.logits, 2)
            loss = out.loss
            loss.backward()
            optimizer.step()

            for j, label in enumerate(data[3]):
                if 9 in train_pred[j][1:].tolist():
                    end_index = train_pred[j][1:].cpu().tolist().index(9) + 1
                else:
                    end_index = len(train_pred[j].tolist())
                data_end_index = data[3][j][1:].cpu().tolist().index(9) + 1
                train_acc += (train_pred[j][1:end_index].cpu().tolist() == data[3][j][1:data_end_index].cpu().tolist())
            train_loss += loss.item()
            tr.clear()

        # TODO: Evaluation loop - calculate accuracy and save model weights
        with torch.no_grad():
            #h = model.init_hidden(batch_size, device)
            model.eval()
            for i, dev_d in enumerate(tqdm(dev_loader)):
                dev_data = [torch.tensor(j).to(device) for j in dev_d[1:]]
                out = model(input_ids = dev_data[0], token_type_ids=dev_data[1],attention_mask=dev_data[2],labels=dev_data[3])

                loss = out.loss 
                _, val_pred = torch.max(out.logits, 2)
                for j, label in enumerate(dev_data[3]):
                    end_index = val_pred[j][1:].cpu().tolist().index(9) + 1
                    data_end_index = dev_data[3][j][1:].cpu().tolist().index(9) + 1
                    val_acc += (val_pred[j][1:end_index].cpu().tolist() == dev_data[3][j][1:data_end_index].cpu().tolist())
                val_loss += loss.item()
            
            print(f"Epoch {epoch + 1}: Train Acc: {train_acc / len(train_loader.dataset)}, Train Loss: {train_loss / len(train_loader)}, Val Acc: {val_acc / len(dev_loader.dataset)}, Val Loss: {val_loss / len(dev_loader)}")
            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(),"./ckpt/slot/model.ckpt")
                print(f"Save model with acc {val_acc / len(dev_loader.dataset):.6f}")
                

    # TODO: Inference on test set
    tt_data = None
    with open("../../../ADL-HW1/data/slot/test.json", "r") as fp:
        tt_data = json.load(fp)
    tt_dataset = SeqClsDataset(tt_data, tag2idx, args.max_len, tokenizer, "test")
    # TODO: create DataLoader for test dataset
    tt_loader = torch.utils.data.DataLoader(tt_dataset, shuffle = False, batch_size = 150, collate_fn = tt_dataset.collate_fn)
    model.eval()
    model.load_state_dict(torch.load("./ckpt/slot/model.ckpt"))
    # load weights into model

    # TODO: predict dataset
    preds = []
    with open("./pred.slot.csv", "w") as fp:
        fp.write("id,tags\n")
        with torch.no_grad():
            for i, d in enumerate(tqdm(tt_loader)):
                ids = d[0]
                data = [torch.tensor(j).to(device) for j in d[1:]]
                out = model(input_ids = data[0], token_type_ids=data[1],attention_mask=data[2])
                _, pred = torch.max(out.logits, 2)
                for j, p in enumerate(pred):
                    data_end_index = p[1:].cpu().tolist().index(9) + 1
                    fp.write(f"{ids[j]},{' '.join(list(map(lambda x:tt_dataset.idx2label(x), list(filter(lambda x: (x != 9), p[1:data_end_index].tolist())))))}\n")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="../../../ADL-HW1/data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="../../../ADL-HW1/cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    # training
    parser.add_argument(
            "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:3"
    )
    parser.add_argument("--num_epoch", type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
