#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
from functools import partial
from attr import has
from pypinyin import pinyin

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup

from datasets.bert_csc_dataset import TestCSCDataset, Dynaimic_CSCDataset
from models.modeling_multitask import Dynamic_GlyceBertForMultiTask
from utils.random_seed import set_random_seed
from datasets.collate_functions import collate_to_max_length_with_id,collate_to_max_length_for_train_dynamic_pron_loss


set_random_seed(2333)

def decode_sentence_and_get_pinyinids(ids):
    dataset = TestCSCDataset(
        data_path='data/test.sighan15.pkl',
        chinese_bert_path='FPT',
    )
    sent = ''.join(dataset.tokenizer.decode(ids).split(' '))
    tokenizer_output = dataset.tokenizer.encode(sent)
    pinyin_tokens = dataset.convert_sentence_to_pinyin_ids(sent, tokenizer_output)
    pinyin_ids = torch.LongTensor(pinyin_tokens).unsqueeze(0)
    return sent,pinyin_ids

class CSCTask(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        """Initialize a models, tokenizer and config."""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        self.bert_dir = args.bert_path
        self.bert_config = BertConfig.from_pretrained(
            self.bert_dir, output_hidden_states=False
        )
        self.model = Dynamic_GlyceBertForMultiTask.from_pretrained(self.bert_dir)
        if args.ckpt_path is not None:
            print("loading from ", args.ckpt_path)
            ckpt = torch.load(args.ckpt_path,)["state_dict"]
            new_ckpt = {}
            for key in ckpt.keys():
                new_ckpt[key[6:]] = ckpt[key]
            self.model.load_state_dict(new_ckpt,strict=False)
            print(self.model.device, torch.cuda.is_available())
        self.vocab_size = self.bert_config.vocab_size

        self.loss_fct = CrossEntropyLoss()
        gpus_string = (
            str(self.args.gpus) if not str(self.args.gpus).endswith(",") else str(self.args.gpus)[:-1]
        )
        self.num_gpus = len(gpus_string.split(","))

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.98),  # according to RoBERTa paper
            lr=self.args.lr,
            eps=self.args.adam_epsilon,
        )
        t_total = (
            len(self.train_dataloader())
            // self.args.accumulate_grad_batches
            * self.args.max_epochs
        )
        warmup_steps = int(self.args.warmup_proporation * t_total)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, pinyin_ids, labels=None, pinyin_labels=None, tgt_pinyin_ids=None, var=1):
        """"""
        attention_mask = (input_ids != 0).long()
        return self.model(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            labels=labels,
            tgt_pinyin_ids=tgt_pinyin_ids, 
            pinyin_labels=pinyin_labels,
            gamma=self.args.gamma if 'gamma' in self.args else 0,
        )

    def compute_loss(self, batch):
        input_ids, pinyin_ids, labels, tgt_pinyin_ids, pinyin_labels = batch
        loss_mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        batch_size, length = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        tgt_pinyin_ids = tgt_pinyin_ids.view(batch_size, length, 8)
        outputs = self.forward(
            input_ids, pinyin_ids, labels=labels, pinyin_labels=pinyin_labels, tgt_pinyin_ids=tgt_pinyin_ids, 
            var= self.args.var if 'var' in self.args else 1
        )
        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx):
        """"""
        loss = self.compute_loss(batch)
        tf_board_logs = {
            "train_loss": loss.item(),
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        # torch.cuda.empty_cache()
        return {"loss": loss, "log": tf_board_logs}

    def validation_step(self, batch, batch_idx):
        """"""
        input_ids, pinyin_ids, labels, pinyin_labels, ids, srcs, tokens_size = batch
        mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        batch_size, length = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        logits = self.forward(
            input_ids,
            pinyin_ids,
        ).logits
        predict_scores = F.softmax(logits, dim=-1)
        predict_labels = torch.argmax(predict_scores, dim=-1) * mask
        return {
            "tgt_idx": labels.cpu(),
            "pred_idx": predict_labels.cpu(),
            "id": ids,
            "src": srcs,
            "tokens_size": tokens_size,
        }

    def validation_epoch_end(self, outputs):
        from metrics.metric import Metric

        # print(len(outputs))
        metric = Metric(vocab_path=self.args.bert_path)
        pred_txt_path = os.path.join(self.args.save_path, "preds.txt")
        pred_lbl_path = os.path.join(self.args.save_path, "labels.txt")
        if len(outputs) == 2:
            self.log("df", 0)
            self.log("cf", 0)
            return {"df": 0, "cf": 0}
        results = metric.metric(
            batches=outputs,
            pred_txt_path=pred_txt_path,
            pred_lbl_path=pred_lbl_path,
            label_path=self.args.label_file,
        )
        self.log("df", results["sent-detect-f1"])
        self.log("cf", results["sent-correct-f1"])
        return {"df": results["sent-detect-f1"], "cf": results["sent-correct-f1"]}

    def train_dataloader(self) -> DataLoader:
        name = "train_all"

        dataset = Dynaimic_CSCDataset(
            data_path=os.path.join(self.args.data_dir, name),
            chinese_bert_path=self.args.bert_path,
            max_length=self.args.max_length,
        )
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = dataset.tokenizer

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_for_train_dynamic_pron_loss, fill_values=[0, 0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader

    def val_dataloader(self):
        dataset = TestCSCDataset(
            data_path='data/test.sighan15.pkl',
            chinese_bert_path=self.args.bert_path,
            max_length=self.args.max_length,
        )
        print('dev dataset', len(dataset))
        self.tokenizer = dataset.tokenizer
        from datasets.collate_functions import collate_to_max_length_with_id

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader

    def test13_dataloader(self):
        dataset = TestCSCDataset(
            data_path='data/test.sighan13.pkl',
            chinese_bert_path=self.args.bert_path,
            max_length=self.args.max_length,
        )
        self.tokenizer = dataset.tokenizer
        from datasets.collate_functions import collate_to_max_length_with_id

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader
    
    def test14_dataloader(self):
        dataset = TestCSCDataset(
            data_path='data/test.sighan14.pkl',
            chinese_bert_path=self.args.bert_path,
            max_length=self.args.max_length,
        )
        self.tokenizer = dataset.tokenizer
        from datasets.collate_functions import collate_to_max_length_with_id

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader

    def test15_dataloader(self):
        dataset = TestCSCDataset(
            data_path='data/test.sighan15.pkl',
            chinese_bert_path=self.args.bert_path,
            max_length=self.args.max_length,
        )
        self.tokenizer = dataset.tokenizer
        from datasets.collate_functions import collate_to_max_length_with_id

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids, pinyin_ids, labels, pinyin_labels, ids, srcs, tokens_size = batch
        mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        batch_size, length = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        logits = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids).logits
        predict_scores = F.softmax(logits, dim=-1)
        predict_labels = torch.argmax(predict_scores, dim=-1) * mask
        

        if '13' in self.args.label_file:
            predict_labels[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))] = \
                input_ids[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))]
        
        pre_predict_labels = predict_labels
        for _ in range(1):
            record_index = []
            for i,(a,b) in enumerate(zip(list(input_ids[0,1:-1]),list(predict_labels[0,1:-1]))):
                if a!=b:
                    record_index.append(i)
            
            input_ids[0,1:-1] = predict_labels[0,1:-1]
            sent, new_pinyin_ids = decode_sentence_and_get_pinyinids(input_ids[0,1:-1].cpu().numpy().tolist())
            if new_pinyin_ids.shape[1] == input_ids.shape[1]:
                pinyin_ids = new_pinyin_ids
            pinyin_ids = pinyin_ids.to(input_ids.device)
            # print(input_ids.device, pinyin_ids.device)
            logits = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids).logits
            predict_scores = F.softmax(logits, dim=-1)
            predict_labels = torch.argmax(predict_scores, dim=-1) * mask

            for i,(a,b) in enumerate(zip(list(input_ids[0,1:-1]),list(predict_labels[0,1:-1]))):
                if a!=b and any([abs(i-x)<=1 for x in record_index]):
                    print(ids,srcs)
                    print(i+1,)
                else:
                    predict_labels[0,i+1] = input_ids[0,i+1]
            if predict_labels[0,i+1] == input_ids[0,i+1]:
                break
            if '13' in self.args.label_file:
                predict_labels[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))] = \
                    input_ids[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))]
        # if not pre_predict_labels.equal(predict_labels):
        #     print([self.tokenizer.id_to_token(id) for id in pre_predict_labels[0][1:-1]])
        #     print([self.tokenizer.id_to_token(id) for id in predict_labels[0][1:-1]])
        return {
            "tgt_idx": labels.cpu(),
            "post_pred_idx": predict_labels.cpu(),
            "pred_idx": pre_predict_labels.cpu(),
            "id": ids,
            "src": srcs,
            "tokens_size": tokens_size,
        }


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument(
        "--label_file",
        default="data/test.sighan15.lbl.tsv",
        type=str,
        help="label file",
    )
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument(
        "--workers", type=int, default=8, help="num workers for dataloader"
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument(
        "--use_memory",
        action="store_true",
        help="load datasets to memory to accelerate.",
    )
    parser.add_argument(
        "--max_length", default=512, type=int, help="max length of datasets"
    )
    parser.add_argument("--checkpoint_path", type=str, help="train checkpoint")
    parser.add_argument(
        "--save_topk", default=5, type=int, help="save topk checkpoint"
    )
    parser.add_argument("--mode", default="train", type=str, help="train or evaluate")
    parser.add_argument(
        "--warmup_proporation", default=0.01, type=float, help="warmup proporation"
    )
    parser.add_argument("--gamma", default=1, type=float, help="phonetic loss weight")
    parser.add_argument(
        "--ckpt_path", default=None, type=str, help="resume_from_checkpoint"
    )
    return parser


def main():
    """main"""
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # create save path if doesn't exit
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model = CSCTask(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_path, "checkpoint"),
        filename="{epoch}-{df:.4f}-{cf:.4f}",
        save_top_k=args.save_topk,
        monitor="cf",
        mode="max",
    )
    logger = TensorBoardLogger(save_dir=args.save_path, name="log")

    # save args
    if not os.path.exists(os.path.join(args.save_path, "checkpoint")):
        os.mkdir(os.path.join(args.save_path, "checkpoint"))
    with open(os.path.join(args.save_path, "args.json"), "w") as f:
        args_dict = args.__dict__
        del args_dict["tpu_cores"]
        json.dump(args_dict, f, indent=4)

    trainer = Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback], logger=logger
    )

    trainer.fit(model)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
