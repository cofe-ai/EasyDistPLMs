# -*- coding:UTF-8 -*-
from src.dataloaders import init_dataloader
from src.datasets import SNLIDataset, IMDBDataset, LabelMap
from src.model import BertCLSModel
from src.utils import LogMessage, torch_distributed_master_process_first, get_ds_config
import torch
import deepspeed
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from filelock import FileLock
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import random
import wandb
import logging

class Trainer:
    def __init__(self, args):
        self.ds_config = get_ds_config(args.deepspeed_config)
        self.timestamp = args.timestamp
        self.fix_seed(args.seed)
        deepspeed.init_distributed()
        self.is_distributed = (args.local_rank != -1)
        self.log_info_message = LogMessage(self.is_distributed)
        self.log_info_message(f"distributed enabled: {self.is_distributed}")
        self.label_map = LabelMap(args.label_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
        self.max_epochs = args.max_epochs
        self.patience = args.patience
        self.train_loader, self.valid_loader = self.prepare_data(train_path=args.train_path,
                                                                 valid_path=args.valid_path,
                                                                 pre_tokenized=args.pre_tokenized,
                                                                 shuffle_train_data=args.shuffle,
                                                                 num_workers=args.num_data_workers,
                                                                 max_seq_len=args.max_seq_len,
                                                                 batch_size=self.ds_config["train_micro_batch_size_per_gpu"],
                                                                 save_cache=args.save_cache,
                                                                 use_cache=args.use_cache)
        self.model, self.optimizer, self.lr_scheduler = self.setup_model_optimizer_and_scheduler(args)
        self.device = self.setup_device(args.local_rank)
        self.log_interval = args.log_interval

        self.use_wandb = args.use_wandb
        if self.use_wandb:
            os.environ["WANDB_API_KEY"] = args.wandb_key
            wandb.init(project=args.task_name, group="torch")
        self.ckpt_path = os.path.join(args.ckpt_path, self.timestamp)
        with FileLock(os.path.expanduser("~/.deepspeed_lock")):
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
        self.best_metric = 0.0
        self.best_epoch = 0


    def setup_device(self, local_rank):
        if torch.cuda.is_available():
            if torch.distributed.is_initialized():
                device = torch.device("cuda", local_rank)
            else:
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logging.info(f"setup device: {device}")
        return device

    def fix_seed(self, seed):
        torch.manual_seed(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(seed)

    def setup_model_optimizer_and_scheduler(self, args):
        model = BertCLSModel(pretrained_path=args.pretrained,
                             num_labels=self.label_map.num_labels,
                             dropout_prob=args.dropout,
                             pooler_type=args.pooler_type)
        grouped_model_params = self.group_optim_params(model, weight_decay=args.weight_decay)
        model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                             model=model,
                                                             model_parameters=grouped_model_params)
        lr_scheduler = self.init_scheduler(optimizer=optimizer,
                                           num_steps_per_epoch=len(self.train_loader),
                                           warmup_ratio=args.warmup)
        return model_engine, optimizer, lr_scheduler

    def group_optim_params(self, model, weight_decay):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def init_scheduler(self, optimizer, num_steps_per_epoch, warmup_ratio):
        total_train_steps = num_steps_per_epoch * self.max_epochs
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=total_train_steps * warmup_ratio,
            num_training_steps=total_train_steps,
        )
        self.log_info_message("setup lr_scheduler")
        return lr_scheduler


    def prepare_data(self,
                     train_path: str,
                     valid_path: str,
                     pre_tokenized: bool,
                     shuffle_train_data: bool,
                     num_workers: int,
                     max_seq_len: int,
                     batch_size: int,
                     save_cache: bool,
                     use_cache: bool
                     ):
        train_set = IMDBDataset(data_type="train", path=train_path, label_map=self.label_map,
                                tokenizer=self.tokenizer, pre_tokenize=pre_tokenized,
                                max_seq_len=max_seq_len, save_cache=save_cache, use_cache=use_cache)

        train_loader = init_dataloader(dataset=train_set,
                                       shuffle=shuffle_train_data,
                                       batch_size=batch_size,
                                       input_pad_id=self.tokenizer.pad_token_id,
                                       num_workers=num_workers,
                                       is_distributed=self.is_distributed
                                       )
        valid_loader = None
        if valid_path is not None:
            valid_set = IMDBDataset(data_type="valid", path=valid_path, label_map=self.label_map,
                                    tokenizer=self.tokenizer, pre_tokenize=pre_tokenized,
                                    max_seq_len=max_seq_len, save_cache=save_cache, use_cache=use_cache)

            valid_loader = init_dataloader(dataset=valid_set,
                                           shuffle=False,
                                           batch_size=batch_size,
                                           input_pad_id=self.tokenizer.pad_token_id,
                                           num_workers=0,  # disable multiprocessing on valid set data loading
                                           is_distributed=self.is_distributed
                                           )
        return train_loader, valid_loader


    def eval(self):
        total_preds = []
        total_golds = []
        total_loss = 0.
        with torch.no_grad():
            for batch in self.valid_loader:
                if torch.cuda.is_available():
                    self.batch2cuda(batch, ignore_keys=["uuid"])
                loss, logits = self.model(batch)
                batch_golds = batch["label"].cpu()
                batch_preds = torch.argmax(logits.cpu(), dim=-1)
                total_golds.extend(batch_golds.tolist())
                total_preds.extend(batch_preds.tolist())
                total_loss += loss.item()
        # use sampler to determine the number of examples in this worker's partition.
        total_loss /= len(self.valid_loader.sampler)
        acc = accuracy_score(total_golds, total_preds)
        return acc, total_loss

    def update_metrics(self, valid_acc, epoch):
        if epoch == 0 or epoch != self.best_epoch:
            if self.best_metric < valid_acc:
                self.best_metric = valid_acc
                self.best_epoch = epoch

    def batch2cuda(self, batch, ignore_keys=["uuid"]):
        for k, v in batch.items():
            if k in ignore_keys:
                continue
            batch[k] = v.to(self.device)

    def save_ckpt(self):
        torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, "best.ckpt"))
        logging.info(f"successfully saved ckpt to {self.ckpt_path}")

    def train(self):

        current_step = 0
        no_improve = 0
        for epoch in range(self.max_epochs):
            num_steps_one_epoch = len(self.train_loader)
            pbar = tqdm(total=num_steps_one_epoch)
            # In distributed mode, calling the set_epoch() method at the beginning of each epoch 
            # before creating the DataLoader iterator is necessary
            # to make shuffling work properly across multiple epochs.
            # Otherwise, the same ordering will be always used.
            if self.is_distributed:
                self.train_loader.sampler.set_epoch(epoch)
            self.model.train()
            for batch in self.train_loader:
                if torch.cuda.is_available():
                    self.batch2cuda(batch, ignore_keys=["uuid"])
                loss, logits = self.model(batch)
                self.model.backward(loss)
                self.model.step()
                if self.model.is_gradient_accumulation_boundary():
                    self.lr_scheduler.step()

                loss = loss.detach().item()
                current_lr = self.lr_scheduler.get_last_lr()[0]


                if self.use_wandb:
                    wandb.log({"loss": loss, "lr": current_lr})
                # logging on rank 0
                if (self.is_distributed and torch.distributed.get_rank() == 0) or not self.is_distributed:
                    current_step += 1
                    if current_step % self.log_interval == 0 or current_step % num_steps_one_epoch == 0:

                        update_steps = self.log_interval
                        if current_step % num_steps_one_epoch == 0:
                            update_steps = current_step % self.log_interval
                            if update_steps == 0:
                                update_steps = self.log_interval

                        logging.info(f"step {current_step}, current loss: {loss}, current_lr: {current_lr}")
                        pbar.update(update_steps)

            pbar.close()
            self.log_info_message(f"epoch {epoch} training finished.")
            self.model.eval()
            valid_acc, valid_loss = self.eval()

            if self.use_wandb:
                wandb.log({"valid_loss": loss, "valid_acc": valid_acc})
            if self.is_distributed:
                with torch_distributed_master_process_first(torch.distributed.get_rank()):
                    self.update_metrics(valid_acc, epoch)
            else:
                self.update_metrics(valid_acc, epoch)

            # save best epoch
            if self.best_epoch == epoch:
                self.save_ckpt()

                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    self.log_info_message(f"no improve within {no_improve} epochs, early stop")
                    break

            self.log_info_message(f"epoch {epoch} finished. \
                            valid acc: {valid_acc}, valid loss: {valid_loss}")
            self.log_info_message(f"current best acc {self.best_metric}, in epoch {self.best_epoch}")
