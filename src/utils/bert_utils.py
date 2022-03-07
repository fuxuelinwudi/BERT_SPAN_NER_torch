# coding:utf-8

import os
import random
import numpy as np
import torch
from collections import defaultdict

from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


class FGM:
    def __init__(self, args, model):
        self.model = model
        self.backup = {}
        self.emb_name = args.emb_name
        self.epsilon = args.epsilon

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, args, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = args.epsilon
        self.emb_name = args.emb_name
        self.alpha = args.alpha

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class FreeLB():
    def __init__(self, args, model, optimizer, batch_cuda, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.batch_cuda = batch_cuda
        self.inputs = {"attention_mask": batch_cuda['attention_mask'], "labels": batch_cuda['labels'],
                       "token_type_ids": batch_cuda['token_type_ids'], "boundary_ids": batch_cuda['boundary_ids'],
                       "matched_word_ids": batch_cuda['matched_word_ids'],
                       "matched_word_mask": batch_cuda['matched_word_mask']}
        self.scaler = scaler

    def attack(self):
        # ============================ Code for adversarial training=============
        # initialize delta
        if isinstance(self.model, torch.nn.DataParallel):
            embeds_init = self.model.module.bert.embeddings.word_embeddings(self.batch_cuda['input_ids'])
        else:
            embeds_init = self.model.bert.embeddings.word_embeddings(self.batch_cuda['input_ids'])

        if self.args.adv_init_mag > 0:

            input_mask = self.inputs['attention_mask'].to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            # check the shape of the mask here..

            if self.args.norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.args.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.args.norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.args.adv_init_mag,
                                                               self.args.adv_init_mag) * input_mask.unsqueeze(2)

        else:
            delta = torch.zeros_like(embeds_init)

        # the main loop
        for astep in range(self.args.adv_steps):
            # (0) forward
            delta.requires_grad_()
            self.inputs['inputs_embeds'] = delta + embeds_init

            if self.args.use_fp16:
                with autocast():
                    adv_loss = self.model(**self.inputs)[0]
            else:
                adv_loss = self.model(**self.inputs)[0]

            adv_loss = adv_loss / self.args.adv_steps

            if self.args.use_fp16:
                self.scaler.scale(adv_loss).backward()
            else:
                adv_loss.backward()

            if astep == self.args.adv_steps - 1:
                # further updates on delta
                break

            # (2) get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # (3) update and clip
            if self.args.norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.args.adv_lr * delta_grad / denorm).detach()
                if self.args.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.args.adv_max_norm).to(embeds_init)
                    reweights = (self.args.adv_max_norm / delta_norm * exceed_mask \
                                 + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.args.norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.args.adv_lr * delta_grad / denorm).detach()
                if self.args.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.args.adv_max_norm, self.args.adv_max_norm).detach()
            else:
                print("Norm type {} not specified.".format(self.args.norm_type))
                exit()

            if isinstance(self.model, torch.nn.DataParallel):
                embeds_init = self.model.module.bert.embeddings.word_embeddings(self.batch_cuda['input_ids'])
            else:
                embeds_init = self.model.bert.embeddings.word_embeddings(self.batch_cuda['input_ids'])


def save_model(args, model, tokenizer):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.output_path)
    tokenizer.save_vocabulary(args.output_path)

    torch.save(args, os.path.join(args.output_path, 'training_config.bin'))


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
