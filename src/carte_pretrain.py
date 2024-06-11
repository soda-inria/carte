"""
CARTE pretrain with knowledge graphs (YAGO).

"""

import os
import torch
import datetime
import pickle
import math

from time import time

from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.data import Batch

from configs.directory import config_directory
from src.carte_model import CARTE_Pretrain
from src.carte_yago_graphlet_construction import Graphlet


class Infonce_loss(_Loss):
    """InfoNCE Loss"""
    def __init__(self, tau: float) -> None:
        super(Infonce_loss, self).__init__()

        self.tau = tau

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return _infonce(input, target, self.tau)


def _infonce(x: torch.tensor, target: Tensor, tau: float = 1.0):
    """Function to calculate the InfoNCE Loss"""
    x_ = x.clone()
    x_ = x_ / tau
    pos_mask = (target - torch.eye(target.size(0), device=x.device)) > 0
    self_mask = torch.eye(x.size(0), dtype=torch.bool, device=x.device)
    x_.masked_fill_(self_mask, -9e15)
    loss = -x_ * pos_mask + torch.logsumexp(x_, dim=0)
    loss = loss.mean()
    return loss


def _create_target(data):
    """Target construction for the pretraining"""
    graph_idx = data.g_idx
    pos_mask = (
        graph_idx.repeat(graph_idx.size(0), 1)
        - graph_idx.repeat(graph_idx.size(0), 1).t()
    )
    target = pos_mask.clone()
    target[pos_mask == 0] = 1
    target[pos_mask != 0] = 0
    return target

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """Cosine Scheduler"""
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [
                (self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.eta_max - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class Index_extractor:
    """Index sampler to sample batches"""
    def __init__(
        self, main_data, num_rel: int = 6, per: float = 0.9, max_nodes: int = 100
    ):
        self.count_h = main_data.edge_index[0].bincount()
        self.list_u = (self.count_h > num_rel - 1).nonzero().view(-1)
        self.list_d = (self.count_h < num_rel).nonzero().view(-1)

        self.prob = dict()
        self.prob["u"] = torch.ceil(
            self.count_h[self.list_u].to(torch.float) / max_nodes
        )
        self.prob["d"] = torch.ceil(
            self.count_h[self.list_d].to(torch.float) / max_nodes
        )
        self.prob_original = self.prob.copy()

        self.per = per

    def reset(self, up: bool):
        if up:
            self.prob["u"] = self.prob_original["u"].clone()
        else:
            self.prob["d"] = self.prob_original["d"].clone()

    def sample(self, n_batch: int):
        num_u = math.ceil(n_batch * self.per)
        num_d = n_batch - num_u

        if num_u > self.prob["u"].nonzero().size(0):
            self.reset(up=True)
        idx_sample_u = torch.multinomial(self.prob["u"], num_u)
        # self.prob["u"][idx_sample_u] -= 1
        idx_sample_u = self.list_u[idx_sample_u]

        if num_d > self.prob["d"].nonzero().size(0):
            self.reset(up=False)
        idx_sample_d = torch.multinomial(self.prob["d"], num_d)
        # self.prob["d"][idx_sample_d] -= 1
        idx_sample_d = self.list_d[idx_sample_d]

        idx_sample = torch.hstack((idx_sample_u, idx_sample_d))
        return idx_sample


class CARTE_KGPretrain:
    """Pretrainer Class to pretrain CARTE with large knowledge graphs"""
    def __init__(
        self,
        num_layers: int = 0,
        load_pretrained: bool = False,
        batch_size: int = 128,
        learning_rate: float = 1e-5,
        num_hops: int = 1,
        max_nodes: int = 15,
        num_perturb: int = 1,
        fraction_perturb: float = 0.5,
        num_steps: int = 100000,
        save_every: int = 1000,
        device: str = "cuda:0",
    ):
        self.num_layers = num_layers
        self.load_pretrained = load_pretrained
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_hops = num_hops
        self.max_nodes = max_nodes
        self.num_perturb = num_perturb
        self.fraction_perturb = fraction_perturb
        self.num_steps = num_steps
        self.save_every = save_every
        self.device = device


    def fit(self, X, domain_name):

        # Preliminary settings
        self.is_fitted_ = False
        self._load_graphlet_construction()
        self.make_batch_ = Batch()
        self.criterion_ = Infonce_loss(tau=1.0)
        self.device_ = torch.device(self.device)
        self.log_ = []

        # Load model, optimizer, and scheduler
        model_run_pretrain = self._load_model()
        model_run_pretrain.to(self.device_)
        optimizer = torch.optim.AdamW(
            model_run_pretrain.parameters(), lr=self.learning_rate
        )
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0=self.num_steps,
            T_mult=1,
            eta_max=1e-4,
            T_up=10000,
            gamma=1,
        )

        # Run train
        step = 0

        idx = self.idx_extract.sample(n_batch=self.n_batch)

        while step < self.num_steps:
            self._run_step(X, model_run_pretrain, scheduler, step)
            step += 1
            if step % self.save_every == 0:
                self._save_checkpoint(model_run_pretrain, step, domain_name)


    def _load_graphlet_construction(self):
        """Load yago data and initialize graphlet constructor"""
        yago_data_dir = f"{config_directory['data_yago']}/yago3_2022_num.pickle"
        with open(yago_data_dir, "rb") as pickle_file:
            data_yago = pickle.load(pickle_file)
        self.gc_ = Graphlet(data_kg=data_yago, num_hops=self.num_hops, max_nodes=self.max_nodes)
        return None

    def _run_step(self, entity_idx, data, model_run_pretrain, optimizer, scheduler, step):
        """A step for feed-forward and backprop"""

        # Set
        start_time = time()
        data = self.gc_.make_batch(entity_idx, self.num_perturb, self.fraction_perturb)
        data_batch = self._collate_batch(data)
        target = _create_target(data_batch)

        # Send to device
        target.to(self.device_)
        data_batch.to(self.device_)  

        # Set the network and optimizers
        model_run_pretrain.train()
        optimizer.zero_grad()

        # Feed-forward and backprop
        out = model_run_pretrain(data)  # Perform a single forward pass.
        loss = self.criterion_(out, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        end_time = time()
        duration = round(end_time - start_time, 4)

        loss = round(loss.detach().item(), 4)
        self.log_.append(f"Step: {step} | Loss(node): {loss} | Duration: {duration}")
        print(
            f"[GPU: {self.device}] | Step: {step} | Loss(node): {loss} | Duration: {duration}"
        )
        del (loss, out, target, data)


    def _collate_batch(self, data):
        """Collate batch to make it suitable inputs"""
        with torch.no_grad():
            data_batch = self.make_batch_.from_data_list(data, follow_batch=["edge_index"])
        return data_batch

    def _load_model(self):
        """Load the CARTE model for training.

        Returns the model that can be used for training.
        """
        # Model configuration
        model_config = dict()
        model_config["input_dim_x"] = self.gc_.data_kg["x_total"].size(1)
        model_config["input_dim_e"] = self.gc_.data_kg["edge_attr_total"].size(1)
        model_config["hidden_dim"] = self.gc_.data_kg["x_total"].size(1)
        model_config["ff_dim"] = self.gc_.data_kg["x_total"].size(1)
        model_config["num_heads"] = 12
        model_config["num_layers"] = self.num_layers
        
        # Set model architecture
        model = CARTE_Pretrain(**model_config)

        return model

    def _save_checkpoint(self, model, step, domain_name):
        """Save checkpoint at the desginated step."""

        # Directory setup



        # Directory setup
        result_save_dir_base = (
            f"{config_directory['checkpoints']}/domain_pretrain/{domain_name}"
        )
        if not os.path.exists(result_save_dir_base):
            os.makedirs(result_save_dir_base, exist_ok=True)
        marker = f"batch_size-{self.batch_size}_num_layers-{self.num_layers}_num_perturb-{self.num_perturb}_perturb_fraction-{self.perturb_fraction}"
        result_save_dir = f"{result_save_dir_base}/{marker}"
        if not os.path.exists(result_save_dir):
            os.makedirs(result_save_dir, exist_ok=True)
        ckpt_dir = result_save_dir + "/ckpt"
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        log_dir = result_save_dir + "/log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        # Save checkpoint and log
        ckpt = model.state_dict()
        torch.save(ckpt, f"{ckpt_dir}/ckpt_step{step}.pt")
        with open(f"{log_dir}/log_train.txt", "w") as output:
            for row in self.log_:
                output.write(str(row) + "\n")
        print(f"Step {step-1} | Training checkpoint saved")

