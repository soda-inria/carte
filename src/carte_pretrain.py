"""
CARTE pretrain with knowledge graphs (YAGO).

"""

import os
import torch
import datetime

from time import time

from src.carte_yago_graphlet_construction import Graphlet


## Target construction for the pretraining
def _create_target_node(data):
    graph_idx = data.g_idx
    pos_mask = (
        graph_idx.repeat(graph_idx.size(0), 1)
        - graph_idx.repeat(graph_idx.size(0), 1).t()
    )

    target = pos_mask.clone()
    target[pos_mask == 0] = 1
    target[pos_mask != 0] = 0
    target = target.type("torch.cuda.FloatTensor")

    return target


class CARTE_KGPretrain:
    def __init__(
        self,
        num_layers: int = 0,
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        num_hop: int = 1,
        num_perturb: int = 1,
        perturb_fraction: float = 0.5,
        num_steps: int = 100000,
        save_every: int = 1000,
        device: str = "cuda:0",
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_perturb = num_perturb
        self.perturb_fraction = perturb_fraction
        self.num_steps = num_steps
        self.save_every = save_every
        self.device = device

        self.graphlet

    def fit(self, X, domain_name):
        return None

    def _run_step(self, step, idx):
        return None        

    def _save_checkpoint(self, step):
        return None        




## Trainer class
class Trainer:
    def __init__(
        self,
        exp_setting: dict
    ) -> None:
        self.__dict__ = exp_setting
        self.model = self.model.to(self.device)
        self.log = []

    def _run_step(self, step, idx):
        start_time = time()
        self.optimizer.zero_grad()
        data = self.graphlet.make_batch(idx, **self.graphlet_setting)
        data = data.to(self.device)
        output_node = self.model(data)

        # loss on nodes
        target_node = _create_target_node(data)
        loss = self.criterion_node(output_node, target_node)

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        end_time = time()
        duration = round(end_time - start_time, 4)

        loss = round(loss.detach().item(), 4)

        self.log.append(f"Step {step} | Loss(node): {loss} | Duration: {duration}")

        print(
            f"[GPU{self.device}] | Step {step} | Loss(node): {loss} | Duration: {duration}"
        )

        del (
            loss,
            output_node,
            target_node,
            data,
        )

    def _save_checkpoint(self, step):
        ckp = self.model.state_dict()
        PATH = self.save_dir + f"/ckpt_step{step}.pt"
        torch.save(ckp, PATH)
        PATH_LOG = self.save_dir + f"/log_train.txt"
        with open(PATH_LOG, "w") as output:
            for row in self.log:
                output.write(str(row) + "\n")
        print(f"Step {step-1} | Training checkpoint saved at {self.save_dir}")

    def train(self):
        self.model.train()
        step = 0
        idx = self.idx_extract.sample(n_batch=self.n_batch)
        while step < self.n_steps:
            self._run_step(step, idx)
            step += 1
            if step % self.save_every == 0:
                self._save_checkpoint(step)


def load_train_objs(
    data_name: str,
    gpu_device: int,
    num_hops: int,
    max_nodes: int,
    per_keep: float,
    n_perturb: int,
    n_batch: int,
    n_steps: int,
    save_every: int,
    num_rel=None,
):
    # create dictionary that sets experiment settings
    exp_setting = dict()

    # load data
    main_data = Load_Yago(data_name=data_name, numerical=True)

    # gpu device settings
    device = torch.device(f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu")
    exp_setting["device"] = device

    # graphlet settings
    graphlet = Graphlet(main_data, num_hops=num_hops, max_nodes=max_nodes)
    exp_setting["graphlet"] = graphlet
    exp_setting["graphlet_setting"] = dict(
        {
            "aggregate": True,
            "per_keep": per_keep,
            "n_perturb": n_perturb,
        }
    )

    # model settings
    model = YATE_Pretrain(
        input_dim_x=300,
        input_dim_e=300,
        hidden_dim=300,
        num_layers=0,
        ff_dim=300,
        num_heads=12,
    )
    num_layers = 0

    exp_setting["model"] = model

    # training settings
    exp_setting["n_batch"] = n_batch
    exp_setting["n_steps"] = n_steps
    exp_setting["save_every"] = save_every

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    exp_setting["optimizer"] = optimizer

    criterion_node = Infonce_loss(tau=1.0)
    loss_abv = "CL"

    # Other losses: torch.nn. BCEWithLogitsLoss BCELoss L1Loss / Infonce_loss Max_sim_loss
    exp_setting["criterion_node"] = criterion_node

    # set train for batch
    idx_extract = Index_extractor(main_data, num_rel=num_rel, per=0.9)
    exp_setting["idx_extract"] = idx_extract

    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        T_0=n_steps,
        T_mult=1,
        eta_max=1e-4,
        T_up=10000,
        gamma=1,
    )
    exp_setting["scheduler"] = scheduler

    # directory for saving ckpt
    now = datetime.datetime.now()
    save_dir = (
        os.getcwd()
        + "/data/saved_model/"
        + data_name
        + "_"
        + now.strftime(
            f"%d%m_NB{n_batch}_NH{num_hops}_NL{num_layers}_NP{n_perturb}_MN{max_nodes}_{loss_abv}"
        )
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    exp_setting["save_dir"] = save_dir

    return exp_setting