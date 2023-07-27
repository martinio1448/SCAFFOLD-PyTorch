import pickle
import random

import torch
from rich.progress import track
from tqdm import tqdm

from base import ServerBase
from client.scaffold import SCAFFOLDClient
from config.util import clone_parameters, get_args


class SCAFFOLDServer(ServerBase):
    def __init__(self):
        super(SCAFFOLDServer, self).__init__(get_args(), "SCAFFOLD")

        self.trainer = SCAFFOLDClient(
            backbone=self.backbone(self.args.dataset),
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            logger=self.logger,
            gpu=self.args.gpu,
        )
        self.c_global = [
            torch.zeros_like(param).to(self.device)
            for param in self.backbone(self.args.dataset).parameters()
        ]
        self.global_lr = 1.0
        self.training_acc = [[] for _ in range(self.global_epochs)]

    def train(self):
        self.logger.log("=" * 30, "TRAINING", "=" * 30, style="bold green")
        progress_bar = (
            track(
                range(self.global_epochs),
                "[bold green]Training...",
                console=self.logger,
            )
            if not self.args.log
            else tqdm(range(self.global_epochs), "Training...")
        )
        for E in progress_bar:

            if E % self.args.verbose_gap == 0:
                self.logger.log("=" * 30, f"ROUND: {E}", "=" * 30)

            selected_clients = random.sample(
                self.client_id_indices, self.args.client_num_per_round
            )
            res_cache = []
            for client_id in selected_clients:
                client_local_params = clone_parameters(self.global_params_dict)
                res, stats = self.trainer.train(
                    client_id=client_id,
                    model_params=client_local_params,
                    c_global=self.c_global,
                    verbose=(E % self.args.verbose_gap) == 0,
                    round_number=E
                )
                res_cache.append(res)

                self.num_correct[E].append(stats["correct"])
                self.num_samples[E].append(stats["size"])
            self.aggregate(res_cache, E)

            if E % self.args.save_period == 0 and self.args.save_period > 0:
                torch.save(
                    self.global_params_dict,
                    self.temp_dir / "global_model.pt",
                )
                with open(self.temp_dir / "epoch.pkl", "wb") as f:
                    pickle.dump(E, f)

            torch.cuda.empty_cache()

    def aggregate(self, res_cache, E: int):
        y_delta_cache = list(zip(*res_cache))[0]
        c_delta_cache = list(zip(*res_cache))[1]
        trainable_parameter = filter(
            lambda param: param.requires_grad, self.global_params_dict.values()
        )

        # update global model
        avg_weight = torch.tensor(
            [
                1 / self.args.client_num_per_round
                for _ in range(self.args.client_num_per_round)
            ],
            device=self.device,
        )
        for param, y_del in zip(trainable_parameter, zip(*y_delta_cache)):
            x_del = torch.sum(avg_weight * torch.stack(y_del, dim=-1), dim=-1)
            param.data += self.global_lr * x_del

        # update global control
        for c_g, c_del in zip(self.c_global, zip(*c_delta_cache)):
            c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1)
            c_g.data += (
                self.args.client_num_per_round / len(self.client_id_indices)
            ) * c_del

        torch.save(self.c_global, f"data/control_variates/control_variates_global_r{E}.pt")


if __name__ == "__main__":
    server = SCAFFOLDServer()
    server.run()
