import pickle
import random
import datetime

import os
import torch
from rich.progress import track, Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from tqdm import tqdm
from pathlib import Path

from base import ServerBase
from client.scaffold import SCAFFOLDClient
from config.util import clone_parameters, get_args


class SCAFFOLDServer(ServerBase):
    def __init__(self):
        super(SCAFFOLDServer, self).__init__(get_args(), "SCAFFOLD")

        self.trainer = SCAFFOLDClient(
            backbone=self.backbone(self.args.dataset, self.colorized),
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            logger=self.logger,
            gpu=self.args.gpu,
            output_dir=self.args.output_dir,
            num_client=self.client_num_in_total,
            augment=self.args.augment,
            writer=self.writer
        )
      
        self.c_global = [
            torch.zeros_like(param).to(self.device)
            for param in self.backbone(self.args.dataset, self.colorized).parameters()
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
            else tqdm(range(self.global_epochs), "Training...", position=1, leave=True)
        )


        with Progress(console=self.logger) as pg:
            global_epoch_progress = pg.add_task("[bold green]Global Training...")

            handler = torch.profiler.tensorboard_trace_handler(f"{self.args.output_dir}/profiling")

            def self_trace(*args, **kwargs):
                print("Trace_Ready")
                handler(*args, **kwargs)

            stats_cache = []
            print(f"outputting profiler to: {self.args.output_dir}/profiling")
            with torch.profiler.profile(
                on_trace_ready=self_trace,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                for E in pg.track(range(self.global_epochs), task_id=global_epoch_progress):

                    if E % self.args.verbose_gap == 0:
                        self.logger.log("=" * 30, f"ROUND: {E}", "=" * 30)

                    selected_clients = random.sample(
                        self.client_id_indices, self.args.client_num_per_round
                    )

                    (loss, correct) = self.trainer.evaluate(True, self.trainer.global_dataset["val"])

                    res_cache = []
                    round_stats_cache = [None] * len(self.client_id_indices)
                    for client_id in selected_clients:
                        client_local_params = clone_parameters(self.global_params_dict)
                        res, stats = self.trainer.train(
                            client_id=client_id,
                            progress_tracker=pg,
                            model_params=client_local_params,
                            c_global=self.c_global,
                            verbose=(E % self.args.verbose_gap) == 0,
                            round_number=E,
                            prev_acc=(loss, correct),
                            profiler=prof
                        )
                        res_cache.append(res)

                        self.num_correct[E].append(stats["correct"])
                        self.num_samples[E].append(stats["size"])
                        round_stats_cache[client_id] = stats
                    self.aggregate(res_cache, E)
                    stats_cache.append(round_stats_cache)

                    if E % self.args.save_period == 0 and self.args.save_period > 0:
                        torch.save(
                            self.global_params_dict,
                            self.temp_dir / f"global_model_{E}.pt",
                        )
                        with open(self.temp_dir / "epoch.pkl", "wb") as f:
                            pickle.dump(E, f)

                    with open(f"{self.args.output_dir}/stats_{E}.pkl", "wb") as f:
                        pickle.dump(stats_cache, f)
                    # torch.cuda.empty_cache()

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

        torch.save(self.c_global,  f"{self.args.output_dir}/control_variates/control_variates_global_r{E}.pt")


if __name__ == "__main__":
    server = SCAFFOLDServer()
    server.run()
