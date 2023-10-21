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
from typing import List
import numpy as np


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
      
        temp_backbone = self.backbone(self.args.dataset, self.colorized)
        global_trainable_parameters = filter(
                lambda p: p.requires_grad, temp_backbone.parameters()
            )
        
        self.c_global = [
            torch.zeros_like(param).to(self.device)
            for param in global_trainable_parameters
        ]
        self.global_lr = self.args.global_lr
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

            # handler = torch.profiler.tensorboard_trace_handler(f"{self.args.output_dir}/profiling")

            # def self_trace(*args, **kwargs):
            #     print("Trace_Ready")
            #     handler(*args, **kwargs)

            stats_cache = []
            # print(f"outputting profiler to: {self.args.output_dir}/profiling")
            # with torch.profiler.profile(
            #     activities=[torch.profiler.ProfilerActivity.CPU],
            #     schedule=torch.profiler.schedule(
            #         wait=1,
            #         warmup=1,
            #         active=8,
            #         repeat=5),
            #     on_trace_ready=self_trace,
            #     record_shapes=False,
            #     profile_memory=True,
            #     with_stack=True,
                

            # ) as prof:
            
            for E in pg.track(range(self.global_epochs), task_id=global_epoch_progress):

                if E % self.args.verbose_gap == 0:
                    self.logger.log("=" * 30, f"ROUND: {E}", "=" * 30)

                selected_clients = random.sample(
                    self.client_id_indices, self.args.client_num_per_round
                )

                selected_clients.sort()

                self.trainer.set_parameters(clone_parameters(self.global_params_dict))
                (loss, correct) = self.trainer.evaluate(dataset=self.trainer.global_dataset["test"], epoch=E, output_tag="global_val", transforms=self.trainer.get_test_transforms(E))

                res_cache = []
                cv_cache = []
                round_stats_cache = [None] * len(self.client_id_indices)
                
                client_task = pg.add_task("[Green]Training clients")
                for client_id in pg.track(selected_clients, task_id=client_task):
                    client_local_params = clone_parameters(self.global_params_dict)
                    # self.writer.open()
                    res, stats, c_plus = self.trainer.train(
                        client_id=client_id,
                        progress_tracker=pg,
                        model_params=client_local_params,
                        c_global=self.c_global,
                        verbose=(E % self.args.verbose_gap) == 0,
                        round_number=E,
                        prev_acc=(loss, correct),
                        profiler=None
                    )
                    res_cache.append(res)
                    cv_cache.append(c_plus)
                    self.num_correct[E].append(stats["correct"])
                    self.num_samples[E].append(stats["size"])
                    round_stats_cache[client_id] = stats
                    self.writer.close()

                self.writer.add_scalars(f"global_eval", {"accuracy": correct/len(self.trainer.global_dataset["test"])}, global_step=E)
                self.aggregate(res_cache, E)
                stats_cache.append(round_stats_cache)
                self.log_group_stats(selected_clients, E, cv_cache)
                del cv_cache

                if E % self.args.save_period == 0 and self.args.save_period > 0:
                    torch.save(
                        self.global_params_dict,
                        self.temp_dir / f"global_model_{E}.pt",
                    )
                    with open(self.temp_dir / "epoch.pkl", "wb") as f:
                        pickle.dump(E, f)

                with open(f"{self.args.output_dir}/stats_{E}.pkl", "wb") as f:
                    pickle.dump(stats_cache, f)
                
                pg.update(client_task, visible=False)
                # torch.cuda.empty_cache()
            
            self.logger.print("Finished Training!")


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


    def log_group_stats(self, client_ids: List[int], global_epoch: int, variates):
        num_clients = len(client_ids)
        group_indices = np.ma.arange(0,num_clients).reshape((num_clients//2, 2))
        layer_variates = [np.asarray(layer) for layer in zip(*variates)]

        group_means = []
        generic_group_difs = []
        specific_group_difs = []
        for group_num, indices in enumerate(group_indices):
            layer_means = []
            generic_layer_difs = []
            specific_layer_difs = []
            # print(f"Going for group {group_num}")
            for index, layer_lst in enumerate(layer_variates):
                layer = np.asarray(layer_lst)
                inner_group_means = layer[group_indices].mean(axis=1)

                sumrange = tuple(range(1,inner_group_means.ndim))
                dif_to_other_groups = np.sqrt(((inner_group_means-inner_group_means[group_num])**2).sum(axis=sumrange))
                generic_layer_difs.append(dif_to_other_groups)
                # print(dif_to_other_groups.shape, dif_to_other_groups, inner_group_means.shape)
                # print(layer.shape)
                inner_mean = inner_group_means[group_num]
                # inner_mean = layer[indices].mean(axis=0)
                # print(layer[indices].shape)
                inner_dif = np.sqrt(((layer[indices][0]-layer[indices][1])**2).sum())
                group_indices[group_num] = np.ma.masked
                # print(layer[indices].shape, layer[group_indices.flatten().compressed()].shape, group_indices.flatten().compressed())
                outer_mean = layer[group_indices.flatten().compressed()].mean(axis=0)
                # print(inner_group_means.shape, layer[indices].shape, inner_mean.shape)
                inter_difs = np.sqrt(((inner_mean-outer_mean)**2).sum())
                group_indices.mask[group_num] = False
                # print(inner_mean.shape, outer_mean.shape)
                # layer_means.append(np.stack((inner_mean, outer_mean)))
                specific_layer_difs.append(np.asarray((inner_dif, inter_difs)))
            group_means.append(layer_means)
            generic_group_difs.append(generic_layer_difs)
            specific_group_difs.append(specific_layer_difs)

        specific_stat_mean = np.asarray(specific_group_difs).mean(axis=1)
        generic_stat_map = np.asarray(generic_group_difs)
        generic_stat_mean = generic_stat_map.mean(axis=1)

        for group_id, group_stats in enumerate(specific_stat_mean):
            scalar_dict = {
                "inner_cv_dif" : group_stats[0],
                "inter_cv_dif" : group_stats[1],
            }
            self.writer.add_scalars(f"group_{group_id}", scalar_dict, global_epoch)

        torch.save(generic_stat_map, f"{self.args.output_dir}/cv_dif_r{global_epoch}.pt")
        del variates, generic_group_difs, group_means, specific_stat_mean

        for group_id, means in enumerate(generic_stat_mean):
            scalar_dict = {f"group_{index}": stat for index, stat in enumerate(means)}

            self.writer.add_scalars(f"group_{group_id}_generic", scalar_dict, global_epoch)

if __name__ == "__main__":
    server = SCAFFOLDServer()
    server.run()