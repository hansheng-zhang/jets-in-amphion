# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from tqdm import tqdm
from models.tts.base import TTSTrainer
from models.tts.fastspeech2.fs2 import FastSpeech2
from models.tts.fastspeech2.jets_loss import GeneratorLoss, DiscriminatorLoss
from models.tts.fastspeech2.fs2_dataset import FS2Dataset, FS2Collator
from optimizer.optimizers import NoamLR
from models.vocoders.gan.discriminator.mpd import MultiScaleMultiPeriodDiscriminator


class FastSpeech2Trainer(TTSTrainer):
    def __init__(self, args, cfg):
        TTSTrainer.__init__(self, args, cfg)
        self.cfg = cfg

    def _build_dataset(self):
        return FS2Dataset, FS2Collator

    def __build_scheduler(self):
        return NoamLR(self.optimizer, **self.cfg.train.lr_scheduler)

    def _write_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar("train/" + key, value, self.step)
        lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        self.sw.add_scalar("learning_rate", lr, self.step)

    def _write_valid_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar("val/" + key, value, self.step)

    def _build_criterion(self):
        criterion = {
            "generator": GeneratorLoss(self.cfg),
            "discriminator": DiscriminatorLoss(self.cfg),
        }
        return criterion

    def get_state_dict(self):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        return state_dict

    def _build_optimizer(self):
        optimizer_g = torch.optim.AdamW(
            self.model["generator"].parameters(),
            self.cfg.train.learning_rate,
            betas=self.cfg.train.AdamW.betas,
            eps=self.cfg.train.AdamW.eps,
        )
        optimizer_d = torch.optim.AdamW(
            self.model["discriminator"].parameters(),
            self.cfg.train.learning_rate,
            betas=self.cfg.train.AdamW.betas,
            eps=self.cfg.train.AdamW.eps,
        )
        optimizer = {"optimizer_g": optimizer_g, "optimizer_d": optimizer_d}

        return optimizer

    def _build_scheduler(self):
        scheduler = NoamLR(self.optimizer, **self.cfg.train.lr_scheduler)
        return scheduler

    def _build_model(self):
        net_g = FastSpeech2(self.cfg)
        net_d = MultiScaleMultiPeriodDiscriminator()
        self.model = {"generator": net_g, "discriminator": net_d}
        return self.model

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        self.model.train()
        epoch_sum_loss: float = 0.0
        epoch_step: int = 0
        epoch_losses: dict = {}
        for batch in tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.epoch}",
            unit="batch",
            colour="GREEN",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.04,
            disable=not self.accelerator.is_main_process,
        ):
            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                loss, train_losses = self._train_step(batch)
                self.accelerator.backward(loss)
                grad_clip_thresh = self.cfg.train.grad_clip_thresh
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_thresh)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            self.batch_count += 1

            # Update info for each step
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss += loss
                for key, value in train_losses.items():
                    if key not in epoch_losses.keys():
                        epoch_losses[key] = value
                    else:
                        epoch_losses[key] += value

                self.accelerator.log(
                    {
                        "Step/Train Loss": loss,
                        "Step/Learning Rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=self.step,
                )
                self.step += 1
                epoch_step += 1

        self.accelerator.wait_for_everyone()

        epoch_sum_loss = (
            epoch_sum_loss
            / len(self.train_dataloader)
            * self.cfg.train.gradient_accumulation_step
        )

        for key in epoch_losses.keys():
            epoch_losses[key] = (
                epoch_losses[key]
                / len(self.train_dataloader)
                * self.cfg.train.gradient_accumulation_step
            )
        return epoch_sum_loss, epoch_losses

    def _train_step(self, batch):
        train_losses = {}
        total_loss = 0
        training_stats = {}

        # batch["linear"] = batch["linear"].transpose(2, 1)  # [b, d, t]
        # batch["mel"] = batch["mel"].transpose(2, 1)  # [b, d, t]
        # batch["audio"] = batch["audio"].unsqueeze(1)  # [b, d, t]

        # Train Discriminator
        # Generator output
        outputs_g = self.model["generator"](batch)
        speech_hat_, *_ = outputs_g

        # Discriminator output
        p = self.model["discriminator"](batch["audio"])
        p_hat = self.model["discriminator"](speech_hat_.detach())
        ##  Discriminator loss
        loss_d = self.criterion["discriminator"](p, p_hat)
        train_losses.update(loss_d)

        # BP and Grad Updated
        self.optimizer["optimizer_d"].zero_grad()
        self.accelerator.backward(loss_d["loss_disc_all"])
        self.optimizer["optimizer_d"].step()

        ## Train Generator
        p = self.model["discriminator"](batch["audio"].detach())
        p_hat = self.model["discriminator"](speech_hat_)
        outputs_d = (p, p_hat)
        loss_g = self.criterion["generator"](outputs_g, outputs_d, batch["audio"])
        train_losses.update(loss_g)

        # BP and Grad Updated
        self.optimizer["optimizer_g"].zero_grad()
        self.accelerator.backward(loss_g["loss_gen_all"])
        self.optimizer["optimizer_g"].step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        total_loss = loss_g["loss_gen_all"] + loss_d["loss_disc_all"]

        return (
            total_loss.item(),
            train_losses,
            training_stats,
        )

    @torch.no_grad()
    def _valid_step(self, data):
        valid_loss = {}
        total_valid_loss = 0
        valid_stats = {}

        preds = self.model(data)

        valid_losses = self.criterion(data, preds)

        total_valid_loss = valid_losses["loss"]
        for key, value in valid_losses.items():
            valid_losses[key] = value.item()

        return total_valid_loss, valid_losses, valid_stats
