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
from torch.optim.lr_scheduler import ExponentialLR
from models.vocoders.gan.discriminator.mpd import MultiScaleMultiPeriodDiscriminator


class FastSpeech2Trainer(TTSTrainer):
    def __init__(self, args, cfg):
        TTSTrainer.__init__(self, args, cfg)
        self.cfg = cfg

    def _build_dataset(self):
        return FS2Dataset, FS2Collator

    def __build_scheduler(self):
        return NoamLR(self.optimizer, **self.cfg.train.lr_scheduler)

    def _write_summary(
        self,
        losses,
        stats,
        images={},
        audios={},
        audio_sampling_rate=24000,
        tag="train",
    ):
        for key, value in losses.items():
            self.sw.add_scalar(tag + "/" + key, value, self.step)
        self.sw.add_scalar(
            "learning_rate",
            self.optimizer["optimizer_g"].param_groups[0]["lr"],
            self.step,
        )

        if len(images) != 0:
            for key, value in images.items():
                self.sw.add_image(key, value, self.global_step, batchformats="HWC")
        if len(audios) != 0:
            for key, value in audios.items():
                self.sw.add_audio(key, value, self.global_step, audio_sampling_rate)

        for key, value in losses.items():
            self.sw.add_scalar("train/" + key, value, self.step)
        lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        self.sw.add_scalar("learning_rate", lr, self.step)

    def _write_valid_summary(
        self, losses, stats, images={}, audios={}, audio_sampling_rate=24000, tag="val"
    ):
        for key, value in losses.items():
            self.sw.add_scalar(tag + "/" + key, value, self.step)

        if len(images) != 0:
            for key, value in images.items():
                self.sw.add_image(key, value, self.global_step, batchformats="HWC")
        if len(audios) != 0:
            for key, value in audios.items():
                self.sw.add_audio(key, value, self.global_step, audio_sampling_rate)

    def _build_criterion(self):
        criterion = {
            "generator": GeneratorLoss(self.cfg),
            "discriminator": DiscriminatorLoss(self.cfg),
        }
        return criterion

    def get_state_dict(self):
        state_dict = {
            "generator": self.model["generator"].state_dict(),
            "discriminator": self.model["discriminator"].state_dict(),
            "optimizer_g": self.optimizer["optimizer_g"].state_dict(),
            "optimizer_d": self.optimizer["optimizer_d"].state_dict(),
            "scheduler_g": self.scheduler["scheduler_g"].state_dict(),
            "scheduler_d": self.scheduler["scheduler_d"].state_dict(),
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
        scheduler_g = ExponentialLR(
            self.optimizer["optimizer_g"],
            gamma=self.cfg.train.lr_decay,
            last_epoch=self.epoch - 1,
        )
        scheduler_d = ExponentialLR(
            self.optimizer["optimizer_d"],
            gamma=self.cfg.train.lr_decay,
            last_epoch=self.epoch - 1,
        )

        scheduler = {"scheduler_g": scheduler_g, "scheduler_d": scheduler_d}
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
        self.model["generator"].train()
        self.model["discriminator"].train()
        epoch_sum_loss: float = 0.0
        epoch_losses: dict = {}
        epoch_step: int = 0
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
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, training_stats = self._train_step(batch)
            self.batch_count += 1

            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss += total_loss
                for key, value in train_losses.items():
                    if key not in epoch_losses.keys():
                        epoch_losses[key] = value
                    else:
                        epoch_losses[key] += value

                self.accelerator.log(
                    {
                        "Step/Generator Loss": train_losses["loss_gen_all"],
                        "Step/Discriminator Loss": train_losses["loss_disc_all"],
                        "Step/Generator Learning Rate": self.optimizer[
                            "optimizer_d"
                        ].param_groups[0]["lr"],
                        "Step/Discriminator Learning Rate": self.optimizer[
                            "optimizer_g"
                        ].param_groups[0]["lr"],
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
        speech = batch["audio"].unsqueeze(1)
        p = self.model["discriminator"](speech)
        p_hat = self.model["discriminator"](speech_hat_.detach())
        ##  Discriminator loss
        loss_d = self.criterion["discriminator"](p, p_hat)
        train_losses.update(loss_d)

        # BP and Grad Updated
        self.optimizer["optimizer_d"].zero_grad()
        self.accelerator.backward(loss_d["loss_disc_all"])
        self.optimizer["optimizer_d"].step()

        ## Train Generator
        p = self.model["discriminator"](speech.detach())
        p_hat = self.model["discriminator"](speech_hat_)
        outputs_d = (p, p_hat)
        loss_g = self.criterion["generator"](outputs_g, outputs_d, speech)
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
