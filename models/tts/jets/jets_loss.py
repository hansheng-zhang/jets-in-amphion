from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.vocoders.gan.discriminator.mpd import MultiScaleMultiPeriodDiscriminator

def get_segments(
    x: torch.Tensor,
    start_idxs: torch.Tensor,
    segment_size: int,
) -> torch.Tensor:
    """Get segments.

    Args:
        x (Tensor): Input tensor (B, C, T).
        start_idxs (Tensor): Start index tensor (B,).
        segment_size (int): Segment size.

    Returns:
        Tensor: Segmented tensor (B, C, segment_size).

    """
    b, c, t = x.size()
    segments = x.new_zeros(b, c, segment_size)
    for i, start_idx in enumerate(start_idxs):
        segments[i] = x[i, :, start_idx : start_idx + segment_size]
    return segments

class GeneratorAdversarialLoss(torch.nn.Module):
    """Generator adversarial loss module."""

    def __init__(self):
        super().__init__()

    def forward(self, outputs) -> torch.Tensor:
        adv_loss = 0.0
        for x in outputs:
            l = F.mse_loss(x, x.new_ones(x.size()))
            adv_loss += l

        return adv_loss
    
class FeatureMatchLoss(torch.nn.Module):
    """Feature matching loss module."""

    def __init__(self):
        super().__init__()

    def forward(self, feats_hat, feats) -> torch.Tensor:
        feat_match_loss = 0.0
        for i, (feats_hat_, feats_) in enumerate(zip(feats_hat, feats)):
            feat_match_loss_ = 0.0
            for j, (feat_hat_, feat_) in enumerate(zip(feats_hat_, feats_)):
                feat_match_loss_ += F.l1_loss(feat_hat_, feat_.detach())
            feat_match_loss_ /= j + 1
            feat_match_loss += feat_match_loss_
        feat_match_loss /= i + 1

        return feat_match_loss

class DurationPredictorLoss(torch.nn.Module):
    """Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, offset=1.0, reduction="mean"):
        """Initilize duration predictor loss module.

        Args:
            offset (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.

        """
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.offset = offset

    def forward(self, outputs, targets):
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)

        return loss

class VarianceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # define criterions
        reduction = "mean"
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)

    def forward(
        self,
        d_outs: torch.Tensor,
        ds: torch.Tensor,
        p_outs: torch.Tensor,
        ps: torch.Tensor,
        e_outs: torch.Tensor,
        es: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # apply mask to remove padded part
        duration_masks = make_non_pad_mask(ilens).to(ds.device)
        d_outs = d_outs.masked_select(duration_masks)
        ds = ds.masked_select(duration_masks)
        pitch_masks = make_non_pad_mask(ilens).unsqueeze(-1).to(ds.device)
        p_outs = p_outs.masked_select(pitch_masks)
        e_outs = e_outs.masked_select(pitch_masks)
        ps = ps.masked_select(pitch_masks)
        es = es.masked_select(pitch_masks)

        # calculate loss
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)

        return duration_loss, pitch_loss, energy_loss

class ForwardSumLoss(torch.nn.Module):
    """Forwardsum loss described at https://openreview.net/forum?id=0NQwnnwAORi"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        log_p_attn: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
        blank_prob: float = np.e**-1,
    ) -> torch.Tensor:
        B = log_p_attn.size(0)

        # a row must be added to the attention matrix to account for
        #    blank token of CTC loss
        # (B,T_feats,T_text+1)
        log_p_attn_pd = F.pad(log_p_attn, (1, 0, 0, 0, 0, 0), value=np.log(blank_prob))

        loss = 0
        for bidx in range(B):
            # construct target sequnece.
            # Every text token is mapped to a unique sequnece number.
            target_seq = torch.arange(1, ilens[bidx] + 1).unsqueeze(0)
            cur_log_p_attn_pd = log_p_attn_pd[
                bidx, : olens[bidx], : ilens[bidx] + 1
            ].unsqueeze(1)  # (T_feats,1,T_text+1)
            cur_log_p_attn_pd = F.log_softmax(cur_log_p_attn_pd, dim=-1)
            loss += F.ctc_loss(
                log_probs=cur_log_p_attn_pd,
                targets=target_seq,
                input_lengths=olens[bidx : bidx + 1],
                target_lengths=ilens[bidx : bidx + 1],
                zero_infinity=True,
            )
        loss = loss / B
        return loss

class MelSpectrogramLoss(torch.nn.Module):
    """Mel-spectrogram loss."""

    def __init__(
        self,
        fs: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: str = "hann",
        n_mels: int = 80,
        fmin: Optional[int] = 0,
        fmax: Optional[int] = None,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        """Initialize Mel-spectrogram loss.

        Args:
            fs (int): Sampling rate.
            n_fft (int): FFT points.
            hop_length (int): Hop length.
            win_length (Optional[int]): Window length.
            window (str): Window type.
            n_mels (int): Number of Mel basis.
            fmin (Optional[int]): Minimum frequency for Mel.
            fmax (Optional[int]): Maximum frequency for Mel.
            center (bool): Whether to use center window.
            normalized (bool): Whether to use normalized one.
            onesided (bool): Whether to use oneseded one.

        """
        super().__init__()
        
        self.fs=fs
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.win_length=win_length
        self.window=window
        self.n_mels=n_mels
        self.fmin = 0 if fmin is None else fmin
        self.fmax = fs / 2 if fmax is None else fmax
        self.center=center
        self.normalized=normalized
        self.onesided=onesided
    
    def logmel(self, feat):
        mel_options = dict(
            sr=self.fs,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            htk=self.htk,
        )
        melmat = librosa.filters.mel(**mel_options)
        mel_feat = torch.matmul(feat, melmat)
        mel_feat = torch.clamp(mel_feat, min=1e-10)
        logmel_feat = mel_feat.log10() 
        return logmel_feat
    
    def wav_to_mel(self, input):
        stft_kwargs = dict(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=self.center,
            window=self.window,
            normalized=self.normalized,
            onesided=self.onesided,
        )
        input_stft = torch.stft(input, **stft_kwargs)
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        input_feats = self.logmel(input_amp)
        return input_feats
    
    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        mel_hat = self.wav_to_mel(y_hat.squeeze(1))
        mel = self.wav_to_mel(y.squeeze(1))
        mel_loss = F.l1_loss(mel_hat, mel)

        return mel_loss





class GeneratorLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.mel_loss = MelSpectrogramLoss()
        self.generator_adv_loss = GeneratorAdversarialLoss()
        self.feat_match_loss = FeatureMatchLoss()
        self.var_loss = VarianceLoss()
        self.forwardsum_loss = ForwardSumLoss()

        self.lambda_adv = cfg.lambda_adv
        self.lambda_mel = cfg.lambda_mel
        self.lambda_feat_match = cfg.lambda_feat_match
        self.lambda_var = cfg.lambda_var
        self.lambda_align = cfg.lambda_align
    
    def forward(
        self,
        outputs_g,
        outputs_d,
        speech
    ):
        loss_g = {}
        
        (
        speech_hat_,
        bin_loss,
        log_p_attn,
        start_idxs,
        d_outs,
        ds,
        p_outs,
        ps,
        e_outs,
        es,
        text_lengths,
        feats_lengths
        ) = outputs_g
        
        (
        p,
        p_hat
        ) = outputs_d
        
        speech_ = get_segments(
                x=speech,
                start_idxs=start_idxs * self.generator.upsample_factor,
                segment_size=self.generator.segment_size * self.generator.upsample_factor,
        )
        
        mel_loss = self.mel_loss(speech_hat_, speech_)
        adv_loss = self.generator_adv_loss(p_hat)
        feat_match_loss = self.feat_match_loss(p_hat, p)
        dur_loss, pitch_loss, energy_loss = self.var_loss(
            d_outs, ds, p_outs, ps, e_outs, es, text_lengths
        )
        
        forwardsum_loss = self.forwardsum_loss(log_p_attn, text_lengths, feats_lengths)

        mel_loss = mel_loss * self.lambda_mel
        adv_loss = adv_loss * self.lambda_adv
        feat_match_loss = feat_match_loss * self.lambda_feat_match
        g_loss = mel_loss + adv_loss + feat_match_loss
        var_loss = (dur_loss + pitch_loss + energy_loss) * self.lambda_var
        align_loss = (forwardsum_loss + bin_loss) * self.lambda_align

        loss = g_loss + var_loss + align_loss

        loss_g["loss_gen_all"] = loss

        return loss_g
    
class DiscriminatorAdversarialLoss(torch.nn.Module):
    """Discriminator adversarial loss module."""

    def __init__(
        self,
        average_by_discriminators: bool = True,
        loss_type: str = "mse",
    ):
        """Initialize DiscriminatorAversarialLoss module.

        Args:
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            loss_type (str): Loss type, "mse" or "hinge".

        """
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.fake_criterion = self._mse_fake_loss
            self.real_criterion = self._mse_real_loss
        else:
            self.fake_criterion = self._hinge_fake_loss
            self.real_criterion = self._hinge_real_loss

    def forward(
        self,
        outputs_hat: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        outputs: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcualate discriminator adversarial loss.

        Args:
            outputs_hat (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs calculated from generator.
            outputs (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs calculated from groundtruth.

        Returns:
            Tensor: Discriminator real loss value.
            Tensor: Discriminator fake loss value.

        """
        if isinstance(outputs, (tuple, list)):
            real_loss = 0.0
            fake_loss = 0.0
            for i, (outputs_hat_, outputs_) in enumerate(zip(outputs_hat, outputs)):
                if isinstance(outputs_hat_, (tuple, list)):
                    # NOTE(kan-bayashi): case including feature maps
                    outputs_hat_ = outputs_hat_[-1]
                    outputs_ = outputs_[-1]
                real_loss += self.real_criterion(outputs_)
                fake_loss += self.fake_criterion(outputs_hat_)
            if self.average_by_discriminators:
                fake_loss /= i + 1
                real_loss /= i + 1
        else:
            real_loss = self.real_criterion(outputs)
            fake_loss = self.fake_criterion(outputs_hat)

        return real_loss, fake_loss

    def _mse_real_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, x.new_ones(x.size()))

    def _mse_fake_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, x.new_zeros(x.size()))

    def _hinge_real_loss(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.min(x - 1, x.new_zeros(x.size())))

    def _hinge_fake_loss(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.min(-x - 1, x.new_zeros(x.size())))

class DiscriminatorLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(DiscriminatorLoss, self).__init__()
        self.cfg = cfg
        self.discriminator = MultiScaleMultiPeriodDiscriminator()
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss()

    def forward(self, speech_real, speech_generated):
        loss_d = {}

        disc_real_outputs = self.discriminator(speech_real)
        disc_generated_outputs = self.discriminator(speech_generated)
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            dr = dr.float()
            dg = dg.float()
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg**2)
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        loss_d["loss_disc_all"] = loss

        # jets original
        # real_loss, fake_loss = self.discriminator_adv_loss(disc_generated_outputs, disc_real_outputs)
        # loss_d["loss_disc_all"] = real_loss + fake_loss

        # stats = dict(
        #     discriminator_loss=loss.item(),
        #     discriminator_real_loss=real_loss.item(),
        #     discriminator_fake_loss=fake_loss.item(),
        # )
        # loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)


        return loss_d