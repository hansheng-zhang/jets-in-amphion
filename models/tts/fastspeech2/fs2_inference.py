# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from tqdm import tqdm
from collections import OrderedDict

from models.tts.base.tts_inferece import TTSInference
from models.tts.fastspeech2.fs2_dataset import FS2TestDataset, FS2TestCollator
from utils.util import load_config
from utils.io import save_audio
from models.tts.fastspeech2.fs2 import FastSpeech2
from models.vocoders.vocoder_inference import synthesis
from pathlib import Path
from processors.phone_extractor import phoneExtractor
from text.text_token_collation import phoneIDCollation
import numpy as np
import json
import time


class FastSpeech2Inference(TTSInference):
    def __init__(self, args, cfg):
        TTSInference.__init__(self, args, cfg)
        self.args = args
        self.cfg = cfg
        self.infer_type = args.mode

    def _build_model(self):
        self.model = FastSpeech2(self.cfg)
        return self.model

    def _build_test_dataset(self):
        return FS2TestDataset, FS2TestCollator

    def inference_for_batches(
        self
    ):
        ###### Construct test_batch ######
        n_batch = len(self.test_dataloader)
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(
            "Model eval time: {}, batch_size = {}, n_batch = {}".format(
                now, self.test_batch_size, n_batch
            )
        )
        self.model.eval()

        ###### Inference for each batch ######
        pred_res = []
        with torch.no_grad():
            for i, batch_data in enumerate(
                self.test_dataloader if n_batch == 1 else tqdm(self.test_dataloader)
            ):
                outputs = self.model.inference(batch_data)

                audios, masks = outputs

                for idx in range(audios.size(0)):
                    audio = audios[idx, 0, :].data.cpu().float()
                    # mask = masks[idx, :, :]
                    # audio_length = (
                    #     mask.sum([0, 1]).long() * self.cfg.preprocess.hop_size
                    # )
                    # audio_length = audio_length.cpu().numpy()
                    # audio = audio[:audio_length]
                    pred_res.append(audio)

        return pred_res
