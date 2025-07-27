"""
AIO -- All Model in One
"""
import torch.nn as nn

from models.missingTask import *
# from trains.singleTask import *

__all__ = ['AMIO']



MODEL_MAP = {
    'hgatt_net': HQAENetwork,  # HGAttLayer,  # HGAtt Reconstructor
    'gqa_net': HQAENetwork,    # GQA Reconstructor
}

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        # simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        lastModel = MODEL_MAP[args.modelName]
        self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x, do_test, *args, **kwargs):
        return self.Model(text_x, audio_x, video_x, do_test, *args, **kwargs)
