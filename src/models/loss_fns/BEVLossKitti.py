import torch.nn.modules as nn

from .magnified_loss import MagnifiedMSELoss
from .pose_loss import PoseLoss

from pytorch_helper.settings.spaces import Spaces


@Spaces.register(Spaces.NAME.LOSS_FN, 'BEVLossKitti')
class BEVLossKitti(nn.Module):
    def __init__(self, magnitude_scale, loss_weights):
        super(BEVLoss, self).__init__()
        self.magnitude_scale = magnitude_scale
        self.loss_weights = loss_weights

        self.map_loss_fn = MagnifiedMSELoss(magnitude_scale)

    def forward(self, pred_dict, target_dict):
        loss_dict = dict()
        for k in ['feet_map', 'head_map', 'bev_map']:
            if k in pred_dict and pred_dict[k] is not None:
                loss_dict[k] = self.map_loss_fn(pred_dict[k], target_dict[k])

        # sum of loss
        loss_all = 0
        for k, w in self.loss_weights.items():
            if k in loss_dict:
                loss_all += loss_dict[k] * w
        loss_dict['all'] = loss_all
        return loss_dict