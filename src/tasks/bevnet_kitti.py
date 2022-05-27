import numpy as np
import torch
import torch.nn.functional as F
from pytorch_helper.settings.spaces import Spaces
from pytorch_helper.task import Batch
from pytorch_helper.task import Task
from pytorch_helper.utils.log import get_logger

from settings.options import BEVNetTrainRoutine
from tasks.bevnet import BEVNetTask
from tasks.helper import assemble_3in1_image

__all__ = ['BEVNetKittiTask']

logger = get_logger(__name__)


@Spaces.register(Spaces.NAME.TASK, ['BEVNetKittiTask'])
class BEVNetKittiTask(BEVNetTask):

    def get_3in1_images(self, tag, result, index=0):
        gt_batch = {k: v.data.cpu() for k, v in result.gt.items()}
        pred_batch = {
            k: v.data.cpu() for k, v in result.pred.items()
            if v is not None
        }

        images = dict()
        for k, v in pred_batch.items():
            if v is None or k not in ['head_map', 'feet_map', 'bev_map']:
                continue
            gt_map = gt_batch[k][index]
            pred_map = v[index]
            loss = self.loss_fn.map_loss_fn(
                pred_batch[k][index], gt_batch[k][index]
            ).item()
            pd_height = pred_batch['camera_height'][index].item()
            pd_angle = pred_batch['camera_angle'][index].item() / np.pi * 180
            gt_height = gt_batch['camera_height'][index].item()
            gt_angle = gt_batch['camera_angle'][index].item() / np.pi * 180
            titles = [
                # input
                f'input, map loss={loss:.2e}\n',
                # gt
                f'g.t. camera height={gt_height:.2e}m\n'
                f'camera angle={gt_angle:.2e}',
                # pred
                f'p.d. camera height={pd_height:.2e}m\n'
                f'camera angle={pd_angle:.2e}'
            ]
            images[f'{tag}/{k}'] = assemble_3in1_image(
                gt_batch['image'][index].cpu(), gt_map, pred_map, titles
            )
        return images
