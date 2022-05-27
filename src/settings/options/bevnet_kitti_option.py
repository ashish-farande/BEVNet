from dataclasses import dataclass
from typing import Optional
from typing import Tuple

from pytorch_helper.settings.options import ModelOption
from pytorch_helper.settings.options import OptionBase
from pytorch_helper.settings.options import TaskMode
from pytorch_helper.settings.options import TaskOption
from pytorch_helper.settings.options import TrainRoutine
from pytorch_helper.settings.options import TrainSettingOption
from pytorch_helper.settings.options.descriptors import AutoConvertDescriptor
from pytorch_helper.settings.spaces import Spaces
from pytorch_helper.utils.io import load_pth
from pytorch_helper.utils.log import get_logger
from .test_metric_option import TestMetricOption
from settings.options.bevnet_option import BEVNetTrainSettingOption, BEVNetOption

__all__ = ['BEVNetKittiTaskOption']



@Spaces.register(Spaces.NAME.TASK_OPTION, 'BEVNetKittiTask')
class BEVNetKittiTaskOption(TaskOption):
    model = AutoConvertDescriptor(BEVNetOption.from_dict)
    train_setting = AutoConvertDescriptor(BEVNetTrainSettingOption.from_dict)

    def __post_init__(self, mode, is_distributed):
        mode = TaskMode(mode)
        if mode != TaskMode.TRAIN:
            assert self.test_option, 'Please specify a test option file'
            self.test_option = self.load_option(
                self.test_option, TestMetricOption
            )
        super(BEVNetKittiTaskOption, self).__post_init__(mode, is_distributed)
