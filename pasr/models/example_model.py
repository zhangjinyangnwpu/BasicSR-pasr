from collections import OrderedDict

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class PASRModel(SRModel):
    def init_training_settings(self):
        pass

    def feed_data(self, data):
        pass

    def optimize_parameters(self, current_iter):
        pass
