from torchvision.models.detection import FasterRCNN as FRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator


class ResNet50_FasterRCNN:

    def __init__(self, pretrained=False):
        # Building our FasterRCNN model for objects detection
        backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained)
        num_classes = 4 + 1

        anchor_generator = AnchorGenerator(sizes=(40, 60, 150, 200, 250), aspect_ratios=(0.7, 1.0, 1.3))
        self.model = FRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)

    def train(self):
        self.model.train()

    def to(self, device):
        self.model.to(device)

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def get_state_dict(self):
        return self.model.state_dict()

    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def fit_batch(self, images, target):
        return self.model(images, target)

    def predict_batch(self, images):
        return self.model(images)
