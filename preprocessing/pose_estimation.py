from typing import Dict, List

import torchvision
import torch
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights


class PoseEstimator:
    def __init__(self):
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()

    def predict(self, x: torch.Tensor, save_path: str | None = None) -> List[Dict[str, torch.Tensor]]:
        preds = self.model(x)
        if save_path:
            print(preds[0]['keypoints'])
            self._save_predictions(preds, save_path)
        return preds

    @staticmethod
    def _save_predictions(keypoints: torch.Tensor, path):
        torch.save(keypoints, path)


pe = PoseEstimator()

pe.predict(torch.rand(3, 3, 224, 224), 'keypoints.pth')