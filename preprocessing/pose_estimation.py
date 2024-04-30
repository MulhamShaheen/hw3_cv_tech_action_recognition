from pathlib import Path
from typing import Dict, List

import torchvision
import torch
from PIL.Image import Image
from PIL.ImageShow import show
from torch import tensor
from torchvision.io import read_image
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
from torchvision.transforms import ToTensor
from torchvision.utils import draw_keypoints
from transformers.image_transforms import to_pil_image


class PoseEstimator:
    def __init__(self, device: str = 'cpu'):
        if device == 'cuda:0' and not torch.cuda.is_available():
            print("CUDA is not available on this device. Using CPU instead.")
            self.device = 'cpu'
        else:
            self.device = device

        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT, num_keypoints=17, pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        self.detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.detection_model.to(self.device)
        self.detection_model.eval()

    def predict(self, x: List[torch.Tensor], save_path: str | Path | None = None, preprocess: bool = True) -> List[
        Dict[str, torch.Tensor]]:
        if preprocess:
            for i, img in enumerate(x):
                x[i] = self._detect_main_person(img)
                if x[i] is None:
                    return []
                x[i] = x[i].to(self.device)

        preds = self.model(x)
        for person_pred in preds:
            if person_pred["boxes"].shape[0] == 0:
                return []
            for key in person_pred:
                person_pred[key] = person_pred[key][0].unsqueeze(0)

        if save_path:
            self._save_predictions(preds, save_path)
        return preds

    def _detect_main_person(self, x: torch.Tensor) -> torch.Tensor | None:
        image: Image = to_pil_image(x)
        detects = self.detection_model(image)

        image_center = torch.tensor([image.width / 2, image.height / 2])

        min_distance = float('inf')
        closest_detection = None

        for detect in detects.xyxy[0]:
            detection_center = (detect[:2] + detect[2:4]) / 2
            distance = torch.dist(image_center.to(self.device), detection_center.to(self.device))
            if distance < min_distance:
                min_distance = distance
                closest_detection = detect

        if closest_detection is None:
            return None
        x1, y1, x2, y2 = closest_detection[:4].cpu().detach().numpy().astype(int)
        image = image.crop((x1, y1, x2, y2))

        return ToTensor()(image)

    @staticmethod
    def _save_predictions(keypoints: torch.Tensor, path):
        torch.save(keypoints, path)

