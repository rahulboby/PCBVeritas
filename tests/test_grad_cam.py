import numpy as np
import torch

from xai.grad_cam import PCBGradCAM


class DummyInner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DummyYOLO:
    def __init__(self) -> None:
        self.model = DummyInner()


def test_manual_gradcam_fallback_produces_nonzero_heatmap() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 0] = 255

    cam = PCBGradCAM(yolo_model=DummyYOLO(), config={"visualization": {"alpha": 0.5}})
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    heatmap = cam._manual_gradcam(
        image_tensor=image_tensor,
        target_class=0,
        image_shape=image.shape[:2],
    )

    assert heatmap.shape == image.shape[:2]
    assert heatmap.max() > 0
