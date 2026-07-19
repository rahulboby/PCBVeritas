"""
Grad-CAM and EigenCAM for YOLOv8 PCB Defect Explanation
=========================================================
PURPOSE:
    Generate visual explanations (heatmaps) showing WHICH parts of
    the PCB image the neural network focused on when detecting a defect.
    This is the "explainability" in XAI (Explainable AI).

    Grad-CAM answers: "Why did the model predict 'missing_hole' here?"
    EigenCAM answers:  "What features most activated the detector?"

INPUT:
    - PCB image (numpy array)
    - Trained YOLOv8 model
    - Target class index (optional; if None, uses predicted class)

OUTPUT:
    - Heatmap array (H x W, values 0-1)
    - Overlay image (original + heatmap blended)
    - Comparison image (side-by-side)

HOW GRAD-CAM WORKS:
    1. Run a forward pass through the model
    2. Compute the gradient of the class score with respect to
       feature map activations in the target convolutional layer
    3. Average the gradients spatially (global average pooling)
       → these weights tell us "how important is each feature map?"
    4. Weighted sum of feature maps gives the heatmap
    5. Apply ReLU (only positive activations matter)
    6. Resize and overlay on original image

MATHEMATICS:
    For class c, the Grad-CAM heatmap L^c is:
    
    α^c_k = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)
    
    L^c = ReLU(Σ_k α^c_k * A^k)
    
    Where:
    - y^c = score for class c
    - A^k = k-th feature map activation
    - α^c_k = importance weight of feature map k for class c
    - Z = spatial size of feature map

HOW EIGENCAM WORKS:
    EigenCAM uses PCA (Principal Component Analysis) on feature maps
    instead of gradients. It's faster and doesn't require backprop.
    The first principal component of the feature map captures the
    most discriminative regions.

CONNECTS TO:
    - pipeline/orchestrator.py: Heatmaps generated in pipeline
    - app/pages/xai_page.py: Displayed in Streamlit UI
"""

import warnings
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
import torch
from loguru import logger

# pytorch-grad-cam library
try:
    from pytorch_grad_cam import GradCAM, EigenCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logger.warning(
        "pytorch-grad-cam not installed. XAI features will be disabled.\n"
        "Install with: pip install grad-cam"
    )

from ultralytics import YOLO


class YOLOv8GradCAMWrapper(torch.nn.Module):
    """
    Wrapper to make YOLOv8 compatible with pytorch-grad-cam.
    
    WHY IS THIS NEEDED?
    pytorch-grad-cam expects a standard classifier that outputs
    class scores for a single image. YOLOv8's output is more
    complex (multi-scale detection predictions). This wrapper
    extracts just the classification confidence for the target class
    from YOLOv8's output, making it compatible with CAM methods.
    """

    def __init__(self, model: YOLO, target_class: int = 0) -> None:
        super().__init__()
        self.model = model.model  # Access underlying PyTorch model
        self.target_class = target_class
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns classification-like scores.
        
        YOLOv8 outputs a list of tensors from different detection heads.
        We aggregate these into a single class score vector.
        
        Args:
            x: Input image tensor [B, C, H, W]

        Returns:
            Class confidence tensor [B, num_classes]
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = self.model(x)

        # YOLOv8 output shape depends on model version
        # For detection: outputs is a tuple (predictions, features)
        # predictions shape: [B, num_preds, 4 + num_classes]
        if isinstance(outputs, (list, tuple)):
            pred = outputs[0]
        else:
            pred = outputs

        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        if len(pred.shape) == 3:
            # Ultralytics detection tensors are usually [B, 4 + classes, anchors].
            # Some exports use [B, anchors, 4 + classes], so support both.
            if pred.shape[1] <= pred.shape[2] and pred.shape[1] >= 5:
                class_scores = pred[:, 4:, :].amax(dim=2)  # [B, num_classes]
            else:
                class_scores = pred[..., 4:].amax(dim=1)  # [B, num_classes]
            return class_scores

        # Fallback for different output formats: flatten to classifier-like logits.
        return pred.reshape(pred.shape[0], -1)


class PCBGradCAM:
    """
    Generates Grad-CAM and EigenCAM visual explanations for PCB defect detections.
    
    This class takes the trained YOLOv8 model and produces heatmaps that
    highlight which image regions were most important for the detection decision.
    
    Example:
        cam_generator = PCBGradCAM(yolo_model)
        heatmap, overlay = cam_generator.generate(image, method="GradCAM")
    """

    def __init__(
        self,
        yolo_model: YOLO,
        config: Optional[dict] = None,
    ) -> None:
        """
        Initialize with a loaded YOLO model.

        Args:
            yolo_model: Loaded Ultralytics YOLO model.
            config: XAI configuration dictionary.
        """
        if not GRADCAM_AVAILABLE:
            raise ImportError("Install pytorch-grad-cam: pip install grad-cam")

        self.yolo_model = yolo_model
        self.config = config or {}
        
        # Get the target layer from YOLOv8's backbone
        # model.model[-2] is typically the SPPF layer (Spatial Pyramid Pooling Fast)
        # which aggregates multi-scale features - good for CAM
        self.target_layer = self._get_target_layer()
        
        logger.info(
            f"PCBGradCAM initialized | "
            f"target_layer={self.config.get('grad_cam', {}).get('target_layer_name', 'auto')}"
        )

    def _get_target_layer(self) -> list:
        """
        Identify the target convolutional layer in YOLOv8.
        
        YOLO v8's model.model is a sequential list of blocks:
        - Layers 0-9: Backbone (feature extraction)
        - Layers 10+: Neck and Head (detection)
        
        Layer -2 (second to last of backbone) is a good CAM target
        because it has high-level semantic features while still having
        spatial resolution.

        Returns:
            List containing target layer (pytorch-grad-cam expects a list).
        """
        try:
            pytorch_model = self.yolo_model.model
            # model.model is the sequential module
            # Access second-to-last layer of the model backbone
            target = pytorch_model.model[-2]
            return [target]
        except (AttributeError, IndexError) as e:
            logger.warning(f"Could not access target layer: {e}. Trying fallback.")
            try:
                # Fallback: try to find a convolutional layer
                pytorch_model = self.yolo_model.model
                layers = list(pytorch_model.model.children())
                return [layers[-3]]  # Try third from last
            except Exception as e2:
                logger.error(f"Could not identify target layer: {e2}")
                raise RuntimeError("Cannot find suitable target layer for Grad-CAM")

    def generate(
        self,
        image: np.ndarray,
        method: str = "GradCAM",
        target_class: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a CAM explanation heatmap for the given image.

        Args:
            image: PCB image as BGR numpy array (from cv2.imread).
            method: 'GradCAM', 'GradCAMPlusPlus', or 'EigenCAM'.
            target_class: Class index to explain. If None, uses the highest-confidence class.

        Returns:
            Tuple of:
            - heatmap: Grayscale heatmap array (H x W, float32, values 0-1)
            - overlay: Color overlay image (H x W x 3, BGR uint8)
        """
        # --- Prepare input ---
        # Convert BGR (OpenCV) to RGB (PyTorch convention)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] float32 (required by show_cam_on_image)
        image_float = image_rgb.astype(np.float32) / 255.0

        # Convert to PyTorch tensor: [H, W, C] -> [1, C, H, W]
        # Resize to model input size for consistency
        h, w = image.shape[:2]
        img_size = 640  # YOLOv8 standard size
        image_resized = cv2.resize(image_rgb, (img_size, img_size))
        image_tensor = torch.from_numpy(
            image_resized.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)

        # Move to same device and dtype as model.
        first_param = next(self.yolo_model.model.parameters())
        device = first_param.device
        model_dtype = first_param.dtype if first_param.is_floating_point() else torch.float32
        image_tensor = image_tensor.to(device=device, dtype=model_dtype)
        image_tensor.requires_grad_(method != "EigenCAM")

        # --- Create model wrapper ---
        tc = target_class if target_class is not None else 0
        wrapped_model = YOLOv8GradCAMWrapper(self.yolo_model, tc)
        wrapped_model.eval()

        # --- Select CAM method ---
        cam_method_map = {
            "GradCAM": GradCAM,
            "GradCAMPlusPlus": GradCAMPlusPlus,
            "EigenCAM": EigenCAM,
        }
        
        if method not in cam_method_map:
            logger.warning(f"Unknown CAM method '{method}'. Using GradCAM.")
            method = "GradCAM"
        
        CamClass = cam_method_map[method]

        # --- Generate CAM ---
        try:
            # Grad-CAM needs gradients enabled (model may be in no_grad context).
            targets = [ClassifierOutputTarget(tc)] if target_class is not None and method != "EigenCAM" else None
            try:
                with torch.enable_grad():
                    with CamClass(model=wrapped_model, target_layers=self.target_layer) as cam:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            grayscale_cam = cam(
                                input_tensor=image_tensor,
                                targets=targets,
                            )
            except Exception as cam_error:
                if method == "EigenCAM":
                    raise
                logger.warning(f"{method} failed ({cam_error}); retrying with EigenCAM")
                image_tensor = image_tensor.detach()
                image_tensor.requires_grad_(False)
                with EigenCAM(model=wrapped_model, target_layers=self.target_layer) as cam:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        grayscale_cam = cam(input_tensor=image_tensor, targets=None)
                method = "EigenCAM"

            # grayscale_cam shape: [1, H, W] -> [H, W]
            heatmap = grayscale_cam[0]  # Values 0-1

            # Resize heatmap back to original image size
            heatmap_resized = cv2.resize(heatmap, (w, h))

            # --- Create color overlay ---
            # show_cam_on_image expects RGB float [0,1] image and heatmap [0,1]
            image_float_fullsize = cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB
            ).astype(np.float32) / 255.0
            
            overlay_rgb = show_cam_on_image(
                image_float_fullsize,
                heatmap_resized,
                use_rgb=True,
                colormap=cv2.COLORMAP_JET,
                image_weight=self.config.get("visualization", {}).get("alpha", 0.5),
            )
            
            # Convert overlay back to BGR for OpenCV compatibility
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

            logger.debug(
                f"Generated {method} heatmap | "
                f"size={heatmap_resized.shape} | "
                f"max_activation={heatmap_resized.max():.3f}"
            )

            return heatmap_resized, overlay_bgr

        except Exception as e:
            logger.error(f"CAM generation failed: {e}")
            # Return empty heatmap on failure
            empty_heatmap = np.zeros((h, w), dtype=np.float32)
            return empty_heatmap, image.copy()

    def generate_comparison(
        self,
        image: np.ndarray,
        overlay: np.ndarray,
        title: str = "Grad-CAM",
    ) -> np.ndarray:
        """
        Create side-by-side comparison: original image vs CAM overlay.

        Args:
            image: Original PCB image.
            overlay: CAM overlay image.
            title: Title string for the overlay side.

        Returns:
            Side-by-side comparison image.
        """
        h, w = image.shape[:2]

        # Resize overlay to match original if needed
        if overlay.shape[:2] != (h, w):
            overlay = cv2.resize(overlay, (w, h))

        # Add text labels
        orig_labeled = image.copy()
        cam_labeled = overlay.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(orig_labeled, "Original PCB", (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(cam_labeled, title, (10, 30), font, 0.8, (0, 255, 0), 2)

        # Concatenate side by side with a divider line
        divider = np.ones((h, 3, 3), dtype=np.uint8) * 128
        comparison = np.hstack([orig_labeled, divider, cam_labeled])

        return comparison

    def generate_per_class(
        self,
        image: np.ndarray,
        class_names: list[str],
        output_dir: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        """
        Generate Grad-CAM heatmaps for each defect class.
        
        This shows which regions would be activated for EACH class,
        which helps understand if the model is focusing on the right features.

        Args:
            image: PCB image.
            class_names: List of class names.
            output_dir: Optional directory to save individual heatmaps.

        Returns:
            Dict mapping class_name -> overlay image.
        """
        class_to_id = {
            "missing_hole": 0, "mouse_bite": 1, "open_circuit": 2,
            "short": 3, "spur": 4, "spurious_copper": 5,
        }

        results = {}
        
        for class_name in class_names:
            class_id = class_to_id.get(class_name, 0)
            try:
                heatmap, overlay = self.generate(image, method="GradCAM", target_class=class_id)
                results[class_name] = overlay

                if output_dir:
                    out_path = Path(output_dir) / f"gradcam_{class_name}.jpg"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), overlay)
                    logger.info(f"Saved per-class CAM: {out_path}")

            except Exception as e:
                logger.error(f"Failed to generate CAM for class {class_name}: {e}")

        return results

    def save_heatmap(
        self,
        heatmap: np.ndarray,
        overlay: np.ndarray,
        save_dir: str = "outputs/heatmaps",
        prefix: str = "gradcam",
    ) -> dict[str, str]:
        """
        Save heatmap and overlay images to disk.

        Args:
            heatmap: Grayscale heatmap array.
            overlay: Color overlay image.
            save_dir: Output directory.
            prefix: Filename prefix.

        Returns:
            Dict of saved file paths.
        """
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = {}

        # Save raw heatmap as grayscale PNG
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_path = str(out_dir / f"{prefix}_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap_colored)
        saved["heatmap"] = heatmap_path

        # Save overlay
        overlay_path = str(out_dir / f"{prefix}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay)
        saved["overlay"] = overlay_path

        logger.info(f"Saved heatmaps to: {out_dir}")
        return saved
