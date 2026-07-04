"""
PCB Defect Detector - YOLOv8s
==============================
PURPOSE:
    Core inference class that wraps YOLOv8s for PCB defect detection.
    Provides a clean API for detecting, classifying, and localizing
    PCB defects from input images.

INPUT:
    - Image path (str or Path) or numpy array (BGR, as from cv2.imread)

OUTPUT:
    - List of Detection objects containing:
      * class_name: str (e.g., "missing_hole")
      * class_id: int (0-5)
      * confidence: float (0.0-1.0)
      * bbox: [x1, y1, x2, y2] in absolute pixels
      * bbox_normalized: [cx, cy, w, h] normalized 0-1
      * crop: numpy array of the defect region

HOW IT WORKS:
    1. Loads YOLOv8s model with trained weights
    2. Runs forward pass to get predictions
    3. Applies NMS (Non-Maximum Suppression) to remove duplicates
    4. Filters by confidence threshold
    5. Returns structured Detection results

YOLOv8 ARCHITECTURE (simplified):
    Input Image (640x640)
         ↓
    Backbone (CSPDarknet) - extracts multi-scale features
         ↓
    Neck (FPN+PAN) - combines features from different scales
         ↓
    Detection Head - predicts boxes, classes, confidence
         ↓
    NMS - removes duplicate detections
         ↓
    Final Detections

CONNECTS TO:
    - pipeline/orchestrator.py: Used in the main inference pipeline
    - xai/grad_cam.py: Detector model passed for CAM generation
    - retrieval/embedder.py: Defect crops passed for embedding
    - app/pages/: Results displayed in Streamlit
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import numpy as np
import cv2
import torch
import yaml
from loguru import logger
from ultralytics import YOLO


# Detection result dataclass
@dataclass
class Detection:
    """
    Represents a single detected defect.
    
    A dataclass is used here for clean attribute access and
    easy serialization to JSON.
    """
    class_name: str              # e.g., "missing_hole"
    class_id: int                # e.g., 0
    confidence: float            # e.g., 0.967
    bbox: list[float]            # [x1, y1, x2, y2] absolute pixels
    bbox_normalized: list[float] # [cx, cy, w, h] normalized 0-1
    crop: Optional[np.ndarray] = field(default=None, repr=False)  # Defect crop image
    
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict (excludes numpy crop)."""
        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 2) for v in self.bbox],
            "bbox_normalized": [round(v, 6) for v in self.bbox_normalized],
        }
    
    @property
    def area(self) -> float:
        """Bounding box area in pixels squared."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class PCBDefectDetector:
    """
    YOLOv8s-based PCB defect detector.
    
    This class handles model loading, inference, and post-processing.
    It is designed to be instantiated once and called multiple times
    for efficient batch processing.
    
    Example:
        detector = PCBDefectDetector("models/detector/best.pt")
        detections = detector.detect("pcb_image.jpg")
        for det in detections:
            print(f"{det.class_name}: {det.confidence:.1%} at {det.bbox}")
    """

    # PCB defect class names (must match training order)
    CLASS_NAMES = [
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper",
    ]

    def __init__(
        self,
        weights_path: str = "models/detector/best.pt",
        config: Optional[dict] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the detector by loading model weights.

        Args:
            weights_path: Path to trained YOLOv8 .pt weights file.
            config: Inference configuration dictionary.
            device: Override device ('cuda', 'cpu', '0'). Auto-detected if None.
        """
        self.weights_path = Path(weights_path)
        self.config = config or {}
        
        # Determine device
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "0"  # First GPU
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.warning("No GPU detected. Running on CPU (slower).")

        # Confidence and IoU thresholds from config
        self.conf_threshold = self.config.get("detection", {}).get("conf_threshold", 0.35)
        self.iou_threshold = self.config.get("detection", {}).get("iou_threshold", 0.45)
        self.max_det = self.config.get("detection", {}).get("max_detections", 300)
        self.img_size = self.config.get("detection", {}).get("image_size", 640)
        
        # Crop padding
        self.crop_padding = self.config.get("crops", {}).get("padding", 20) \
            if "crops" in self.config else 20

        # Load model
        self.model = self._load_model()
        
        logger.info(
            f"PCBDefectDetector ready | weights={self.weights_path.name} | "
            f"device={self.device} | conf={self.conf_threshold} | iou={self.iou_threshold}"
        )

    def _load_model(self) -> YOLO:
        """
        Load YOLOv8 model from weights file.
        
        YOLOv8 models are loaded using Ultralytics' YOLO class.
        The model is automatically moved to the target device.

        Returns:
            Loaded YOLO model instance.
        """
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {self.weights_path}\n"
                "Please train the model first: python detector/train.py\n"
                "Or download pretrained weights to: models/detector/best.pt"
            )

        logger.info(f"Loading YOLOv8 weights from: {self.weights_path}")
        model = YOLO(str(self.weights_path))
        
        return model

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        return_crops: bool = True,
    ) -> list[Detection]:
        """
        Run defect detection on a single PCB image.

        Args:
            image: Image path (str/Path) or numpy array (BGR format from cv2).
            return_crops: If True, extract and store defect crop images.

        Returns:
            List of Detection objects sorted by confidence (highest first).
        """
        # Load image if path is given
        if isinstance(image, (str, Path)):
            img_array = cv2.imread(str(image))
            if img_array is None:
                raise ValueError(f"Could not read image: {image}")
        else:
            img_array = image.copy()

        img_h, img_w = img_array.shape[:2]

        # --- Run YOLOv8 inference ---
        # Ultralytics handles preprocessing, forward pass, and NMS internally
        start_time = time.perf_counter()
        results = self.model.predict(
            source=img_array,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,  # Suppress ultralytics output
        )
        inference_time = time.perf_counter() - start_time

        logger.debug(f"Inference time: {inference_time*1000:.1f}ms")

        if not results or results[0].boxes is None:
            logger.info("No defects detected")
            return []

        result = results[0]  # Single image, take first result

        # --- Parse detections ---
        detections = []
        boxes = result.boxes

        # Convert tensors to numpy for processing
        if boxes.xyxy is not None and len(boxes.xyxy) > 0:
            bboxes_abs = boxes.xyxy.cpu().numpy()    # [x1, y1, x2, y2] in pixels
            confidences = boxes.conf.cpu().numpy()   # confidence scores
            class_ids = boxes.cls.cpu().numpy().astype(int)  # class indices

            for bbox, conf, class_id in zip(bboxes_abs, confidences, class_ids):
                x1, y1, x2, y2 = bbox

                # Get class name
                if class_id < len(self.CLASS_NAMES):
                    class_name = self.CLASS_NAMES[class_id]
                else:
                    logger.warning(f"Unknown class ID: {class_id}")
                    class_name = f"class_{class_id}"

                # Compute normalized bbox [cx, cy, w, h]
                cx = (x1 + x2) / 2.0 / img_w
                cy = (y1 + y2) / 2.0 / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                bbox_norm = [cx, cy, bw, bh]

                # Extract crop if requested
                crop = None
                if return_crops:
                    crop = self._extract_crop(img_array, x1, y1, x2, y2)

                detections.append(Detection(
                    class_name=class_name,
                    class_id=int(class_id),
                    confidence=float(conf),
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    bbox_normalized=bbox_norm,
                    crop=crop,
                ))

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.info(
            f"Detected {len(detections)} defects | "
            f"Classes: {[d.class_name for d in detections]} | "
            f"Time: {inference_time*1000:.0f}ms"
        )

        return detections

    def _extract_crop(
        self,
        image: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
        padding: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract a defect crop from the image with optional padding.

        Padding adds context around the detected bbox, which helps
        the SigLIP embedder understand the defect's surroundings.

        Args:
            image: Full PCB image (BGR numpy array).
            x1, y1, x2, y2: Bounding box coordinates.
            padding: Pixels to add around bbox (uses config default if None).

        Returns:
            Cropped region as numpy array.
        """
        pad = padding if padding is not None else self.crop_padding
        h, w = image.shape[:2]

        # Add padding and clamp to image boundaries
        x1_pad = max(0, int(x1) - pad)
        y1_pad = max(0, int(y1) - pad)
        x2_pad = min(w, int(x2) + pad)
        y2_pad = min(h, int(y2) + pad)

        # Ensure valid crop dimensions
        if x2_pad <= x1_pad or y2_pad <= y1_pad:
            return image[max(0,int(y1)):min(h,int(y2)), max(0,int(x1)):min(w,int(x2))]

        return image[y1_pad:y2_pad, x1_pad:x2_pad].copy()

    def detect_batch(
        self,
        images: list[Union[str, Path, np.ndarray]],
        return_crops: bool = True,
    ) -> list[list[Detection]]:
        """
        Run detection on multiple images (more efficient than calling detect() in a loop).

        Args:
            images: List of image paths or numpy arrays.
            return_crops: Extract crop images.

        Returns:
            List of detection lists, one per input image.
        """
        results_all = []
        for img in images:
            results_all.append(self.detect(img, return_crops=return_crops))
        return results_all

    def draw_detections(
        self,
        image: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on a PCB image.

        Uses color-coded boxes per defect class for easy visual distinction.

        Args:
            image: Input PCB image (BGR numpy array).
            detections: List of Detection objects.

        Returns:
            Image with drawn bounding boxes and labels.
        """
        # Class colors (BGR for OpenCV): each class gets a distinct color
        colors = {
            "missing_hole": (0, 0, 255),       # Red
            "mouse_bite": (0, 128, 255),        # Orange
            "open_circuit": (255, 0, 255),      # Magenta
            "short": (255, 0, 0),              # Blue
            "spur": (0, 255, 0),               # Green
            "spurious_copper": (0, 255, 255),  # Yellow/Cyan
        }

        vis_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = colors.get(det.class_name, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{det.class_name.replace('_', ' ').title()} {det.confidence:.1%}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(vis_image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)

            # Draw label text
            cv2.putText(
                vis_image, label, (x1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

        return vis_image

    def get_model_info(self) -> dict:
        """
        Return model information and statistics.

        Returns:
            Dict with model parameters and configuration.
        """
        return {
            "weights": str(self.weights_path),
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "classes": self.CLASS_NAMES,
            "num_classes": len(self.CLASS_NAMES),
        }
