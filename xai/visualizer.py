"""
XAI Visualizer
==============
PURPOSE:
    Combines detection bounding boxes with Grad-CAM heatmaps to create
    rich visual explanations. Produces the final XAI output image shown
    in the Streamlit app and saved to disk.

INPUT:
    - Original PCB image
    - List of Detection objects
    - Grad-CAM heatmap

OUTPUT:
    - Multi-panel visualization: original | detections | heatmap | overlay
    - Per-defect crop with heatmap inset

CONNECTS TO:
    - pipeline/orchestrator.py: Called after detection + CAM generation
    - app/pages/xai_page.py: Images displayed in Streamlit
"""

from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from loguru import logger

from detector.detector import Detection


# Color map for each defect class (RGB)
CLASS_COLORS_RGB = {
    "missing_hole":     (255, 80,  80),
    "mouse_bite":       (255, 165,  0),
    "open_circuit":     (255,  0, 255),
    "short":            (80,  80, 255),
    "spur":             (80, 255,  80),
    "spurious_copper":  (0,  255, 220),
}


class XAIVisualizer:
    """
    Creates polished multi-panel XAI visualizations combining
    object detection and class activation maps.
    """

    def __init__(self, config_path: str = "configs/xai.yaml") -> None:
        import yaml
        try:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {}

    def create_full_panel(
        self,
        original: np.ndarray,
        detections: list[Detection],
        heatmap: np.ndarray,
        overlay: np.ndarray,
        title: str = "PCB XAI Analysis",
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Create a 4-panel visualization:
        [Original] [Detections] [Heatmap] [Overlay+Boxes]

        Args:
            original: BGR PCB image.
            detections: List of Detection objects.
            heatmap: Grayscale CAM heatmap (values 0-1).
            overlay: CAM color overlay (BGR).
            title: Figure title.
            save_path: Optional path to save the figure.

        Returns:
            Combined panel as BGR numpy array.
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(title, fontsize=14, fontweight="bold", color="white")
        fig.patch.set_facecolor("#1a1a2e")

        for ax in axes:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#0f3460")

        # Panel 1: Original image
        orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        axes[0].imshow(orig_rgb)
        axes[0].set_title("Original PCB", color="white", fontsize=11)
        axes[0].axis("off")

        # Panel 2: Detections overlay
        det_img = self._draw_detections_matplotlib(original.copy(), detections)
        det_rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
        axes[1].imshow(det_rgb)
        axes[1].set_title(f"Detections ({len(detections)})", color="white", fontsize=11)
        axes[1].axis("off")

        # Add detection count annotation
        if detections:
            severity_colors = {
                "critical": "red", "high": "orange",
                "medium": "yellow", "low": "lime"
            }
            for i, det in enumerate(detections[:5]):  # Show max 5 labels
                axes[1].text(
                    0.02, 0.98 - i * 0.1,
                    f"{det.class_name.replace('_',' ').title()}: {det.confidence:.0%}",
                    transform=axes[1].transAxes,
                    fontsize=7, color="white",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f3460", alpha=0.8),
                )

        # Panel 3: Heatmap
        axes[2].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[2].set_title("Grad-CAM Heatmap", color="white", fontsize=11)
        axes[2].axis("off")
        plt.colorbar(
            plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1)),
            ax=axes[2],
            fraction=0.046,
            pad=0.04,
        ).set_label("Activation", color="white")

        # Panel 4: CAM overlay with bounding boxes
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay_with_boxes = self._draw_boxes_on_image(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), detections
        )
        axes[3].imshow(overlay_with_boxes)
        axes[3].set_title("CAM + Detections", color="white", fontsize=11)
        axes[3].axis("off")

        # Add legend for defect classes
        patches = []
        detected_classes = set(d.class_name for d in detections)
        for cls in detected_classes:
            color = [c / 255 for c in CLASS_COLORS_RGB.get(cls, (255, 255, 255))]
            patches.append(mpatches.Patch(color=color, label=cls.replace("_", " ").title()))

        if patches:
            axes[3].legend(
                handles=patches, loc="lower right", fontsize=7,
                facecolor="#1a1a2e", labelcolor="white",
                framealpha=0.8, edgecolor="#0f3460"
            )

        plt.tight_layout()

        # Save figure
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                       facecolor="#1a1a2e", edgecolor="none")
            logger.info(f"XAI panel saved: {save_path}")

        # Convert matplotlib figure to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        panel_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        plt.close(fig)

        return panel_bgr

    def create_defect_crop_explanation(
        self,
        detection: Detection,
        heatmap: np.ndarray,
        full_image: np.ndarray,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Create a focused explanation for a single detected defect crop.

        Shows: [Crop] [Crop Heatmap] [Crop Overlay]

        Args:
            detection: Single Detection object with crop image.
            heatmap: Full image heatmap.
            full_image: Full PCB image for context.
            save_path: Optional save path.

        Returns:
            Explanation panel as numpy array.
        """
        if detection.crop is None:
            logger.warning("Detection has no crop image")
            return np.zeros((200, 600, 3), dtype=np.uint8)

        crop = detection.crop
        h_crop, w_crop = crop.shape[:2]

        # Extract heatmap region corresponding to the crop bbox
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        h_full, w_full = full_image.shape[:2]

        # Clamp to image bounds
        x1c = max(0, x1 - 20)
        y1c = max(0, y1 - 20)
        x2c = min(w_full, x2 + 20)
        y2c = min(h_full, y2 + 20)

        # Extract heatmap sub-region
        hm_h, hm_w = heatmap.shape[:2]
        x1_hm = int(x1c * hm_w / w_full)
        y1_hm = int(y1c * hm_h / h_full)
        x2_hm = int(x2c * hm_w / w_full)
        y2_hm = int(y2c * hm_h / h_full)

        crop_heatmap = heatmap[y1_hm:y2_hm, x1_hm:x2_hm]
        if crop_heatmap.size == 0:
            crop_heatmap = np.zeros((h_crop, w_crop), dtype=np.float32)
        else:
            crop_heatmap = cv2.resize(crop_heatmap, (w_crop, h_crop))

        # Color map the crop heatmap
        hm_colored = cv2.applyColorMap(
            (crop_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # Overlay
        crop_float = crop.astype(np.float32) / 255.0
        hm_float = hm_colored.astype(np.float32) / 255.0
        alpha = 0.5
        crop_overlay = cv2.addWeighted(crop, int((1-alpha)*255), hm_colored, int(alpha*255), 0)
        crop_overlay = cv2.addWeighted(crop, 1-alpha, hm_colored, alpha, 0)

        # Stack panels horizontally
        target_size = (128, 128)
        crop_r = cv2.resize(crop, target_size)
        hm_r = cv2.resize(hm_colored, target_size)
        ov_r = cv2.resize(crop_overlay, target_size)

        combined = np.hstack([crop_r, hm_r, ov_r])

        # Add title bar
        title_bar = np.zeros((30, combined.shape[1], 3), dtype=np.uint8)
        title_bar[:] = (30, 20, 60)
        label = (
            f"{detection.class_name.replace('_',' ').title()} | "
            f"Conf: {detection.confidence:.1%}"
        )
        cv2.putText(title_bar, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.45, (255, 255, 255), 1)

        explanation = np.vstack([title_bar, combined])

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, explanation)

        return explanation

    def _draw_detections_matplotlib(
        self, image: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        """Draw bounding boxes on image using color-coded per class."""
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color_rgb = CLASS_COLORS_RGB.get(det.class_name, (255, 255, 255))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 2)

            label = f"{det.class_name.replace('_', ' ')} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color_bgr, -1)
            cv2.putText(image, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        return image

    def _draw_boxes_on_image(
        self, image_rgb: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        """Draw detection boxes on an RGB image (for matplotlib display)."""
        img = image_rgb.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = CLASS_COLORS_RGB.get(det.class_name, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        return img
