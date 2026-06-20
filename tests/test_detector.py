"""
Tests for PCB Defect Detector
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestDetection:
    """Test Detection dataclass."""

    def test_detection_to_dict(self):
        from detector.detector import Detection
        det = Detection(
            class_name="missing_hole",
            class_id=0,
            confidence=0.967,
            bbox=[10.0, 20.0, 100.0, 80.0],
            bbox_normalized=[0.5, 0.4, 0.3, 0.2],
        )
        d = det.to_dict()
        assert d["class_name"] == "missing_hole"
        assert d["class_id"] == 0
        assert abs(d["confidence"] - 0.967) < 0.001
        assert len(d["bbox"]) == 4
        assert len(d["bbox_normalized"]) == 4

    def test_detection_area(self):
        from detector.detector import Detection
        det = Detection(
            class_name="short",
            class_id=3,
            confidence=0.85,
            bbox=[10.0, 20.0, 110.0, 120.0],
            bbox_normalized=[0.5, 0.5, 0.2, 0.2],
        )
        # Area should be 100 * 100 = 10000
        assert det.area == pytest.approx(10000.0)

    def test_detection_default_crop_none(self):
        from detector.detector import Detection
        det = Detection(
            class_name="spur",
            class_id=4,
            confidence=0.55,
            bbox=[0, 0, 50, 50],
            bbox_normalized=[0.1, 0.1, 0.1, 0.1],
        )
        assert det.crop is None


class TestPCBDefectDetector:
    """Test PCBDefectDetector class."""

    def test_class_names_length(self):
        from detector.detector import PCBDefectDetector
        assert len(PCBDefectDetector.CLASS_NAMES) == 6

    def test_class_names_content(self):
        from detector.detector import PCBDefectDetector
        expected = [
            "missing_hole", "mouse_bite", "open_circuit",
            "short", "spur", "spurious_copper"
        ]
        assert PCBDefectDetector.CLASS_NAMES == expected

    def test_extract_crop_basic(self):
        """Test crop extraction with a simple synthetic image."""
        from detector.detector import PCBDefectDetector

        # Mock the model loading
        with patch.object(PCBDefectDetector, '_load_model', return_value=MagicMock()):
            with patch.object(PCBDefectDetector, '_load_config', return_value={}):
                detector = PCBDefectDetector.__new__(PCBDefectDetector)
                detector.crop_padding = 10
                detector.conf_threshold = 0.35
                detector.iou_threshold = 0.45
                detector.max_det = 300
                detector.img_size = 640
                detector.device = "cpu"

                image = np.zeros((200, 200, 3), dtype=np.uint8)
                crop = detector._extract_crop(image, 50, 50, 100, 100, padding=5)

                assert crop is not None
                assert crop.ndim == 3
                # With padding=5: x1=45, y1=45, x2=105, y2=105 → 60x60
                assert crop.shape[0] == 60
                assert crop.shape[1] == 60

    def test_draw_detections_returns_image(self):
        """Test that draw_detections returns same-shape image."""
        from detector.detector import PCBDefectDetector, Detection

        with patch.object(PCBDefectDetector, '_load_model', return_value=MagicMock()):
            with patch.object(PCBDefectDetector, '_load_config', return_value={}):
                detector = PCBDefectDetector.__new__(PCBDefectDetector)

                image = np.zeros((480, 640, 3), dtype=np.uint8)
                detections = [
                    Detection(
                        class_name="missing_hole", class_id=0,
                        confidence=0.9, bbox=[50., 50., 150., 100.],
                        bbox_normalized=[0.15, 0.15, 0.15, 0.07],
                    )
                ]
                result = detector.draw_detections(image, detections)
                assert result.shape == image.shape


class TestDatasetPreparation:
    """Test YOLO format conversion utilities."""

    def test_convert_to_yolo_format(self):
        from scripts.prepare_dataset import convert_to_yolo_format
        cx, cy, w, h = convert_to_yolo_format(10, 20, 110, 80, 200, 100)
        assert abs(cx - 0.3) < 1e-6   # (10+110)/2 / 200 = 60/200 = 0.3
        assert abs(cy - 0.5) < 1e-6   # (20+80)/2 / 100 = 50/100 = 0.5
        assert abs(w - 0.5) < 1e-6    # (110-10) / 200 = 100/200 = 0.5
        assert abs(h - 0.6) < 1e-6    # (80-20) / 100 = 60/100 = 0.6

    def test_convert_to_yolo_clamps_values(self):
        from scripts.prepare_dataset import convert_to_yolo_format
        cx, cy, w, h = convert_to_yolo_format(-10, -5, 110, 80, 100, 100)
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert 0.0 <= w <= 1.0
        assert 0.0 <= h <= 1.0

    def test_class_to_id_mapping(self):
        from scripts.prepare_dataset import CLASS_TO_ID, ID_TO_CLASS
        assert CLASS_TO_ID["missing_hole"] == 0
        assert CLASS_TO_ID["spurious_copper"] == 5
        assert len(CLASS_TO_ID) == 6
        assert ID_TO_CLASS[0] == "missing_hole"
