"""YOLOv8-based PCB defect detection module."""
from detector.detector import PCBDefectDetector
from detector.train import train_detector
from detector.inference import run_inference

__all__ = ["PCBDefectDetector", "train_detector", "run_inference"]
