"""
Configuration Settings for PCB-VLM-XAI
Replaces the previous YAML configuration files.
"""

STREAMLIT_CONFIG = {
    "app": {
        "title": "PCB-VLM-XAI: Intelligent PCB Inspection System",
        "subtitle": "Explainable Vision-Language Defect Analysis",
        "icon": "🔬",
        "layout": "wide",
        "theme": "dark"
    },
    "upload": {
        "max_file_size_mb": 50,
        "allowed_formats": ["jpg", "jpeg", "png", "bmp", "tiff"]
    },
    "pages": [
        {"name": "Upload & Detect", "icon": "📷"},
        {"name": "XAI Visualizations", "icon": "🔍"},
        {"name": "Similar Defects", "icon": "🔎"},
        {"name": "Knowledge Insights", "icon": "📚"},
        {"name": "Inspection Report", "icon": "📋"}
    ],
    "display": {
        "thumbnail_size": [200, 200],
        "gallery_columns": 3,
        "confidence_bar_color": "#00ff88",
        "critical_color": "#ff4444",
        "high_color": "#ff8800",
        "medium_color": "#ffcc00",
        "low_color": "#44ff44"
    },
    "models": {
        "detector_weights": "models/detector/best.pt",
        "embedding_index": "data/embeddings/faiss_index.bin",
        "embedding_metadata": "data/embeddings/metadata.json",
        "llm_dir": "models/llm/fine_tuned_qwen",
        "siglip_dir": "models/embeddings/siglip"
    },
    "cache": {
        "enabled": True,
        "ttl_seconds": 3600
    }
}

INFERENCE_CONFIG = {
    "model": {
        "weights": "models/detector/best.pt",
        "device": "0"
    },
    "detection": {
        "conf_threshold": 0.35,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "image_size": 640,
        "augment": False
    },
    "classes": [
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper"
    ],
    "severity": {
        "missing_hole": "high",
        "mouse_bite": "medium",
        "open_circuit": "critical",
        "short": "critical",
        "spur": "low",
        "spurious_copper": "medium"
    },
    "visualization": {
        "show_labels": True,
        "show_confidence": True,
        "line_thickness": 2,
        "font_scale": 0.6,
        "color_map": {
            "missing_hole": [255, 0, 0],
            "mouse_bite": [255, 128, 0],
            "open_circuit": [255, 0, 255],
            "short": [0, 0, 255],
            "spur": [0, 255, 0],
            "spurious_copper": [0, 255, 255]
        }
    },
    "output": {
        "save_dir": "outputs/detections",
        "save_images": True,
        "save_json": True,
        "save_txt": False
    }
}

TRAINING_CONFIG = {
    "model": {
        "architecture": "yolov8s",
        "pretrained": True,
        "pretrained_weights": "yolov8s.pt"
    },
    "dataset": {
        "path": "data/splits/dataset.yaml",
        "train": "data/splits/train",
        "val": "data/splits/val",
        "test": "data/splits/test",
        "classes": [
            "missing_hole",
            "mouse_bite",
            "open_circuit",
            "short",
            "spur",
            "spurious_copper"
        ],
        "num_classes": 6
    },
    "training": {
        "epochs": 100,
        "batch_size": 16,
        "image_size": 640,
        "workers": 4,
        "device": "0",
        "seed": 42
    },
    "optimizer": {
        "name": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1
    },
    "augmentation": {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0
    },
    "loss": {
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5
    },
    "evaluation": {
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_det": 300
    },
    "output": {
        "project": "runs/detect",
        "name": "pcb_defect_detector",
        "save": True,
        "save_period": 10,
        "plots": True,
        "verbose": True
    },
    "mlflow": {
        "enabled": True,
        "experiment_name": "PCB_Defect_Detection",
        "tracking_uri": "logs/mlflow"
    }
}

RETRIEVAL_CONFIG = {
    "embedding": {
        "model_name": "google/siglip-base-patch16-224",
        "model_dir": "models/embeddings/siglip",
        "image_size": 224,
        "embedding_dim": 768,
        "batch_size": 32,
        "device": "cuda",
        "normalize": True
    },
    "faiss": {
        "index_type": "IndexFlatIP",
        "index_path": "data/embeddings/faiss_index.bin",
        "metadata_path": "data/embeddings/metadata.json",
        "nprobe": 10
    },
    "retrieval": {
        "top_k": 3,
        "min_similarity": 0.5,
        "same_class_weight": 0.3
    },
    "crops": {
        "padding": 20,
        "min_size": 32,
        "save_crops": True,
        "crops_dir": "data/processed/crops"
    },
    "output": {
        "retrieved_dir": "outputs/retrieved",
        "show_similarity_score": True
    }
}

XAI_CONFIG = {
    "grad_cam": {
        "target_layer_name": "model.model[-2]",
        "method": "GradCAM",
        "use_cuda": True
    },
    "eigen_cam": {
        "target_layer_name": "model.model[-2]",
        "method": "EigenCAM"
    },
    "visualization": {
        "colormap": "COLORMAP_JET",
        "alpha": 0.5,
        "normalize": True
    },
    "output": {
        "heatmaps_dir": "outputs/heatmaps",
        "save_raw_heatmap": True,
        "save_overlay": True,
        "save_comparison": True,
        "dpi": 150
    },
    "per_class_cam": {
        "enabled": True,
        "classes": [
            "missing_hole",
            "mouse_bite",
            "open_circuit",
            "short",
            "spur",
            "spurious_copper"
        ]
    }
}

LLM_CONFIG = {
    "model": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "local_dir": "models/llm/qwen2.5-1.5b",
        "device_map": "auto",
        "torch_dtype": "float16",
        "trust_remote_code": True
    },
    "generation": {
        "max_new_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "do_sample": True
    },
    "tokenizer": {
        "max_input_length": 2048,
        "padding_side": "left"
    },
    "system_prompt": (
        "You are an expert PCB (Printed Circuit Board) inspection engineer with 20 years of \n"
        "experience in electronics manufacturing quality control. You analyze PCB defects and \n"
        "provide detailed technical reports for manufacturing engineers.\n\n"
        "When given defect information, you must provide:\n"
        "1. Clear defect description\n"
        "2. Root cause analysis\n"
        "3. Severity assessment\n"
        "4. Manufacturing process implications\n"
        "5. Specific repair or corrective action recommendations\n\n"
        "Be precise, technical, and actionable. Format your response as a structured report.\n"
    ),
    "output": {
        "reports_dir": "outputs/reports",
        "format": "markdown"
    }
}

FINE_TUNING_CONFIG = {
    "base_model": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "local_dir": "models/llm/qwen2.5-1.5b",
        "torch_dtype": "float16",
        "trust_remote_code": True
    },
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    },
    "training": {
        "output_dir": "models/llm/fine_tuned_qwen",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "warmup_steps": 100,
        "learning_rate": 2.0e-4,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "save_steps": 100,
        "eval_steps": 100,
        "logging_steps": 10,
        "max_grad_norm": 1.0,
        "fp16": True,
        "bf16": False,
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
        "group_by_length": True
    },
    "dataset": {
        "path": "data/fine_tuning/pcb_instructions.json",
        "train_split": 0.9,
        "val_split": 0.1,
        "max_length": 1024,
        "seed": 42
    },
    "generation": {
        "synthetic_samples": 500,
        "output_path": "data/fine_tuning/pcb_instructions.json"
    },
    "evaluation": {
        "compute_metrics": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False
    },
    "mlflow": {
        "enabled": True,
        "experiment_name": "PCB_LLM_FineTuning"
    }
}
