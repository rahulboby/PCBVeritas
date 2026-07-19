"""
PCB Inspection Pipeline Orchestrator
======================================
PURPOSE:
    Coordinates the full end-to-end PCB inspection pipeline:
    Image -> Detection -> XAI -> Embedding -> Retrieval -> Knowledge -> LLM -> Report

    This is the central coordinator that calls all modules in sequence
    and assembles the final inspection result.

PIPELINE DIAGRAM:
    PCB Image
      -> YOLOv8s Detection
      -> Grad-CAM/EigenCAM and SigLIP crop embeddings
      -> FAISS similar-case retrieval
      -> Knowledge Engine context
      -> OpenAI-compatible LLM API
      -> Inspection Report

INPUT:
    PCB image (path or numpy array)

OUTPUT:
    PipelineResult object containing:
    - detections: List[Detection]
    - heatmap: numpy array
    - overlay: numpy array
    - retrieved_cases: Dict[int, List[Dict]]
    - knowledge_contexts: Dict[str, str]
    - report: str (markdown)
    - processing_time: float

CONNECTS TO:
    - app/main.py: Entry point for Streamlit app
    - All module classes
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import numpy as np
import cv2
from loguru import logger


@dataclass
class PipelineResult:
    """
    Complete result from the PCB inspection pipeline.
    
    All outputs are stored here and passed to the Streamlit app
    or saved to disk.
    """
    # Input
    image: np.ndarray = field(repr=False)
    image_path: str = ""

    # Detection results
    detections: list = field(default_factory=list)
    detection_image: Optional[np.ndarray] = field(default=None, repr=False)
    num_defects: int = 0

    # XAI results
    heatmap: Optional[np.ndarray] = field(default=None, repr=False)
    cam_overlay: Optional[np.ndarray] = field(default=None, repr=False)
    xai_panel: Optional[np.ndarray] = field(default=None, repr=False)

    # Retrieval results: {detection_index: [similar_case_dicts]}
    retrieved_cases: dict = field(default_factory=dict)

    # Knowledge context: {defect_class: knowledge_string}
    knowledge_contexts: dict = field(default_factory=dict)

    # LLM Report
    report: str = ""

    # Timing
    processing_time: float = 0.0
    stage_times: dict = field(default_factory=dict)

    # Status
    success: bool = True
    error_message: str = ""

    def get_summary(self) -> dict:
        """Return a JSON-serializable summary."""
        return {
            "num_defects": self.num_defects,
            "detections": [d.to_dict() for d in self.detections],
            "retrieved_cases": {
                str(k): v for k, v in self.retrieved_cases.items()
            },
            "processing_time": round(self.processing_time, 3),
            "stage_times": {k: round(v, 3) for k, v in self.stage_times.items()},
            "success": self.success,
        }


class PCBInspectionPipeline:
    """
    End-to-end PCB defect inspection pipeline.

    Modules are loaded lazily on first use to save memory.
    Each module can be disabled individually via configuration.

    Example:
        pipeline = PCBInspectionPipeline()
        result = pipeline.run("pcb_image.jpg")
        print(result.report)
    """

    def __init__(
        self,
        enable_xai: bool = True,
        enable_retrieval: bool = True,
        enable_llm: bool = True,
    ) -> None:
        """
        Initialize the pipeline.

        Args:
            enable_xai: Enable Grad-CAM heatmap generation.
            enable_retrieval: Enable FAISS similarity search.
            enable_llm: Enable LLM report generation.
        """
        self.enable_xai = enable_xai
        self.enable_retrieval = enable_retrieval
        self.enable_llm = enable_llm

        # Module instances (lazy-loaded)
        self._detector = None
        self._cam_generator = None
        self._xai_visualizer = None
        self._embedder = None
        self._search_engine = None
        self._knowledge_engine = None
        self._report_generator = None
        self._llm_unavailable_reason = ""
        self.model_status: dict[str, str] = {}
        logger.info(
            f"PCBInspectionPipeline initialized | "
            f"xai={enable_xai} | retrieval={enable_retrieval} | llm={enable_llm}"
        )

    # ============================================================
    # Lazy module loaders
    # ============================================================

    def _get_detector(self):
        if self._detector is None:
            from detector.detector import PCBDefectDetector
            from configs.settings import INFERENCE_CONFIG
            weights = INFERENCE_CONFIG.get("model", {}).get("weights", "models/detector/best.pt")
            self._detector = PCBDefectDetector(
                weights_path=weights,
                config=INFERENCE_CONFIG,
            )
        return self._detector

    def _get_cam_generator(self):
        if self._cam_generator is None and self.enable_xai:
            try:
                from xai.grad_cam import PCBGradCAM
                from configs.settings import XAI_CONFIG
                detector = self._get_detector()
                self._cam_generator = PCBGradCAM(
                    yolo_model=detector.model,
                    config=XAI_CONFIG,
                )
            except Exception as e:
                logger.warning(f"Could not initialize CAM generator: {e}")
        return self._cam_generator

    def _get_xai_visualizer(self):
        if self._xai_visualizer is None:
            from xai.visualizer import XAIVisualizer
            from configs.settings import XAI_CONFIG
            self._xai_visualizer = XAIVisualizer(
                config=XAI_CONFIG
            )
        return self._xai_visualizer

    def _get_embedder(self):
        if self._embedder is None and self.enable_retrieval:
            try:
                from retrieval.embedder import SigLIPEmbedder
                from configs.settings import RETRIEVAL_CONFIG
                self._embedder = SigLIPEmbedder(
                    config=RETRIEVAL_CONFIG
                )
            except Exception as e:
                logger.warning(f"Could not initialize embedder: {e}")
        return self._embedder

    def _get_search_engine(self):
        if self._search_engine is None and self.enable_retrieval:
            try:
                from retrieval.faiss_search import FAISSSearchEngine
                from configs.settings import RETRIEVAL_CONFIG
                engine = FAISSSearchEngine(
                    config=RETRIEVAL_CONFIG
                )
                engine.load()
                self._search_engine = engine
            except Exception as e:
                logger.warning(f"Could not initialize search engine: {e}")
        return self._search_engine

    def _get_knowledge_engine(self):
        if self._knowledge_engine is None:
            from knowledge.knowledge_engine import KnowledgeEngine
            self._knowledge_engine = KnowledgeEngine()
        return self._knowledge_engine

    def _get_report_generator(self):
        if self._llm_unavailable_reason:
            logger.warning(f"LLM unavailable: {self._llm_unavailable_reason}")
            return None

        if self._report_generator is None and self.enable_llm:
            try:
                from llm.inference.report_generator import PCBReportGenerator
                from configs.settings import LLM_CONFIG
                self._report_generator = PCBReportGenerator(
                    config=LLM_CONFIG,
                )
            except Exception as e:
                logger.warning(f"Could not initialize report generator: {e}")
        return self._report_generator

    # ============================================================
    # Pipeline stages    # ============================================================

    def _stage_detect(self, image: np.ndarray) -> tuple:
        """Run YOLOv8 detection."""
        detector = self._get_detector()
        detections = detector.detect(image, return_crops=True)
        detection_image = detector.draw_detections(image, detections)
        return detections, detection_image

    def _stage_xai(self, image: np.ndarray, detections: list) -> tuple:
        """Generate Grad-CAM heatmap."""
        cam = self._get_cam_generator()
        if cam is None or not detections:
            h, w = image.shape[:2]
            heatmap = np.zeros((h, w), dtype=np.float32)
            overlay = image.copy()
            return heatmap, overlay, None

        # Use the highest-confidence detection's class for CAM
        top_det = detections[0]
        try:
            heatmap, overlay = cam.generate(
                image,
                method="GradCAM",
                target_class=top_det.class_id,
            )
        except Exception as e:
            logger.warning(f"CAM generation failed: {e}")
            h, w = image.shape[:2]
            heatmap = np.zeros((h, w), dtype=np.float32)
            overlay = image.copy()

        # Create full XAI panel
        viz = self._get_xai_visualizer()
        try:
            xai_panel = viz.create_full_panel(
                original=image,
                detections=detections,
                heatmap=heatmap,
                overlay=overlay,
                title=f"Grad-CAM: {top_det.class_name.replace('_',' ').title()}",
            )
        except Exception as e:
            logger.warning(f"XAI panel creation failed: {e}")
            xai_panel = None

        return heatmap, overlay, xai_panel

    def _stage_retrieve(self, detections: list) -> dict:
        """Retrieve similar defects from FAISS index."""
        embedder = self._get_embedder()
        search_engine = self._get_search_engine()

        if embedder is None or search_engine is None:
            return {}

        indexed_crops = [(idx, det.crop) for idx, det in enumerate(detections) if det.crop is not None]
        if not indexed_crops:
            return {}

        retrieved = {idx: [] for idx, _ in indexed_crops}
        try:
            embeddings = embedder.embed_batch([crop for _, crop in indexed_crops])
        except Exception as e:
            logger.warning(f"Retrieval embedding failed: {e}")
            return retrieved

        for (idx, _), embedding in zip(indexed_crops, embeddings):
            try:
                retrieved[idx] = search_engine.search(embedding)
            except Exception as e:
                logger.warning(f"Retrieval search failed for detection {idx}: {e}")

        return retrieved

    def _stage_knowledge(self, detections: list) -> dict:
        """Retrieve knowledge base context for each defect."""
        engine = self._get_knowledge_engine()
        contexts = {}
        for det in detections:
            try:
                contexts[det.class_name] = engine.format_for_rag_prompt(
                    det.class_name, det.confidence
                )
            except Exception as e:
                logger.warning(f"Knowledge retrieval failed for {det.class_name}: {e}")
        return contexts

    def _stage_report(
        self,
        detections: list,
        retrieved_cases: dict,
        knowledge_contexts: dict,
    ) -> str:
        """Generate LLM inspection report."""
        generator = self._get_report_generator()
        if generator is None:
            return self._fallback_report(detections, knowledge_contexts)

        knowledge_engine = self._get_knowledge_engine()
        try:
            report = generator.generate_multi_defect_report(
                detections=[d.to_dict() for d in detections],
                knowledge_engine=knowledge_engine,
                retrieved_cases_map=retrieved_cases,
            )
            return report
        except Exception as e:
            logger.error(f"LLM report generation failed: {e}")
            return self._fallback_report(detections, knowledge_contexts)

    def _fallback_report(self, detections: list, knowledge_contexts: dict) -> str:
        """Generate a template-based report when LLM is unavailable."""
        if not detections:
            return "# PCB Inspection Report\n\n**Result:** No defects detected. PCB passed visual inspection."

        lines = ["# PCB INSPECTION REPORT (Template Mode)", ""]
        lines.append(f"**Defects Found:** {len(detections)}\n")

        for i, det in enumerate(detections, 1):
            lines.append(f"## Defect {i}: {det.class_name.replace('_', ' ').title()}")
            lines.append(f"- **Confidence:** {det.confidence:.1%}")
            lines.append(f"- **Bounding Box:** {[int(v) for v in det.bbox]}\n")

            ctx = knowledge_contexts.get(det.class_name, "")
            if ctx:
                lines.append(ctx)
            lines.append("")

        lines.append("---")
        lines.append("*Report generated by PCB-VLM-XAI (template mode -- LLM unavailable)*")
        return "\n".join(lines)

    # ============================================================
    # Main run method
    # ============================================================

    def run(
        self,
        image: Union[str, Path, np.ndarray],
        run_xai: Optional[bool] = None,
        run_retrieval: Optional[bool] = None,
        run_llm: Optional[bool] = None,
    ) -> PipelineResult:
        """
        Run the complete PCB inspection pipeline.

        Args:
            image: PCB image path or numpy array.
            run_xai: Override enable_xai setting.
            run_retrieval: Override enable_retrieval setting.
            run_llm: Override enable_llm setting.

        Returns:
            PipelineResult with all outputs.
        """
        pipeline_start = time.perf_counter()
        stage_times = {}

        # Resolve feature flags
        do_xai = run_xai if run_xai is not None else self.enable_xai
        do_retrieval = run_retrieval if run_retrieval is not None else self.enable_retrieval
        do_llm = run_llm if run_llm is not None else self.enable_llm

        # Load image
        if isinstance(image, (str, Path)):
            image_path = str(image)
            img_array = cv2.imread(image_path)
            if img_array is None:
                return PipelineResult(
                    image=np.zeros((640, 640, 3), dtype=np.uint8),
                    success=False,
                    error_message=f"Could not read image: {image_path}",
                )
        else:
            img_array = image.copy()
            image_path = ""

        result = PipelineResult(image=img_array, image_path=image_path)

        try:
            # Stage 1: Detection
            logger.info("Pipeline Stage 1: Detection")
            t0 = time.perf_counter()
            detections, detection_image = self._stage_detect(img_array)
            stage_times["detection"] = time.perf_counter() - t0
            result.detections = detections
            result.detection_image = detection_image
            result.num_defects = len(detections)
            logger.info(f"-> {len(detections)} defects detected in {stage_times['detection']*1000:.0f}ms")

            # Stage 2: XAI
            if do_xai:
                logger.info("Pipeline Stage 2: XAI (Grad-CAM)")
                t0 = time.perf_counter()
                heatmap, overlay, xai_panel = self._stage_xai(img_array, detections)
                stage_times["xai"] = time.perf_counter() - t0
                result.heatmap = heatmap
                result.cam_overlay = overlay
                result.xai_panel = xai_panel
                logger.info(f"-> Heatmap generated in {stage_times['xai']*1000:.0f}ms")

            # Stage 3: Retrieval
            if do_retrieval and detections:
                logger.info("Pipeline Stage 3: Retrieval (FAISS)")
                t0 = time.perf_counter()
                retrieved = self._stage_retrieve(detections)
                stage_times["retrieval"] = time.perf_counter() - t0
                result.retrieved_cases = retrieved
                total_retrieved = sum(len(v) for v in retrieved.values())
                logger.info(f"-> {total_retrieved} similar cases retrieved in {stage_times['retrieval']*1000:.0f}ms")

            # Stage 4: Knowledge
            logger.info("Pipeline Stage 4: Knowledge Base")
            t0 = time.perf_counter()
            knowledge_contexts = self._stage_knowledge(detections)
            stage_times["knowledge"] = time.perf_counter() - t0
            result.knowledge_contexts = knowledge_contexts

            # Stage 5: LLM Report
            if do_llm:
                logger.info("Pipeline Stage 5: LLM Report Generation")
                t0 = time.perf_counter()
                report = self._stage_report(
                    detections, result.retrieved_cases, knowledge_contexts
                )
                stage_times["llm"] = time.perf_counter() - t0
                result.report = report
                logger.info(f"-> Report generated in {stage_times['llm']*1000:.0f}ms")
            else:
                result.report = self._fallback_report(detections, knowledge_contexts)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            result.success = False
            result.error_message = str(e)

        result.processing_time = time.perf_counter() - pipeline_start
        result.stage_times = stage_times

        logger.info(
            f"Pipeline complete | "
            f"defects={result.num_defects} | "
            f"total_time={result.processing_time:.2f}s"
        )

        return result

    def preload_all(self) -> None:
        """
        Pre-load all available models into memory.

        The detector and SigLIP run a tiny warmup pass so CUDA kernels are
        initialized before the user uploads an image. The LLM step initializes
        an OpenAI-compatible API client; if its configuration is invalid, the
        pipeline falls back to the template report.
        """
        logger.info("Pre-loading all pipeline models...")
        self.model_status = {}

        detector = self._get_detector()
        detector.warmup()
        self.model_status["detector"] = f"ready on {detector.device}"

        if self.enable_xai:
            self._get_cam_generator()
            self._get_xai_visualizer()
            self.model_status["xai"] = "ready on detector device"

        if self.enable_retrieval:
            embedder = self._get_embedder()
            if embedder:
                embedder.warmup()
                self.model_status["siglip"] = f"ready on {embedder.device}"
            search_engine = self._get_search_engine()
            if search_engine:
                self.model_status["faiss"] = "ready"

        self._get_knowledge_engine()
        self.model_status["knowledge"] = "ready"

        if self.enable_llm:
            gen = self._get_report_generator()
            if gen:
                try:
                    gen.load_model()
                    gen.warmup()
                    self.model_status["llm"] = gen.device_summary()
                except Exception as e:
                    self._llm_unavailable_reason = str(e)
                    self._report_generator = None
                    self.model_status["llm"] = f"disabled: {e}"
                    logger.error(f"LLM API preload failed: {e}")

        logger.info("All available models pre-loaded.")


from typing import Optional, Union
