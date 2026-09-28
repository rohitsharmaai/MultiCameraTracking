"""Assemble the production object graph from config + registry."""

from __future__ import annotations

import logging
from typing import Optional

from src.config import AppConfig, load_config
from src.database.repository import IdentityRepository
from src.database.session import create_engine_from_url, init_db
from src.detection.rfdetr import RFDETRPersonDetector, configure_cuda_runtime
from src.events import EventBus
from src.face.auraface import AuraFaceEncoder
from src.face.yunet import YuNetDetector
from src.identity.manager import GlobalIdentityManager
from src.identity.matcher import IdentityMatcher, MatcherConfig
from src.identity.topology import CameraTopology
from src.observation.manager import ObservationManager
from src.pipeline.features import FeatureExtractor
from src.pipeline.video import VideoProcessor
from src.pose.rtmpose import RTMPoseBody2d
from src.registry import ModelRegistry
from src.reid.osnet import OSNetEncoder
from src.reid.youtu import YoutuReIDEncoder
from src.segmentation.pphumanseg import PPHumanSeg


logger = logging.getLogger(__name__)


def build_registry(cfg: AppConfig) -> ModelRegistry:
    return ModelRegistry(
        models_yaml=cfg.project_root / "configs" / "models.yaml",
        manifest_yaml=cfg.project_root / "models" / "MANIFEST.yaml",
        models_dir=cfg.models_dir,
    )


def build_store(cfg: AppConfig):
    engine = create_engine_from_url(cfg.database_url, environment=cfg.environment)
    init_db(engine)
    return IdentityRepository(engine)


def build_manager(cfg: AppConfig, store=None, events: Optional[EventBus] = None) -> GlobalIdentityManager:
    store = store or build_store(cfg)
    topology = CameraTopology(cfg.topology)
    scoring = cfg.thresholds.get("scoring", {})
    reid = cfg.thresholds.get("reid", {})
    cam = cfg.thresholds.get("camera_transition", {})
    matcher = IdentityMatcher(
        MatcherConfig(
            accept_threshold=float(reid.get("body_similarity_threshold", 0.62)),
            minimum_score_margin=float(reid.get("minimum_score_margin", 0.08)),
            scoring=scoring,
            topology_enabled=bool(cam.get("topology_enabled", True)),
            temporal_enabled=bool(cam.get("temporal_constraints_enabled", True)),
            unknown_edge_penalty=float(cam.get("unknown_edge_penalty", 0.15)),
            max_gap=float(cam.get("default_max_transition_seconds", 180.0)),
            body_similarity_threshold=float(reid.get("body_similarity_threshold", 0.62)),
            face_similarity_threshold=float(reid.get("face_similarity_threshold", 0.45)),
            gallery_support_k=int(reid.get("gallery_support_k", 1)),
        ),
        topology,
    )
    return GlobalIdentityManager(store, matcher, topology, cfg.thresholds, events=events)


def build_models(cfg: AppConfig, registry: ModelRegistry, *, load_weights: bool = True):
    det_cfg = cfg.thresholds.get("detection", {})
    detector = RFDETRPersonDetector(
        registry,
        confidence_threshold=float(det_cfg.get("confidence_threshold", 0.4)),
        person_class_id=int(det_cfg.get("person_class_id", 1)),
        input_resolution=int(det_cfg.get("input_resolution", 704)),
        allow_cpu_fallback=cfg.allow_cpu_fallback,
    )
    obs_cfg = cfg.thresholds.get("observation", {})
    segmenter = PPHumanSeg(registry)
    pose = RTMPoseBody2d(registry)
    body_model = str(cfg.thresholds.get("reid", {}).get("body_model", "OSNet-x1.0"))
    body = YoutuReIDEncoder(registry) if body_model == "YoutuReID" else OSNetEncoder(registry)
    use_colour = bool(obs_cfg.get("body_colour", False))
    if body_model == "YoutuReID" and not use_colour:
        # A plain Youtu vector (768) has the length of an OSNet+colour vector
        # (512+256); keep the fused layout so the two can never be confused.
        logger.warning("reid.body_model=YoutuReID requires observation.body_colour; enabling it")
        use_colour = True
    face_det = YuNetDetector(registry)
    face_enc = AuraFaceEncoder(registry)
    if load_weights:
        configure_cuda_runtime()
        for model in (segmenter, pose, body, face_det, face_enc):
            try:
                if hasattr(model, "name") and model.name in {"PPHumanSeg", "RTMPose-Body2d", "OSNet-x1.0", "YoutuReID", "YuNet", "AuraFace-v1"}:
                    path = registry.resolve_path(registry.get(model.name))
                    if path is not None and path.exists():
                        registry.verify_on_disk(model.name)
                        model.load()
            except FileNotFoundError:
                pass
        try:
            detector.load()
        except Exception as exc:
            if load_weights:
                raise RuntimeError(
                    "RF-DETR-L failed to load. Install `rfdetr` and use RFDETRLarge only. "
                    f"Original error: {exc}"
                ) from exc
    extractor = FeatureExtractor(
        segmenter=segmenter,
        pose=pose,
        body_encoder=body,
        face_detector=face_det,
        face_encoder=face_enc,
        min_mask_coverage=float(obs_cfg.get("min_mask_coverage", 0.25)),
        require_unique_face=bool(obs_cfg.get("face_require_unique_bbox", False)),
        max_body_overlap=float(obs_cfg.get("max_body_overlap", 1.0)),
        min_face_px=float(obs_cfg.get("min_face_px", 0.0)),
        use_colour=use_colour,
    )
    return detector, extractor


def build_processor(
    cfg: Optional[AppConfig] = None,
    *,
    load_weights: bool = True,
    detector=None,
    extractor=None,
    store=None,
    snapshot_store=None,
) -> VideoProcessor:
    cfg = cfg or load_config()
    events = EventBus()
    registry = build_registry(cfg)
    store = store or build_store(cfg)
    manager = build_manager(cfg, store=store, events=events)
    if detector is None or extractor is None:
        built_det, built_ext = build_models(cfg, registry, load_weights=load_weights)
        detector = detector or built_det
        extractor = extractor or built_ext
    if snapshot_store is None:
        # Save body/face crops for every entry point (CLI and GUI).  Without
        # this the gallery persists embeddings but the operator UI has no
        # images to show after a rerun.
        from src.gui.snapshots import SnapshotStore, default_snapshot_dir

        snapshot_store = SnapshotStore(default_snapshot_dir(cfg.project_root))
    obs_cfg = cfg.thresholds.get("observation", {})
    observations = ObservationManager(
        top_k_body=int(obs_cfg.get("top_k_body", 8)),
        top_k_face=int(obs_cfg.get("top_k_face", 3)),
        min_body_quality=float(obs_cfg.get("min_body_quality", 0.35)),
        min_face_quality=float(obs_cfg.get("min_face_quality", 0.45)),
        min_bbox_area=float(obs_cfg.get("min_bbox_area", 1600)),
        min_sharpness=float(obs_cfg.get("min_sharpness", 20.0)),
        quality_weights=cfg.thresholds.get("quality", {}),
        events=events,
    )
    return VideoProcessor(
        manager=manager,
        detector=detector,
        feature_extractor=extractor,
        observation_manager=observations,
        store=store,
        events=events,
        tracking_cfg=cfg.thresholds.get("tracking", {}),
        output_dir=cfg.project_root / "outputs",
        snapshot_store=snapshot_store,
    )
