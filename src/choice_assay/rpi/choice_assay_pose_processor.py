from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import ultralytics
from expidite_rpi.core import api, file_naming
from expidite_rpi.core import configuration as root_cfg
from expidite_rpi.core.dp import DataProcessor
from expidite_rpi.core.dp_config_objects import DataProcessorCfg, Stream
from ultralytics import YOLO
from ultralytics.engine.results import Results

logger = root_cfg.setup_logger("choice_assay")

BEE_CLASS_ID = 0
TUBES_CLASS_ID = 1

CA_XY_DATA_TYPE_ID = "CAPOSE"
CA_XY_STREAM_INDEX: int = 0
CA_KEYPOINT_NAMES: list[str] = [
    "L_antenna",
    "R_antenna",
    "L_mandible",
    "R_mandible",
    "Top_prob",
    "Tube_prob",
    "End_prob",
    "L_tube",
    "R_tube",
]
KEYPOINT_COUNT = len(CA_KEYPOINT_NAMES)

CA_MARKED_UP_VID_DATA_TYPE_ID = "CAMARKEDUP"
CA_MARKED_UP_VID_STREAM_INDEX: int = 1


@dataclass
class ChoiceAssayPoseProcessorCfg(DataProcessorCfg):
    model_path: Path
    fps: int = 5


DEFAULT_CHOICE_ASSAY_POSE_PROCESSOR_CFG = ChoiceAssayPoseProcessorCfg(
    description="YOLO pose processor for choice assay sub-videos",
    outputs=[
        Stream(
            description="Pose keypoints per frame for choice assay clips",
            type_id=CA_XY_DATA_TYPE_ID,
            index=CA_XY_STREAM_INDEX,
            format=api.FORMAT.DF,
            fields=(
                [f"{name}_{suffix}" for name in CA_KEYPOINT_NAMES for suffix in ["x", "y", "conf"]]
                + [
                    "source_filename",
                    "frame_index",
                    "frame_start_time",
                ]
            ),
        ),
        Stream(
            description="Marked up videos with pose keypoints drawn on frames",
            type_id=CA_MARKED_UP_VID_DATA_TYPE_ID,
            index=CA_MARKED_UP_VID_STREAM_INDEX,
            format=api.FORMAT.AVI,
            cloud_container="expidite-choiceassay-markedup",
            sample_probability=0.02,
        ),
    ],
    model_path=Path(__file__).resolve().parent.parent / "resources" / "best.pt",
)


class ChoiceAssayPoseProcessor(DataProcessor):
    def __init__(self, config: ChoiceAssayPoseProcessorCfg, sensor_index: int) -> None:
        super().__init__(config, sensor_index)
        self.dp_config = config
        self._model_diagnostics_logged = False

    def _log_model_diagnostics(self, model: YOLO) -> None:
        if self._model_diagnostics_logged:
            return

        names = getattr(model, "names", None)
        if isinstance(names, dict):
            name_keys = sorted(int(k) for k in names)
            logger.info(
                "Pose model diagnostics: ultralytics=%s model_path=%s names_count=%d names_keys=%s",
                ultralytics.__version__,
                self.dp_config.model_path,
                len(names),
                name_keys,
            )
        else:
            logger.warning(
                "Pose model diagnostics: ultralytics=%s model_path=%s names type is %s",
                ultralytics.__version__,
                self.dp_config.model_path,
                type(names).__name__,
            )

        self._model_diagnostics_logged = True

    def _log_first_frame_diagnostics(self, result: Results, video_path: Path) -> None:
        boxes = result.boxes
        if boxes is None or boxes.cls is None:
            logger.info("Pose frame diagnostics: video=%s frame=0 detected_classes=[]", video_path)
            return

        class_ids = [int(c.item()) for c in boxes.cls]
        unique_class_ids = sorted(set(class_ids))
        names = getattr(result, "names", None)
        missing_names = []
        if isinstance(names, dict):
            missing_names = [class_id for class_id in unique_class_ids if class_id not in names]

        logger.info(
            "Pose frame diagnostics: video=%s frame=0 detected_classes=%s missing_name_ids=%s",
            video_path,
            unique_class_ids,
            missing_names,
        )

    def _load_model(self) -> YOLO:
        model_path = self.dp_config.model_path
        if not model_path.exists():
            msg = f"Pose model not found at {model_path}"
            raise FileNotFoundError(msg)

        model = YOLO(model_path)

        self._log_model_diagnostics(model)

        return model

    def _select_keypoints(self, result: Results) -> np.ndarray | None:
        """ Selects the keypoints from the YOLO result.
        We expect 0 or 1 bee detections (class 0) per frame, and
        usually 1 tubes detection (class 1) per frame.
        We want to select the highest-confidence detection for each class in case we get duplicate detections.
        The bee data is keypoints 0-6 and the tube data is keypoints 7-8, so we need to combine the detections
        into a single data structure for the output."""
        keypoints = result.keypoints
        if keypoints is None or keypoints.data is None:
            return None
        assert isinstance(keypoints.data, torch.Tensor), (
            f"Expected keypoints.data to be a torch.Tensor, got {type(keypoints.data)}"
        )

        kpt_data = keypoints.data.cpu().numpy()
        if kpt_data.size == 0:
            return None

        boxes = result.boxes
        if boxes is None or boxes.cls is None or boxes.conf is None:
            return None

        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        # Output is always the full keypoint layout expected by downstream code.
        # Start with NaNs so missing class detections remain explicit.
        combined_keypoints = np.full((KEYPOINT_COUNT, 3), np.nan, dtype=float)

        def _best_class_index(class_id: int) -> int | None:
            class_indices = np.where(cls == class_id)[0]
            if class_indices.size == 0:
                return None
            best_local = class_indices[np.argmax(conf[class_indices])]
            return int(best_local)

        bee_idx = _best_class_index(BEE_CLASS_ID)
        tubes_idx = _best_class_index(TUBES_CLASS_ID)

        if bee_idx is None and tubes_idx is None:
            return None

        if bee_idx is not None:
            bee_kpts = kpt_data[bee_idx]
            bee_count = min(7, bee_kpts.shape[0], KEYPOINT_COUNT)
            combined_keypoints[:bee_count] = bee_kpts[:bee_count]

        if tubes_idx is not None:
            tubes_kpts = kpt_data[tubes_idx]
            combined_keypoints[7:9] = tubes_kpts[:2]

        return combined_keypoints

    def _frame_to_row(
        self,
        frame_index: int,
        keypoints: np.ndarray,
        source_filename: str,
        start_time: pd.Timestamp,
    ) -> dict:
        frame_start_time = start_time + timedelta(seconds=frame_index / self.dp_config.fps)

        row = {
            "source_filename": source_filename,
            "frame_index": frame_index,
            "frame_start_time": frame_start_time,
        }

        for idx in range(KEYPOINT_COUNT):
            keypoint_name = CA_KEYPOINT_NAMES[idx]
            row[f"{keypoint_name}_x"] = float(keypoints[idx, 0])
            row[f"{keypoint_name}_y"] = float(keypoints[idx, 1])
            row[f"{keypoint_name}_conf"] = float(keypoints[idx, 2])
        return row

    def _process_video_file(self, video_path: Path) -> pd.DataFrame:
        try:
            parts = file_naming.parse_record_filename(video_path)
            start_time: datetime = parts.get(api.RECORD_ID.TIMESTAMP.value, api.utc_now())
            end_time: datetime = parts.get(api.RECORD_ID.END_TIME.value, start_time)

            rows: list[dict] = []

            save_markup_video = self.save_sample(
                self.get_stream(CA_MARKED_UP_VID_STREAM_INDEX).sample_probability
            )
            markup_dir = root_cfg.TMP_DIR / "YOLO"

            model = self._load_model()
            results = model(
                video_path,
                stream=True,
                verbose=False,
                conf=0.25,
                classes=[0, 1], # Bee, Tubes
                save=save_markup_video,
                save_dir=markup_dir,
            )

            # Process the YOLO results frame by frame as they are generated
            try:
                for frame_index, result in enumerate(results):
                    if frame_index == 0:
                        self._log_first_frame_diagnostics(result, video_path)

                    keypoints = self._select_keypoints(result)

                    # Only save a row if the model produced a result for the frame.
                    # If the model fails to produce a result, we skip saving data for that frame.
                    if keypoints is not None:
                        row = self._frame_to_row(
                            frame_index,
                            keypoints,
                            video_path.name,
                            start_time,
                        )
                        rows.append(row)
            except KeyError:
                names = getattr(model, "names", None)
                name_keys = sorted(int(k) for k in names) if isinstance(names, dict) else []
                logger.exception(
                    "KeyError while iterating YOLO stream for video=%s ultralytics=%s names_keys=%s",
                    video_path,
                    ultralytics.__version__,
                    name_keys,
                )
                raise

            if save_markup_video:
                marked_up_video_path = markup_dir / (video_path.stem + ".avi")
                self.save_recording(
                    stream_index=CA_MARKED_UP_VID_STREAM_INDEX,
                    temporary_file=marked_up_video_path,
                    start_time=start_time,
                    end_time=end_time,
                    override_sampling=api.OVERRIDE.SAVE,
                )

            return pd.DataFrame(rows)
        except Exception:
            logger.exception("Error processing video file %s", video_path)
            return pd.DataFrame()

    def process_data(self, input_data: pd.DataFrame | list[Path]) -> None:
        assert isinstance(input_data, list), f"Expected list of files, got {type(input_data)}"
        files: list[Path] = input_data  # type: ignore[invalid-assignment]
        results: list[pd.DataFrame] = []

        for f in files:
            try:
                result = self._process_video_file(f)
                results.append(result)
            except Exception:
                logger.exception(f"{root_cfg.RAISE_WARN()}Exception occurred processing video {f!s}")

        if results:
            output_df = pd.concat(results)
            self.save_data(stream_index=CA_XY_STREAM_INDEX, sensor_data=output_df)
