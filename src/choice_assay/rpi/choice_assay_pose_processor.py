from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from expidite_rpi.core import api, file_naming
from expidite_rpi.core import configuration as root_cfg
from expidite_rpi.core.dp import DataProcessor
from expidite_rpi.core.dp_config_objects import DataProcessorCfg, Stream
from ultralytics import YOLO
from ultralytics.engine.results import Results

logger = root_cfg.setup_logger("choice_assay")

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
]

CA_MARKED_UP_VID_DATA_TYPE_ID = "CAMARKEDUP"
CA_MARKED_UP_VID_STREAM_INDEX: int = 1


@dataclass
class ChoiceAssayPoseProcessorCfg(DataProcessorCfg):
    model_path: Path
    keypoint_count: int = len(CA_KEYPOINT_NAMES)
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
            sample_probability=1.0,
        ),
    ],
    # model_path=Path(__file__).resolve().parent.parent / "resources" / "beecam_ncnn_model",
    model_path=Path(__file__).resolve().parent.parent.parent.parent
    / "ml_runs"
    / "pose"
    / "bee_pose"
    / "weights"
    / "best.pt",
)


class ChoiceAssayPoseProcessor(DataProcessor):
    def __init__(self, config: ChoiceAssayPoseProcessorCfg, sensor_index: int) -> None:
        super().__init__(config, sensor_index)
        self.dp_config = config

    def _load_model(self) -> YOLO:
        model_path = self.dp_config.model_path
        if not model_path.exists():
            msg = f"Pose model not found at {model_path}"
            raise FileNotFoundError(msg)

        return YOLO(model_path)

    def _select_keypoints(self, result: Results, keypoint_count: int) -> np.ndarray | None:
        keypoints = result.keypoints
        if keypoints is None:
            return None

        kpt_data = keypoints.data.cpu().numpy()
        if kpt_data.size == 0:
            return None

        boxes = result.boxes
        if boxes is not None:
            conf = boxes.conf
            if conf is not None and len(conf) > 0:
                best_idx = int(conf.argmax().item())
            else:
                best_idx = 0
        else:
            best_idx = 0

        return kpt_data[best_idx]

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

        for idx in range(self.dp_config.keypoint_count):
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
                verbose=True,
                conf=0.25,
                max_det=1,
                save=save_markup_video,
                save_dir=markup_dir,
            )

            # Process the YOLO results frame by frame as they are generated
            for frame_index, result in enumerate(results):
                keypoints = self._select_keypoints(result, self.dp_config.keypoint_count)

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
