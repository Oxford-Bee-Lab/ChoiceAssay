from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter

import cv2
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
TUBES_CLASS_ID = 0

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
    bee_model_path: Path
    tubes_model_path: Path
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
    bee_model_path=Path(__file__).resolve().parent.parent / "resources" / "bee_best.pt",
    tubes_model_path=Path(__file__).resolve().parent.parent / "resources" / "tubes_best.pt",
)


class ChoiceAssayPoseProcessor(DataProcessor):
    def __init__(self, config: ChoiceAssayPoseProcessorCfg, sensor_index: int) -> None:
        super().__init__(config, sensor_index)
        self.dp_config = config
        self._model_diagnostics_logged = False
        self._bee_model: YOLO | None = None
        self._tubes_model: YOLO | None = None

    def _log_model_diagnostics(self, model: YOLO) -> None:
        if self._model_diagnostics_logged:
            return

        names = getattr(model, "names", None)
        if isinstance(names, dict):
            name_keys = sorted(int(k) for k in names)
            logger.info(
                "Pose model diagnostics: ultralytics=%s model_path=%s names_count=%d names_keys=%s",
                ultralytics.__version__,
                self.dp_config.bee_model_path,
                len(names),
                name_keys,
            )
        else:
            logger.warning(
                "Pose model diagnostics: ultralytics=%s model_path=%s names type is %s",
                ultralytics.__version__,
                self.dp_config.bee_model_path,
                type(names).__name__,
            )

        self._model_diagnostics_logged = True

    def _load_models(self) -> tuple[YOLO, YOLO]:
        if self._bee_model is not None and self._tubes_model is not None:
            return self._bee_model, self._tubes_model

        bee_model_path = self.dp_config.bee_model_path
        if not bee_model_path.exists():
            msg = f"Pose model not found at {bee_model_path}"
            raise FileNotFoundError(msg)

        tubes_model_path = self.dp_config.tubes_model_path
        if not tubes_model_path.exists():
            msg = f"Pose model not found at {tubes_model_path}"
            raise FileNotFoundError(msg)

        bee_model = YOLO(bee_model_path)
        tubes_model = YOLO(tubes_model_path)

        self._log_model_diagnostics(bee_model)
        self._log_model_diagnostics(tubes_model)
        self._bee_model = bee_model
        self._tubes_model = tubes_model
        return bee_model, tubes_model

    def _extract_first_frame(self, video_path: Path) -> np.ndarray | None:
        """Extract the first frame from the video using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Failed to open video file %s", video_path)
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.error("Failed to read first frame from video file %s", video_path)
            return None

        return frame

    def _get_tubes(
        self, model: YOLO, video_path: Path
    ) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
        """Get the L_tube and R_tube keypoints from the first frame of the video."""
        # First extract the first frame from the video
        frame = self._extract_first_frame(video_path)
        if frame is None:
            return None, None

        frame_width = frame.shape[1]

        results = model(frame, stream=False, verbose=False, conf=0.7, imgsz=416, max_det=2)
        if not results:
            return None, None

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.cls is None or boxes.conf is None or len(boxes.cls) == 0:
            return None, None

        # The model returns bounding boxes for any tubes identified.
        # We expect 0, 1 or 2 tube detections (class 0) per frame.
        # We assign the detection with the higher X value to R_tube and the lower one to L_tube.
        # If we only get 1 tube, we assign it to L_tube if X < 0.5 and to R_tube if X >= 0.5.
        outcome = None, None
        if len(boxes.cls) == 2:
            tube_boxes = boxes.xyxy.cpu().numpy()
            tube_centers_x = (tube_boxes[:, 0] + tube_boxes[:, 2]) / 2
            sorted_indices = np.argsort(tube_centers_x)
            L_tube_idx, R_tube_idx = sorted_indices
            L_tube_x = tube_centers_x[L_tube_idx]
            R_tube_x = tube_centers_x[R_tube_idx]
            L_tube_y = tube_boxes[L_tube_idx, 3]
            R_tube_y = tube_boxes[R_tube_idx, 3]
            L_tube_conf = boxes.conf.cpu().numpy()[L_tube_idx]
            R_tube_conf = boxes.conf.cpu().numpy()[R_tube_idx]
            outcome = (L_tube_x, L_tube_y, L_tube_conf), (R_tube_x, R_tube_y, R_tube_conf)
        elif len(boxes.cls) == 1:
            tube_boxes = boxes.xyxy.cpu().numpy()
            tube_centers_x = (tube_boxes[:, 0] + tube_boxes[:, 2]) / 2
            tube_idx = 0
            tube_x = tube_centers_x[tube_idx]
            tube_y = tube_boxes[tube_idx, 3]
            tube_conf = boxes.conf.cpu().numpy()[tube_idx]
            if (tube_x / frame_width) < 0.5:
                outcome = (tube_x, tube_y, tube_conf), None
            else:
                outcome = None, (tube_x, tube_y, tube_conf)
        else:
            assert len(boxes.cls) == 0

        return outcome

    def _select_keypoints(
        self,
        result: Results,
    ) -> np.ndarray | None:
        """Selects the keypoints from the YOLO result.
        We expect 0 or 1 bee detections (class 0) per frame.
        The bee data is keypoints 0-6.
        """
        keypoints = result.keypoints
        if keypoints is None or keypoints.data is None:
            return None
        assert isinstance(keypoints.data, torch.Tensor), (
            f"Expected keypoints.data to be a torch.Tensor, got {type(keypoints.data)}"
        )

        kpt_data = keypoints.data.cpu().numpy()
        if kpt_data.shape[0] != 1:
            return None

        boxes = result.boxes
        if boxes is None or boxes.cls is None or boxes.conf is None:
            return None

        # Return the 7 keypoints for the single instance of the bee (class 0)
        return kpt_data[0, 0:7, :]

    def _frame_to_row(
        self,
        frame_index: int,
        keypoints: np.ndarray,
        L_tube: tuple[float, float, float] | None,
        R_tube: tuple[float, float, float] | None,
        source_filename: str,
        start_time: pd.Timestamp,
    ) -> dict:
        frame_start_time = start_time + timedelta(seconds=frame_index / self.dp_config.fps)

        row = {
            "source_filename": source_filename,
            "frame_index": frame_index,
            "frame_start_time": frame_start_time,
        }

        # First add the bee keypoints (0-6)
        for idx in range(KEYPOINT_COUNT - 2):
            keypoint_name = CA_KEYPOINT_NAMES[idx]
            row[f"{keypoint_name}_x"] = float(keypoints[idx, 0])
            row[f"{keypoint_name}_y"] = float(keypoints[idx, 1])
            row[f"{keypoint_name}_conf"] = float(keypoints[idx, 2])

        # Then add the tube keypoints (7-8)
        if L_tube is not None:
            row["L_tube_x"] = float(L_tube[0])
            row["L_tube_y"] = float(L_tube[1])
            row["L_tube_conf"] = float(L_tube[2])
        else:
            row["L_tube_x"] = None
            row["L_tube_y"] = None
            row["L_tube_conf"] = None

        if R_tube is not None:
            row["R_tube_x"] = float(R_tube[0])
            row["R_tube_y"] = float(R_tube[1])
            row["R_tube_conf"] = float(R_tube[2])
        else:
            row["R_tube_x"] = None
            row["R_tube_y"] = None
            row["R_tube_conf"] = None

        return row

    def _process_video_file(self, video_path: Path) -> pd.DataFrame:
        try:
            t0 = perf_counter()
            parts = file_naming.parse_record_filename(video_path)
            start_time: datetime = parts.get(api.RECORD_ID.TIMESTAMP.value, api.utc_now())
            end_time: datetime = parts.get(api.RECORD_ID.END_TIME.value, start_time)

            rows: list[dict] = []

            save_markup_video = self.save_sample(
                self.get_stream(CA_MARKED_UP_VID_STREAM_INDEX).sample_probability
            )
            markup_dir = root_cfg.TMP_DIR / "YOLO"

            t_model_start = perf_counter()
            bee_model, tubes_model = self._load_models()
            t_model_done = perf_counter()

            # First get the L_tube and R_tube points for the first frame
            t_tubes_start = perf_counter()
            L_tube, R_tube = self._get_tubes(tubes_model, video_path)
            t_tubes_done = perf_counter()

            t_predict_start = perf_counter()
            results = bee_model(
                video_path,
                stream=True,
                verbose=False,
                conf=0.25,
                imgsz=416,
                max_det=1,
                save=save_markup_video,
                save_dir=markup_dir,
            )
            t_predict_setup_done = perf_counter()

            # Process the YOLO results frame by frame as they are generated
            frames_seen = 0
            frames_with_keypoints = 0
            try:
                for frame_index, result in enumerate(results):
                    frames_seen += 1

                    keypoints = self._select_keypoints(result)

                    # Only save a row if the model produced a result for the frame.
                    # If the model fails to produce a result, we skip saving data for that frame.
                    if keypoints is not None:
                        frames_with_keypoints += 1
                        row = self._frame_to_row(
                            frame_index,
                            keypoints,
                            L_tube,
                            R_tube,
                            video_path.name,
                            start_time,
                        )
                        rows.append(row)
            except KeyError:
                names = getattr(bee_model, "names", None)
                name_keys = sorted(int(k) for k in names) if isinstance(names, dict) else []
                logger.exception(
                    "KeyError while iterating YOLO stream for video=%s ultralytics=%s names_keys=%s",
                    video_path,
                    ultralytics.__version__,
                    name_keys,
                )
                raise

            t_stream_done = perf_counter()

            if save_markup_video:
                logger.info("Saving marked-up video for %s to %s", video_path, markup_dir)
                marked_up_video_path = markup_dir / (video_path.stem + ".avi")
                self.save_recording(
                    stream_index=CA_MARKED_UP_VID_STREAM_INDEX,
                    temporary_file=marked_up_video_path,
                    start_time=start_time,
                    end_time=end_time,
                    override_sampling=api.OVERRIDE.SAVE,
                )

            t_markup_done = perf_counter()
            output_df = pd.DataFrame(rows)
            t_df_done = perf_counter()

            logger.info(
                (
                    "Pose timings: video=%s model_load=%.3fs tube_detect=%.3fs predict_setup=%.3fs "
                    "stream_iter=%.3fs dataframe=%.3fs markup_save=%.3fs total=%.3fs "
                    "frames=%d rows=%d rows_per_frame=%.3f"
                ),
                video_path,
                t_model_done - t_model_start,
                t_tubes_done - t_tubes_start,
                t_predict_setup_done - t_predict_start,
                t_stream_done - t_predict_setup_done,
                t_df_done - t_markup_done,
                t_markup_done - t_stream_done,
                t_df_done - t0,
                frames_seen,
                frames_with_keypoints,
                (frames_with_keypoints / frames_seen) if frames_seen else 0.0,
            )

            return output_df
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
