from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from expidite_rpi.core import api, file_naming
from expidite_rpi.core import configuration as root_cfg
from expidite_rpi.core.dp import DataProcessor
from expidite_rpi.core.dp_config_objects import DataProcessorCfg, Stream

logger = root_cfg.setup_logger("choice_assay")

CA_LEFT_VIDEO_DATA_TYPE_ID = "CAVIDEOLEFT"
CA_RIGHT_VIDEO_DATA_TYPE_ID = "CAVIDEORIGHT"
CA_LEFT_VIDEO_STREAM_INDEX: int = 0
CA_RIGHT_VIDEO_STREAM_INDEX: int = 1


@dataclass
class ChoiceAssayTrapcamParams:
    min_motion_pixels: int = 1800
    side_dominance_ratio: float = 1.5
    min_motion_run_frames: int = 3  # On assumption 5 fps
    grace_frames: int = 10  # Bridge a 2 second gap in motion if the same side is active before and after
    blur_kernel: tuple[int, int] = (5, 5)
    left_detection_roi: tuple[int, int, int, int] = (210, 373, 560, 578)
    right_detection_roi: tuple[int, int, int, int] = (1170, 373, 1520, 578)
    left_recording_roi: tuple[int, int, int, int] = (164, 167, 601, 604)
    right_recording_roi: tuple[int, int, int, int] = (1124, 167, 1561, 604)


DEFAULT_CHOICE_ASSAY_TRAPCAM_PROCESSOR_CFG = DataProcessorCfg(
    description="Background-subtraction trapcam processor for left/right arena sub-videos",
    outputs=[
        Stream(
            description="Trapcam motion-triggered left arena video",
            type_id=CA_LEFT_VIDEO_DATA_TYPE_ID,
            index=CA_LEFT_VIDEO_STREAM_INDEX,
            format=api.FORMAT.MP4,
            cloud_container="expidite-choiceassay-trapcam",
            sample_probability="1.0",
        ),
        Stream(
            description="Trapcam motion-triggered right arena video",
            type_id=CA_RIGHT_VIDEO_DATA_TYPE_ID,
            index=CA_RIGHT_VIDEO_STREAM_INDEX,
            format=api.FORMAT.MP4,
            cloud_container="expidite-choiceassay-trapcam",
            sample_probability="1.0",
        ),
    ],
)


class ChoiceAssayTrapcamProcessor(DataProcessor):
    def __init__(self, config: DataProcessorCfg, sensor_index: int) -> None:
        super().__init__(config, sensor_index)
        self.params = ChoiceAssayTrapcamParams()

        # Initialise background subtractor for motion detection once here
        # This allows it to maintain state across multiple video files.
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=True)

    def _motion_score(
        self,
        fgmask: np.ndarray,
        roi: tuple[int, int, int, int],
    ) -> int:
        """Count non-zero pixels within the ROI."""
        x1, y1, x2, y2 = roi
        roi_mask = fgmask[y1:y2, x1:x2]
        return int(cv2.countNonZero(roi_mask))

    def _detect_active_side(self, left_score: int, right_score: int) -> str | None:
        """Determine active side based on motion scores and configured thresholds/ratios."""
        params = self.params
        left_active = left_score >= params.min_motion_pixels
        right_active = right_score >= params.min_motion_pixels

        if left_active and not right_active:
            return "left"
        if right_active and not left_active:
            return "right"
        if not left_active and not right_active:
            return None

        # If both sides are active, apply dominance ratio to determine if one side is clearly dominant
        # If not, this is likely a change in lighting
        if left_score >= int(right_score * params.side_dominance_ratio):
            return "left"
        if right_score >= int(left_score * params.side_dominance_ratio):
            return "right"
        return None

    def _get_stream_index(self, side: str) -> int:
        return CA_LEFT_VIDEO_STREAM_INDEX if side == "left" else CA_RIGHT_VIDEO_STREAM_INDEX

    def _get_recording_roi(self, side: str) -> tuple[int, int, int, int]:
        return self.params.left_recording_roi if side == "left" else self.params.right_recording_roi

    def _build_writer(
        self, fps: float, frame_shape: tuple[int, int], output_path: str | Path
    ) -> cv2.VideoWriter:
        width, height = frame_shape
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    def _extract_motion_data(self, video_path: Path) -> tuple[pd.DataFrame, float]:
        """Pass 1: scan video and record frame-wise motion metrics into a DataFrame."""
        params = self.params

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            logger.error(f"Could not open video for trapcam processing: {video_path}")
            return pd.DataFrame(), 0.0

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        motion_rows: list[dict] = []
        frame_index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, params.blur_kernel, 0)
                fgmask = self.subtractor.apply(gray)
                _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
                fgmask = cv2.medianBlur(fgmask, 5)

                left_score = self._motion_score(fgmask, params.left_detection_roi)
                right_score = self._motion_score(fgmask, params.right_detection_roi)
                active_side = self._detect_active_side(left_score, right_score)

                motion_rows.append(
                    {
                        "frame_index": frame_index,
                        "left_score": left_score,
                        "right_score": right_score,
                        "raw_active_side": active_side,
                    }
                )
                frame_index += 1
        finally:
            capture.release()

        return pd.DataFrame(motion_rows), fps

    def _filter_motion_into_clean_periods(self, motion_df: pd.DataFrame) -> list[dict]:
        """Pass 2: convert raw frame-side detections into robust side-specific motion periods."""
        if motion_df.empty:
            return []

        params = self.params
        motion_df_cleaned = motion_df[["frame_index", "raw_active_side"]].copy()
        motion_df_cleaned = motion_df_cleaned.sort_values("frame_index").reset_index(drop=True)

        # 1) Remove short side bursts (likely noise) using run-length filtering.
        side_runs = (
            motion_df_cleaned["raw_active_side"].ne(motion_df_cleaned["raw_active_side"].shift()).cumsum()
        )
        run_lengths = motion_df_cleaned.groupby(side_runs, sort=False)["raw_active_side"].transform("size")
        stable_side = motion_df_cleaned["raw_active_side"].where(run_lengths >= params.min_motion_run_frames)

        # 2) Bridge short gaps (None) up to grace_frames only when both sides agree.
        forward_filled = stable_side.ffill(limit=params.grace_frames)
        backward_filled = stable_side.bfill(limit=params.grace_frames)
        clean_side = forward_filled.where(forward_filled == backward_filled)

        # 3) Convert cleaned side labels into contiguous periods.
        active_mask = clean_side.notna()
        if not active_mask.any():
            return []

        active_df = motion_df_cleaned.loc[active_mask, ["frame_index"]].copy()
        active_df["side"] = clean_side.loc[active_mask]
        period_id = active_df["side"].ne(active_df["side"].shift()).cumsum()

        periods_df = (
            active_df.groupby(period_id, sort=False)
            .agg(side=("side", "first"), start_frame=("frame_index", "min"), end_frame=("frame_index", "max"))
            .reset_index(drop=True)
        )

        return periods_df.to_dict("records")

    def _write_period_clips(
        self,
        video_path: Path,
        periods: list[dict],
        fps: float,
    ) -> None:
        """Pass 3: write arena-targeted clips from filtered motion periods."""
        if not periods:
            return

        # Get the start timestamp from the video_path filename to calculate clip timestamps later
        parts = file_naming.parse_record_filename(video_path.name)
        start_timestamp = parts.get(api.RECORD_ID.TIMESTAMP.value, api.utc_now())

        for period in periods:
            side = period["side"]
            start_frame = int(period["start_frame"])
            end_frame = int(period["end_frame"])
            writer: cv2.VideoWriter | None = None

            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                logger.error(f"Could not open video for clip writing: {video_path}")
                return

            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                output_file = file_naming.get_temporary_filename(api.FORMAT.MP4)

                current_frame = start_frame
                while current_frame <= end_frame:
                    ok, frame = capture.read()
                    if not ok:
                        break

                    roi = self._get_recording_roi(side)
                    x1, y1, x2, y2 = roi
                    roi_frame = frame[y1:y2, x1:x2]
                    if roi_frame.ndim != 3 or roi_frame.size == 0:
                        current_frame += 1
                        continue

                    if writer is None:
                        roi_height, roi_width = roi_frame.shape[:2]
                        writer = self._build_writer(fps, (roi_width, roi_height), output_file)

                    writer.write(roi_frame)
                    current_frame += 1

                if writer is not None:
                    writer.release()
                    clip_start = start_timestamp + timedelta(seconds=start_frame / fps)
                    clip_end = start_timestamp + timedelta(seconds=end_frame / fps)
                    self.save_recording(
                        stream_index=self._get_stream_index(side),
                        temporary_file=Path(output_file),
                        start_time=clip_start,
                        end_time=clip_end,
                    )
                    logger.info(
                        f"Trapcam wrote {side} clip from frames {start_frame}-{end_frame} ({video_path.name})"
                    )
            finally:
                if writer is not None:
                    writer.release()
                capture.release()

    def _process_video_file(self, video_path: Path) -> None:
        """Run two-pass trapcam analysis then write side-targeted clips from filtered periods."""
        motion_df, fps = self._extract_motion_data(video_path)
        if motion_df.empty:
            logger.info(f"No motion detected in: {video_path.name}")
            return

        filtered_periods = self._filter_motion_into_clean_periods(motion_df)
        if not filtered_periods:
            logger.info(f"No meaningful motion periods detected in {video_path.name}")
            return

        self._write_period_clips(video_path, filtered_periods, fps)

    # Main entry point for Expidite to call with new video files to process
    def process_data(self, input_data: pd.DataFrame | list[Path]) -> None:
        """This is the function called by Expidite to process new data.
        It receives a list of file paths to new video files that have been recorded by the sensor.
        """
        if input_data is None:
            return

        # This function will only ever receive a list of file, never a DataFrame
        assert not isinstance(input_data, pd.DataFrame), "Trapcam process_data should not receive a DataFrame"
        video_files = [Path(f) for f in input_data]

        for video_file in video_files:
            try:
                if not video_file.exists():
                    logger.warning(f"Trapcam input file does not exist: {video_file}")
                    continue
                self._process_video_file(video_file)
            except Exception:
                logger.exception("Trapcam processing failed for %s", video_file)
