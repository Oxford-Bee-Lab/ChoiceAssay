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

CA_VIDEO_DATA_TYPE_ID = "CAVIDEO"
CA_VIDEO_STREAM_INDEX: int = 0

CA_MASK_DATA_TYPE_ID = "CAMASK"
CA_MASK_STREAM_INDEX: int = 1

CA_IMAGES_DATA_TYPE_ID = "CAIMAGES"
CA_IMAGES_STREAM_INDEX: int = 2


@dataclass
class ChoiceAssayTrapcamParams(DataProcessorCfg):
    min_motion_pixels: int = 50
    min_motion_run_frames: int = 2  # Minimum consecutive frames of motion to count as a valid period
    pre_activity_frames: int = 10  # Include ~2 seconds before detected motion (at 5 fps)
    post_activity_frames: int = 10  # Include ~2 seconds after detected motion (at 5 fps)
    save_mask_video: bool = False  # When True, write a full-length foreground mask video alongside each input
    # Background subtraction tuning
    bg_history: int = 400
    bg_var_threshold: float = 100.0


DEFAULT_CHOICE_ASSAY_TRAPCAM_PROCESSOR_CFG = ChoiceAssayTrapcamParams(
    description="Background-subtraction trapcam processor for motion-triggered full-frame clips",
    outputs=[
        Stream(
            description="Trapcam motion-triggered full-frame video",
            type_id=CA_VIDEO_DATA_TYPE_ID,
            index=CA_VIDEO_STREAM_INDEX,
            format=api.FORMAT.MP4,
            cloud_container="expidite-choiceassay-trapcam",
            sample_probability="1.0",
        ),
        Stream(
            description="Trapcam motion mask",
            type_id=CA_MASK_DATA_TYPE_ID,
            index=CA_MASK_STREAM_INDEX,
            format=api.FORMAT.MP4,
            cloud_container="expidite-choiceassay-mask",
            sample_probability="0.02",
        ),
        Stream(
            description="Sample images from the first frame after a restart",
            type_id=CA_IMAGES_DATA_TYPE_ID,
            index=CA_IMAGES_STREAM_INDEX,
            format=api.FORMAT.JPG,
            cloud_container="expidite-choiceassay-images",
            sample_probability="1.0",
        ),
    ],
)


class ChoiceAssayTrapcamProcessor(DataProcessor):
    def __init__(self, config: DataProcessorCfg, sensor_index: int) -> None:
        super().__init__(config, sensor_index)
        self.params: ChoiceAssayTrapcamParams = config  # type: ignore

        # Initialise background subtractor for motion detection once here.
        # This allows it to maintain state across multiple video files.
        # detectShadows=False: avoids misclassifying the body of a slow/stationary object as shadow
        # (shadow pixels are labelled 127 and would be stripped by our >200 threshold anyway, but
        # shadow detection causes interior pixels to be suppressed before they reach the threshold).
        self.subtractor = cv2.createBackgroundSubtractorKNN(
            history=self.params.bg_history,
            dist2Threshold=self.params.bg_var_threshold,
            detectShadows=False,
        )
        # To track whether we've saved an image from the first frame after a restart
        self.first_frame_image_saved = False

    def _motion_score(
        self,
        fgmask: np.ndarray,
    ) -> int:
        """Count non-zero pixels across the full frame."""
        return int(cv2.countNonZero(fgmask))

    def _build_writer(
        self, fps: float, frame_shape: tuple[int, int], output_path: str | Path
    ) -> cv2.VideoWriter:
        width, height = frame_shape
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    def _extract_motion_data(self, video_path: Path) -> tuple[pd.DataFrame, float, Path | None]:
        """Pass 1: scan video and record frame-wise motion metrics into a DataFrame.

        If params.save_mask_video is True, also writes a full-length foreground mask video
        (same duration as the input) and returns its temporary file path as the third element.
        """
        params = self.params

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            logger.error(f"Could not open video for trapcam processing: {video_path}")
            return pd.DataFrame(), 0.0, None

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        motion_rows: list[dict] = []
        frame_index = 0
        mask_writer: cv2.VideoWriter | None = None
        mask_output_path: Path | None = None

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                # Shrink the image so we can process it faster; we don't need high-res for motion detection.
                frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fgmask = self.subtractor.apply(gray)
                motion_score = self._motion_score(fgmask)
                # Ignore the first 4 frames which are often very noisy as the background model initializes
                if frame_index >= 4:
                    motion_rows.append({"frame_index": frame_index, "motion_score": motion_score})

                if params.save_mask_video:
                    if mask_writer is None:
                        h, w = fgmask.shape[:2]
                        mask_output_path = Path(file_naming.get_temporary_filename(api.FORMAT.MP4))
                        mask_writer = self._build_writer(fps, (w, h), mask_output_path)
                    mask_frame = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
                    motion_detected = motion_score >= params.min_motion_pixels
                    label = f"Frame {frame_index}  Motion: {'YES' if motion_detected else 'no '} "
                    label += f"Score: {motion_score}"
                    text_color = (255, 220, 120) if motion_detected else (0, 255, 0)
                    cv2.putText(
                        mask_frame, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA
                    )
                    mask_writer.write(mask_frame)

                frame_index += 1
        finally:
            if mask_writer is not None:
                mask_writer.release()
            capture.release()

        return pd.DataFrame(motion_rows), fps, mask_output_path

    def _filter_motion_into_clean_periods(self, motion_df: pd.DataFrame) -> list[dict]:
        """Pass 2: convert raw frame motion detections into robust motion periods."""
        if motion_df.empty:
            return []

        params = self.params
        motion_df_cleaned = motion_df[["frame_index", "motion_score"]].copy()
        motion_df_cleaned = motion_df_cleaned.sort_values("frame_index").reset_index(drop=True)
        raw_active = motion_df_cleaned["motion_score"] >= params.min_motion_pixels

        # 1) Remove short active bursts (likely noise) using run-length filtering.
        active_runs = raw_active.ne(raw_active.shift()).cumsum()
        run_lengths = motion_df_cleaned.groupby(active_runs, sort=False)["frame_index"].transform("size")
        stable_active = raw_active & (run_lengths >= params.min_motion_run_frames)

        # 2) Convert stable activity labels into contiguous periods.
        active_mask = stable_active
        if not active_mask.any():
            return []

        active_df = motion_df_cleaned.loc[active_mask, ["frame_index"]].copy()
        period_id = active_df["frame_index"].diff().ne(1).cumsum()

        periods_df = (
            active_df.groupby(period_id, sort=False)
            .agg(start_frame=("frame_index", "min"), end_frame=("frame_index", "max"))
            .reset_index(drop=True)
        )

        # 3) Expand each period to include configurable lead-in and tail time.
        total_frames = int(motion_df_cleaned["frame_index"].max())
        periods_df["start_frame"] = (periods_df["start_frame"] - params.pre_activity_frames).clip(lower=0)
        periods_df["end_frame"] = (periods_df["end_frame"] + params.post_activity_frames).clip(
            upper=total_frames
        )

        # 4) Merge any periods that overlap (or touch) after expansion.
        merged_periods: list[dict[str, int]] = []
        for period in periods_df.to_dict("records"):
            start_frame = int(period["start_frame"])
            end_frame = int(period["end_frame"])

            if not merged_periods:
                merged_periods.append({"start_frame": start_frame, "end_frame": end_frame})
                continue

            last_period = merged_periods[-1]
            if start_frame <= (last_period["end_frame"] + 1):
                last_period["end_frame"] = max(last_period["end_frame"], end_frame)
            else:
                merged_periods.append({"start_frame": start_frame, "end_frame": end_frame})

        return merged_periods

    def _write_period_clips(
        self,
        video_path: Path,
        periods: list[dict],
        fps: float,
    ) -> None:
        """Pass 3: write full-frame clips from filtered motion periods."""
        if not periods:
            return

        # Get the start timestamp from the video_path filename to calculate clip timestamps later
        parts = file_naming.parse_record_filename(video_path.name)
        start_timestamp = parts.get(api.RECORD_ID.TIMESTAMP.value, api.utc_now())

        for period in periods:
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

                    if frame.ndim != 3 or frame.size == 0:
                        current_frame += 1
                        continue

                    if writer is None:
                        frame_height, frame_width = frame.shape[:2]
                        writer = self._build_writer(fps, (frame_width, frame_height), output_file)

                    writer.write(frame)
                    current_frame += 1

                if writer is not None:
                    writer.release()
                    writer = None
                    clip_start = start_timestamp + timedelta(seconds=start_frame / fps)
                    clip_end = start_timestamp + timedelta(seconds=end_frame / fps)
                    self.save_recording(
                        stream_index=CA_VIDEO_STREAM_INDEX,
                        temporary_file=Path(output_file),
                        start_time=clip_start,
                        end_time=clip_end,
                    )
                    logger.info(
                        f"Trapcam wrote clip from frames {start_frame}-{end_frame} ({video_path.name})"
                    )
            finally:
                if writer is not None:
                    writer.release()
                capture.release()

    def _save_first_frame_image(self, video_path: Path) -> None:
        """Save an image from the first frame of the first video we process after a restart."""
        try:
            capture = cv2.VideoCapture(str(video_path))
            if capture.isOpened():
                ok, frame = capture.read()
                if ok and frame.ndim == 3 and frame.size > 0:
                    image_output_path = file_naming.get_temporary_filename(api.FORMAT.JPG)
                    cv2.imwrite(str(image_output_path), frame)
                    self.save_recording(
                        stream_index=CA_IMAGES_STREAM_INDEX,
                        temporary_file=image_output_path,
                        start_time=api.utc_now(),
                        end_time=api.utc_now(),
                    )
                    logger.info(f"Trapcam saved first-frame image from {video_path.name}")
                    self.first_frame_image_saved = True
                capture.release()
        except Exception:
            logger.exception(f"Failed to save first-frame image from {video_path.name}")

    def _process_video_file(self, video_path: Path) -> None:
        """Run two-pass trapcam analysis then write full-frame clips from filtered periods."""
        # Save an image from the first frame of the first video we process after a restart
        if not self.first_frame_image_saved:
            self._save_first_frame_image(video_path)

        motion_df, fps, mask_path = self._extract_motion_data(video_path)
        if motion_df.empty:
            logger.info(f"No motion detected in: {video_path.name}")
            return

        # Save the full-length mask video if one was produced during motion extraction.
        if mask_path is not None:
            parts = file_naming.parse_record_filename(video_path.name)
            start_timestamp = parts.get(api.RECORD_ID.TIMESTAMP.value, api.utc_now())
            end_timestamp = start_timestamp + timedelta(seconds=len(motion_df) / fps)
            self.save_recording(
                stream_index=CA_MASK_STREAM_INDEX,
                temporary_file=mask_path,
                start_time=start_timestamp,
                end_time=end_timestamp,
            )
            logger.info(f"Trapcam saved full mask video for {video_path.name}")

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
