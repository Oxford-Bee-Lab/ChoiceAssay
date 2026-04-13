import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from expidite_rpi import Stream, api
from expidite_rpi.core import configuration as root_cfg

# Add src to path to import directly from the file to avoid dependency issues
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import directly from the trapcam module file to avoid YOLO dependency chain
from choice_assay.rpi.choice_assay_trapcam import (
    CA_MASK_DATA_TYPE_ID,
    CA_MASK_STREAM_INDEX,
    CA_VIDEO_DATA_TYPE_ID,
    CA_VIDEO_STREAM_INDEX,
    ChoiceAssayTrapcamParams,
    ChoiceAssayTrapcamProcessor,
)

logger = root_cfg.setup_logger("choice_assay_trapcam_test", level=logging.DEBUG)

# Test configuration with mask videos enabled
TRAPCAM_TEST_CFG = ChoiceAssayTrapcamParams(
    description="Test trapcam processor with mask video enabled",
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
            sample_probability="1.0",
        ),
    ],
    # Enable mask video generation for testing
    save_mask_video=True,
)


class Test_ChoiceAssayTrapcamProcessor:
    """Test class for ChoiceAssayTrapcamProcessor with parameterised video inputs."""

    @pytest.mark.parametrize(
        "video_filename",
        [
            "downsampled.mp4",
            "V3_RPICAM_d83addbca346_00_00_20260312T175714542_20260312T180015121.mp4",
            "V3_RPICAM_d83addbca346_00_00_20260313T141126803_20260313T141427379.mp4",
        ],
    )
    @pytest.mark.trapcam
    @pytest.mark.unittest
    def test_trapcam_processor_generates_mask_and_trapcam_videos(self, video_filename: str) -> None:
        """Test that ChoiceAssayTrapcamProcessor generates both mask and trapcam videos."""
        # Get test video path
        test_resources_path = Path(__file__).parent / "resources"
        video_path = test_resources_path / video_filename

        # Ensure the test video exists
        assert video_path.exists(), f"Test video not found: {video_path}"

        # Create processor instance
        processor = ChoiceAssayTrapcamProcessor(TRAPCAM_TEST_CFG, sensor_index=0)

        # Create output directory in TMP_DIR for this test
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        test_output_dir = Path(root_cfg.TMP_DIR) / f"trapcam_test_{timestamp}_{video_filename}"
        test_output_dir.mkdir(parents=True, exist_ok=True)

        # Track saved files for verification
        saved_files = []

        def save_video_to_tmp_dir(stream_index: int, temporary_file: Path, start_time, end_time):
            """Save video files to TMP_DIR so they can be accessed after processing."""
            # Determine file type based on stream index
            if stream_index == CA_MASK_STREAM_INDEX:
                file_type = "mask"
            elif stream_index == CA_VIDEO_STREAM_INDEX:
                file_type = "trapcam"
            else:
                file_type = f"stream_{stream_index}"

            # Create a descriptive filename
            output_filename = (
                f"{file_type}_{start_time.strftime('%H%M%S')}_to_{end_time.strftime('%H%M%S')}.mp4"
            )
            output_path = test_output_dir / output_filename

            # Copy the temporary file to our output directory
            if temporary_file.exists():
                shutil.copy2(temporary_file, output_path)
                saved_files.append(
                    {
                        "stream_index": stream_index,
                        "file_path": output_path,
                        "file_type": file_type,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                )
                logger.info(f"Saved {file_type} video to: {output_path}")
            else:
                logger.warning(f"Temporary file does not exist: {temporary_file}")

        processor.save_recording = save_video_to_tmp_dir

        # Process the video
        logger.info(f"Processing test video: {video_filename}")
        logger.info(f"Output videos will be saved to: {test_output_dir}")
        processor.process_data([video_path])

        # Verify that videos were saved for both mask and trapcam
        mask_files = [f for f in saved_files if f["stream_index"] == CA_MASK_STREAM_INDEX]
        trapcam_files = [f for f in saved_files if f["stream_index"] == CA_VIDEO_STREAM_INDEX]

        # Assert mask video was generated and saved
        assert len(mask_files) > 0, f"No mask videos were saved for {video_filename}"
        logger.info(f"[OK] Mask video saved for {video_filename}: {len(mask_files)} file(s)")

        # Assert trapcam video was generated and saved (motion-triggered clips)
        assert len(trapcam_files) > 0, f"No trapcam videos were saved for {video_filename}"
        logger.info(f"[OK] Trapcam video(s) saved for {video_filename}: {len(trapcam_files)} file(s)")

        # Verify saved files actually exist and have content
        for saved_file in saved_files:
            file_path = saved_file["file_path"]
            assert file_path.exists(), f"Saved file does not exist: {file_path}"
            assert file_path.stat().st_size > 0, f"Saved file is empty: {file_path}"
            logger.info(
                f"[OK] Verified saved {saved_file['file_type']} video: {file_path} "
                f"({file_path.stat().st_size} bytes)"
            )

        logger.info(
            f"[OK] Test passed for {video_filename} - both mask and trapcam videos saved successfully"
        )
        logger.info(f"[OK] Videos available in: {test_output_dir}")

    @pytest.mark.trapcam
    @pytest.mark.unittest
    def test_trapcam_processor_no_mask_when_disabled(self) -> None:
        """Test that no mask video is generated when save_mask_video=False."""
        # Create configuration with mask video disabled
        cfg_no_mask = ChoiceAssayTrapcamParams(
            description="Test trapcam processor without mask video",
            outputs=TRAPCAM_TEST_CFG.outputs,
            save_mask_video=False,  # Disable mask video
            min_motion_pixels=10,
            min_motion_run_frames=1,
        )

        # Get test video path
        test_resources_path = Path(__file__).parent / "resources"
        video_path = (
            test_resources_path / "V3_RPICAM_d83addbca346_00_00_20260312T175714542_20260312T180015121.mp4"
        )

        # Create processor instance
        processor = ChoiceAssayTrapcamProcessor(cfg_no_mask, sensor_index=0)

        # Create output directory in TMP_DIR for this test
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        test_output_dir = Path(root_cfg.TMP_DIR) / f"trapcam_test_no_mask_{timestamp}"
        test_output_dir.mkdir(parents=True, exist_ok=True)

        # Track saved files for verification
        saved_files = []

        def save_video_to_tmp_dir_no_mask(stream_index: int, temporary_file: Path, start_time, end_time):
            """Save video files to TMP_DIR for no-mask test."""
            # Determine file type based on stream index
            if stream_index == CA_MASK_STREAM_INDEX:
                file_type = "mask"
            elif stream_index == CA_VIDEO_STREAM_INDEX:
                file_type = "trapcam"
            else:
                file_type = f"stream_{stream_index}"

            # Create a descriptive filename
            output_filename = (
                f"{file_type}_{start_time.strftime('%H%M%S')}_to_{end_time.strftime('%H%M%S')}.mp4"
            )
            output_path = test_output_dir / output_filename

            # Copy the temporary file to our output directory
            if temporary_file.exists():
                shutil.copy2(temporary_file, output_path)
                saved_files.append(
                    {"stream_index": stream_index, "file_path": output_path, "file_type": file_type}
                )
                logger.info(f"Saved {file_type} video to: {output_path}")

        processor.save_recording = save_video_to_tmp_dir_no_mask

        # Process the video
        logger.info("Processing video with save_mask_video=False")
        logger.info(f"Output videos will be saved to: {test_output_dir}")
        processor.process_data([video_path])

        # Verify that no mask video was generated
        mask_files = [f for f in saved_files if f["stream_index"] == CA_MASK_STREAM_INDEX]
        assert len(mask_files) == 0, "Mask video should not be generated when save_mask_video=False"

        logger.info("[OK] No mask video generated when save_mask_video=False")
        logger.info(f"[OK] Test output directory: {test_output_dir}")
