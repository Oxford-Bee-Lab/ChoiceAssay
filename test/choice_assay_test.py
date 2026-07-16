import logging
from dataclasses import replace
from pathlib import Path
from time import sleep

import pandas as pd
import pytest
from expidite_rpi import DeviceCfg, DPtree, RpiCore, Stream, api
from expidite_rpi import configuration as root_cfg
from expidite_rpi.sensors.sensor_rpicam_vid import (
    RPICAM_REVIEW_MODE_STREAM,
    RPICAM_STREAM,
    RPICAM_STREAM_INDEX,
    RpicamSensor,
    RpicamSensorCfg,
)
from expidite_rpi.utils.rpi_emulator import RpiEmulator, RpiTestRecording

from choice_assay.rpi.choice_assay_pose_processor import (
    CA_KEYPOINT_NAMES,
    CA_MARKED_UP_VID_DATA_TYPE_ID,
    CA_MARKED_UP_VID_STREAM_INDEX,
    CA_XY_DATA_TYPE_ID,
    CA_XY_STREAM_INDEX,
    DEFAULT_CHOICE_ASSAY_POSE_PROCESSOR_CFG,
    ChoiceAssayPoseProcessor,
)
from choice_assay.rpi.choice_assay_trapcam import (
    CA_IMAGES_DATA_TYPE_ID,
    CA_IMAGES_STREAM_INDEX,
    CA_MASK_DATA_TYPE_ID,
    CA_MASK_STREAM_INDEX,
    CA_VIDEO_DATA_TYPE_ID,
    CA_VIDEO_STREAM_INDEX,
    ChoiceAssayTrapcamParams,
    ChoiceAssayTrapcamProcessor,
)

logger = root_cfg.setup_logger("choice_assay", level=logging.DEBUG)

root_cfg.ST_MODE = root_cfg.SOFTWARE_TEST_MODE.TESTING

TRAPCAM_PROCESSOR_CFG = ChoiceAssayTrapcamParams(
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
            sample_probability="1.0",
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

outputs = [
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
]
CHOICE_ASSAY_POSE_PROCESSOR_CFG = replace(DEFAULT_CHOICE_ASSAY_POSE_PROCESSOR_CFG, outputs=outputs)


def create_choice_assay_device() -> list[DPtree]:
    """Create a dual-arena choice assay camera device."""
    # Define the video sensor
    sampling_stream = replace(RPICAM_STREAM, sample_probability=0.02)
    cfg = RpicamSensorCfg(
        sensor_type=api.SENSOR_TYPE.CAMERA,
        sensor_index=0,
        sensor_model="PiCameraModule3",
        description="Video sensor that uses rpicam-vid",
        outputs=[sampling_stream, RPICAM_REVIEW_MODE_STREAM],
        rpicam_cmd=(
            "rpicam-vid --framerate 5 --width 800 --height 608 -o FILENAME -t 180000 --exposure sport"
        ),
    )
    my_sensor = RpicamSensor(cfg)

    # Define the Trapcam dataprocessor
    trapcam_dp = ChoiceAssayTrapcamProcessor(
        TRAPCAM_PROCESSOR_CFG,
        my_sensor.sensor_index,
    )

    # Define the ML dataprocessor
    pose_dp = ChoiceAssayPoseProcessor(
        CHOICE_ASSAY_POSE_PROCESSOR_CFG,
        my_sensor.sensor_index,
    )

    my_tree = DPtree(my_sensor)
    my_tree.connect((my_sensor, RPICAM_STREAM_INDEX), trapcam_dp)
    my_tree.connect((trapcam_dp, CA_VIDEO_STREAM_INDEX), pose_dp)

    return [my_tree]


class Test_choice_assay:
    @pytest.fixture
    def inventory(self) -> list[DeviceCfg]:
        return [
            DeviceCfg(
                name="Alex",
                device_id="d01111111111",  # This is the DUMMY MAC address for windows
                notes="Testing choice assay device",
                dp_trees_create_method=create_choice_assay_device,
            ),
        ]

    @pytest.mark.parametrize(
        "test_input",
        [
            {
                "src_vid": "V3_RPICAM_d83addbca346_00_00_20260312T175714542_20260312T180015121.mp4",
            },
            {
                "src_vid": "V3_CAVIDEO_d01111111111_00_00_20260313T195514678_20260313T195604277.mp4",
            },
        ],
    )
    @pytest.mark.unittest
    def test_choice_assay(
        self,
        test_input: dict[str, str],
        rpi_emulator: RpiEmulator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        src_vid = test_input["src_vid"]
        captured_pose_dataframes: list[pd.DataFrame] = []

        original_save_data = ChoiceAssayPoseProcessor.save_data

        def capture_save_data(self, stream_index: int, sensor_data: pd.DataFrame) -> None:
            if stream_index == CA_XY_STREAM_INDEX:
                captured_pose_dataframes.append(sensor_data.copy())
            original_save_data(self, stream_index, sensor_data)

        monkeypatch.setattr(ChoiceAssayPoseProcessor, "save_data", capture_save_data)

        # Set the file to be fed into the choice assay device
        rpi_emulator.set_recordings(
            [
                RpiTestRecording(
                    cmd_prefix="rpicam-vid",
                    recordings=[Path(__file__).parent / "resources" / src_vid],
                ),
            ]
        )

        # Limit the RpiCore to 1 recording so we can easily validate the results
        rpi_emulator.set_recording_cap(1, type_id="RPICAM")

        # Configure RpiCore with the choice assay device
        sc = RpiCore(rpi_emulator.inventory)
        sc.start()
        while not (rpi_emulator.recordings_cap_hit(type_id="RPICAM")):
            # Wait for the recordings to be fed in....
            sleep(1)
        while rpi_emulator.recordings_still_to_process():
            # Wait for the recordings to be processed....
            sleep(1)
        sc.stop()

        # We should have identified bees in the video and save the info to the FLOWERCAM datastream
        rpi_emulator.assert_records("expidite-fair", {"V3_*": rpi_emulator.ONE_OR_MORE})
        rpi_emulator.assert_records(
            "expidite-system-records", {"V3_HEART*": 1, "V3_SCORE*": 1, "V3_SCORP*": 1}
        )
        rpi_emulator.assert_records(
            "expidite-journals",
            {"V3_CAPOSE_*": 1},
        )
        rpi_emulator.assert_records(
            "expidite-choiceassay-trapcam",
            {
                "V3_CAVIDEO_*": rpi_emulator.ONE_OR_MORE,
            },
        )

        assert captured_pose_dataframes, "Expected CAPOSE data to be saved"

        pose_df = pd.concat(captured_pose_dataframes, ignore_index=True)
        assert not pose_df.empty, "Expected CAPOSE dataframe to contain at least one row"

        bee_columns = [f"{name}_{suffix}" for name in CA_KEYPOINT_NAMES[:7] for suffix in ["x", "y", "conf"]]
        tube_columns = [f"{name}_{suffix}" for name in CA_KEYPOINT_NAMES[7:] for suffix in ["x", "y", "conf"]]

        assert set(bee_columns).issubset(pose_df.columns)
        assert set(tube_columns).issubset(pose_df.columns)
        assert pose_df[bee_columns].notna().any(axis=1).any(), "Expected at least one row with bee keypoints"
        assert pose_df[tube_columns].notna().any(axis=1).any(), "Expected at least one tube field value"
