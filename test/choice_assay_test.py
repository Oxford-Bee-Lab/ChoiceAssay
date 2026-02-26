import logging
from pathlib import Path
from time import sleep

import pytest
from expidite_rpi import DeviceCfg, RpiCore
from expidite_rpi import configuration as root_cfg
from expidite_rpi.utils.rpi_emulator import RpiEmulator, RpiTestRecording

from choice_assay import my_fleet_config

logger = root_cfg.setup_logger("choice_assay", level=logging.DEBUG)

root_cfg.ST_MODE = root_cfg.SOFTWARE_TEST_MODE.TESTING


class Test_choice_assay:
    @pytest.fixture
    def inventory(self) -> list[DeviceCfg]:
        return [
            DeviceCfg(
                name="Alex",
                device_id="d01111111111",  # This is the DUMMY MAC address for windows
                notes="Testing choice assay device",
                dp_trees_create_method=my_fleet_config.create_choice_assay_device,
            ),
        ]

    @pytest.mark.parametrize(
        "test_input",
        [
            {
                "src_vid": "left_04-12-2025_21-32-15.mp4",
            },
            {
                "src_vid": "left_10_12-2025_17-07-24.mp4",
            },
        ],
    )
    @pytest.mark.unittest
    def test_choice_assay(self, test_input: dict[str, str], rpi_emulator: RpiEmulator) -> None:
        src_vid = test_input["src_vid"]

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
        rpi_emulator.set_recording_cap(1, type_id="CHOICEASSAYVID")

        # Configure RpiCore with the choice assay device
        sc = RpiCore(rpi_emulator.inventory)
        sc.start()
        while not (rpi_emulator.recordings_cap_hit(type_id="CHOICEASSAYVID")):
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
                "V3_CA_RIGHT_VIDEO_DATA_TYPE_ID_*": rpi_emulator.ONE_OR_MORE,
                "V3_CA_LEFT_VIDEO_DATA_TYPE_ID_*": rpi_emulator.ONE_OR_MORE,
            },
        )
