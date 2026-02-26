from expidite_rpi.core import api
from expidite_rpi.core.device_config_objects import DeviceCfg, WifiClient
from expidite_rpi.core.dp_tree import DPtree
from expidite_rpi.sensors.sensor_rpicam_vid import (
    RPICAM_REVIEW_MODE_STREAM,
    RPICAM_STREAM,
    RPICAM_STREAM_INDEX,
    RpicamSensor,
    RpicamSensorCfg,
)

from choice_assay.rpi.choice_assay_pose_processor import (
    DEFAULT_CHOICE_ASSAY_POSE_PROCESSOR_CFG,
    ChoiceAssayPoseProcessor,
)
from choice_assay.rpi.choice_assay_trapcam import (
    CA_LEFT_VIDEO_STREAM_INDEX,
    CA_RIGHT_VIDEO_STREAM_INDEX,
    DEFAULT_CHOICE_ASSAY_TRAPCAM_PROCESSOR_CFG,
    ChoiceAssayTrapcamProcessor,
)

###############################################################################
# RpiCore config model
#
# At the top level, we are defining configuration for a fleet of devices.
# This fleet config must be returned as a list of DeviceCfg objects.
# The inventory is passed to RpiCore when it is first configured:
#
#   RpiCore.configure(fleet_config=example.my_fleet_config.INVENTORY)
#
# The DeviceCfg contains:
# - name: a friendly name for the device (eg Alex)
# - notes: free-form notes on what the device is being used for (eg "Experiment 1")
# - system: a SystemCfg object which defines how the system parameters, such as cloud storage configuration
# - sensor_ds_list: a list of SensorDsCfg objects defining each Sensor and its associated Datastreams
#
# A Datastream defines a source of data coming from a Sensor.
# A Sensor may produce multiple Datastreams, each with a different type of data.
# The Sensor configuration is stored in a SensorCfg object.
# The Datastream configuration is stored in a DatastreamCfg object.
# The combined config of a sensor and its datastreams are in a SensorDsCfg object.
#
# The data produced by a Datastream (eg video files) may be processed by 0 or more DataProcessors.
# In the video file example, a DataProcessor might use an ML algorithm to identify bees in a video
# and output the number of bees identified.
# DataProcessors act in a chain, with data being passed from one to the next.
# The DataProcessors associated with a Datastream are defined on the DatastreamCfg
# as lists of DataProcessorCfg objects.
#
###############################################################################

# Pre-configure the devices with awareness of wifi APs
WIFI_CLIENTS: list[WifiClient] = [
    WifiClient("bee-ops", 100, "abcdabcd"),
    WifiClient(ssid="GNX103510", pw="XQSX3SSAPSPH", priority=70),
    WifiClient(ssid="choice_assay", pw="choice_assay", priority=90),
]


################################################################################
# Build up the ChoiceAssay DPtree for each device
#
# Core logic:
# - Sensor
#   - Standard rpicam sensor takes 3 min videos
# - Trapcam with extra logic:
#   - Tuning threshold to be sensitive but avoid shadow type triggering
#   - Continuous fgbg
#   - Function to determine if activity in either side (takes contours list)
#   - Function to save video based on side
#     - ROI to one side or other based on blob location
#     - Save to different data type id based on side
# - ChoiceAssay ML processor to XY data
#   - Pose estimation key points rather than boxes - will require new boxes-to-df logic
#   - ChoiceAssayTrap video if ML detects anything
#   - Aggregate to final mid-proboscis visible seconds
#################################################################################
def create_choice_assay_device() -> list[DPtree]:
    """Create a dual-arena choice assay camera device."""
    # Define the video sensor
    # Camera configuration
    # width: int = 1640
    # height: int = 1232

    cfg = RpicamSensorCfg(
        sensor_type=api.SENSOR_TYPE.CAMERA,
        sensor_index=0,
        sensor_model="PiCameraModule3",
        description="Video sensor that uses rpicam-vid",
        outputs=[RPICAM_STREAM, RPICAM_REVIEW_MODE_STREAM],
        rpicam_cmd="rpicam-vid --framerate 5 --width 1640 --height 1232 -o FILENAME -t 180000",
    )
    my_sensor = RpicamSensor(cfg)

    # Define the Trapcam dataprocessor
    trapcam_dp = ChoiceAssayTrapcamProcessor(
        DEFAULT_CHOICE_ASSAY_TRAPCAM_PROCESSOR_CFG,
        my_sensor.sensor_index,
    )

    # Define the ML dataprocessor
    pose_dp_left = ChoiceAssayPoseProcessor(
        DEFAULT_CHOICE_ASSAY_POSE_PROCESSOR_CFG,
        my_sensor.sensor_index,
    )
    pose_dp_right = ChoiceAssayPoseProcessor(
        DEFAULT_CHOICE_ASSAY_POSE_PROCESSOR_CFG,
        my_sensor.sensor_index,
    )

    my_tree = DPtree(my_sensor)
    my_tree.connect((my_sensor, RPICAM_STREAM_INDEX), trapcam_dp)
    my_tree.connect((trapcam_dp, CA_LEFT_VIDEO_STREAM_INDEX), pose_dp_left)
    my_tree.connect((trapcam_dp, CA_RIGHT_VIDEO_STREAM_INDEX), pose_dp_right)

    return [my_tree]


###############################################################################
# Define per-device configuration for the fleet of devices
###############################################################################
INVENTORY: list[DeviceCfg] = [
    DeviceCfg(
        name="ChoiceAssayRPi-11",
        device_id="d83addae96de",
        notes="Dual-arena choice assay camera with motion detection",
        dp_trees_create_method=create_choice_assay_device,
        wifi_clients=WIFI_CLIENTS,
        tags={
            "Location": "Wytham Field Station",
            "ExperimentType": "BeeChoiceAssay",
        },
    ),
    DeviceCfg(
        name="ChoiceAssayRPi-12",
        device_id="2ccf675765fd",
        notes="Dual-arena choice assay camera with motion detection",
        dp_trees_create_method=create_choice_assay_device,
        wifi_clients=WIFI_CLIENTS,
        tags={
            "Location": "Wytham Field Station",
            "ExperimentType": "BeeChoiceAssay",
        },
        log_level=10,
    ),
]
