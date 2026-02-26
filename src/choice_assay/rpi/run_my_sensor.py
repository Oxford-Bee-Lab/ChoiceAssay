###############################################################################
# The run_my_sensor script is invoked by the RpiCore at startup.
# It provides a means for users to customize the behavior of the RpiCore
# and run their own code.
#
# By default, it:
# - loads the fleet configuration specified in system_cfg.my_fleet_config
# - starts the RpiCore
#################################################################################
from time import sleep

from expidite_rpi import RpiCore
from expidite_rpi.core import configuration as root_cfg

logger = root_cfg.setup_logger("expidite")


def main() -> None:
    """Run RpiCore as defined in the system.cfg file."""
    try:
        # Configure the RpiCore with the fleet configuration
        # This will load the configuration and check for errors
        logger.info("Creating RpiCore...")
        sc = RpiCore()

        # Load_configuration loads the configuration specified in system_cfg.my_fleet_config
        logger.info("Configuring RpiCore...")
        inventory = root_cfg.load_configuration()
        if inventory is None:
            logger.error("Failed to load inventory. Exiting...")
            return

        sc.configure(inventory)

        # Start the RpiCore and begin data collection
        logger.info("Starting RpiCore...")
        sc.start()
        while True:
            logger.info(sc.status())
            sleep(1800)

    except KeyboardInterrupt:
        logger.exception("Keyboard interrupt => stopping RpiCore... this may take up to 180s.")
        sc.stop()
    except Exception:
        logger.exception("Error")
        sc.stop()


if __name__ == "__main__":
    main()
