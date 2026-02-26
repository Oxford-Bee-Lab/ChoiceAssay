import time
from collections.abc import Generator

import pytest
from expidite_rpi import DeviceCfg
from expidite_rpi.core import configuration as root_cfg
from expidite_rpi.utils.rpi_emulator import RpiEmulator

# Set up logger for test execution
test_logger = root_cfg.setup_logger("expidite")

# Simple ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    tr = terminalreporter
    tr.write_sep("=", "Test Results Summary")
    reports = tr.getreports("passed") + tr.getreports("failed") + tr.getreports("skipped")

    for rep in reports:
        duration = getattr(rep, "duration", None)
        outcome = rep.outcome.upper()

        if outcome == "PASSED":
            outcome_colored = f"{GREEN}{outcome}{RESET}"
        elif outcome == "FAILED":
            outcome_colored = f"{RED}{outcome}{RESET}"
        elif outcome == "SKIPPED":
            outcome_colored = f"{YELLOW}{outcome}{RESET}"
        else:
            outcome_colored = outcome

        test_name = rep.nodeid.split("::")[-1]
        tr.write_line(f"{test_name} - {outcome_colored} - {duration:.3f}s")


@pytest.fixture(autouse=True)
def log_test_lifecycle(request):
    """Pytest fixture that automatically logs the start and end of every test.

    This runs for all tests without requiring a decorator.
    """
    test_name = request.node.name
    module_name = request.node.module.__name__ if request.node.module else "unknown"

    # Log test start
    start_time = time.time()
    test_logger.info(f"[PYTEST START] {module_name}::{test_name}")

    yield  # This is where the test runs

    # Log test end
    duration = time.time() - start_time
    test_result = "PASSED" if not request.node.rep_call.failed else "FAILED"

    if test_result == "PASSED":
        test_logger.info(
            f"[PYTEST END] {module_name}::{test_name} - {test_result} (Duration: {duration:.3f}s)"
        )
    else:
        test_logger.error(
            f"[PYTEST END] {module_name}::{test_name} - {test_result} (Duration: {duration:.3f}s)"
        )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results for the log_test_lifecycle fixture."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture
def inventory() -> list[DeviceCfg]:
    """Pytest fixture that should be overridden in each test to provide device inventory.

    This fixture should be defined in each test class or test file that uses rpi_emulator.
    This default will use the inventory defined in the CFG_DIR/system.cfg file.

    Example usage in test:
        @pytest.fixture
        def inventory(self) -> List[DeviceCfg]:
            return [
                DeviceCfg(
                    name="TestDevice",
                    device_id="d01111111111",
                    dp_trees_create_method=create_test_device,
                ),
            ]
    """
    try:
        inventory = root_cfg.load_configuration()
        assert inventory is not None
        return inventory
    except Exception as e:
        msg = f"Failed to load inventory from configuration: {e}"
        raise RuntimeError(msg) from e


@pytest.fixture
def rpi_emulator(inventory: list[DeviceCfg]) -> Generator[RpiEmulator, None, None]:
    """Pytest fixture that provides an RpiEmulator instance with mocked timers.

    Automatically applies mock_timers to the provided inventory.

    Requires an 'inventory' fixture to be defined in the test.

    Usage:
        @pytest.fixture
        def inventory(self) -> List[DeviceCfg]:
            return [DeviceCfg(...)]

        @pytest.mark.unittest
        def test_my_sensor(self, rpi_emulator: RpiEmulator):
            rpi_emulator.set_recording_cap(1, type_id="MYSENSOR")
            # inventory is already mocked with timers
            # test code here...
    """
    with RpiEmulator.get_instance() as emulator:
        # Automatically mock timers for the provided inventory
        mocked_inventory = emulator.mock_timers(inventory)
        # Store the mocked inventory on the emulator for easy access
        emulator.inventory = mocked_inventory
        yield emulator
