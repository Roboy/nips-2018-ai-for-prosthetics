import pytest


def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true",
                     default=False, help="run integration tests")
    parser.addoption("--all", action="store_true", default=False, help="run all tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--all"):
        return

    collected_tests = items
    if config.getoption("--integration"):
        skip_marker = pytest.mark.skip(reason="running only integration tests")
        tests_to_skip = [t for t in collected_tests if "integration" not in t.keywords]
    else:
        skip_marker = pytest.mark.skip(reason="running only unit tests")
        tests_to_skip = [t for t in collected_tests if "integration" in t.keywords]

    for test in tests_to_skip:
        test.add_marker(skip_marker)
