import torch

from balle2018.runtime import configure_cache


def pytest_configure(config):
    cache = configure_cache("baseline")
    if config.option.basetemp is None:
        config.option.basetemp = str(cache / "test-tmp")
    torch.set_num_threads(2)
