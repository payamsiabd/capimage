import pytest

try:
    import torch
except ImportError:
    torch = None


@pytest.fixture(autouse=True)
def grad_enabled():
    # Importing vlmeval (tests/test_vlmevalkit_integration.py) runs torch.set_grad_enabled(False) globally.
    if torch is None:
        yield
    else:
        with torch.enable_grad():
            yield
