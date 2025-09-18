"""Networks for torch."""

from .randlanet import RandLANet

__all__ = ["RandLANet"]

try:
    from .openvino_model import OpenVINOModel

    __all__.append("OpenVINOModel")
except Exception:
    pass
