"""Parser layer: raw Ascend profiler output -> unified ProfileData model."""
from .profile import ProfileData, load_profile

__all__ = ["ProfileData", "load_profile"]
