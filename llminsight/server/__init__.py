"""Server layer: stdlib http.server backend + static web hosting."""
from .app import serve, STATE, AppState

__all__ = ["serve", "STATE", "AppState"]
