from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GridCoords:
    """Coordinate vectors for plotly and slicing."""
    x: np.ndarray  # shape (N,)
    y: np.ndarray  # shape (N,)
    z: np.ndarray  # shape (N,)
    extent: float
    N: int


def make_cartesian_grid(N: int, extent: float):
    """
    Returns:
      X, Y, Z: 3D arrays shape (N,N,N)
      coords: GridCoords with 1D coordinate vectors
    """
    if N < 10:
        raise ValueError("Grid size N too small; use >= 30 for meaningful visuals.")
    if extent <= 0:
        raise ValueError("extent must be positive.")

    x = np.linspace(-extent, extent, N, dtype=np.float64)
    y = np.linspace(-extent, extent, N, dtype=np.float64)
    z = np.linspace(-extent, extent, N, dtype=np.float64)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    coords = GridCoords(x=x, y=y, z=z, extent=extent, N=N)
    return X, Y, Z, coords