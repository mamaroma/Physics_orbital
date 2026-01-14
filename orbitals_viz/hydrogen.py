import numpy as np
from scipy.special import sph_harm, genlaguerre, factorial


def cartesian_to_spherical(X, Y, Z):
    """
    Returns r, theta, phi in physics convention:
      r >= 0
      theta in [0, pi] (polar angle from +z)
      phi in [0, 2pi) (azimuth from +x toward +y)
    """
    r = np.sqrt(X * X + Y * Y + Z * Z)
    theta = np.zeros_like(r)
    # avoid division by 0
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.arccos(np.where(r > 0, Z / r, 1.0))
    phi = np.arctan2(Y, X)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return r, theta, phi


def radial_R_nl(n: int, l: int, r: np.ndarray, Z: float = 1.0, a0: float = 1.0):
    """
    Radial part R_{nl}(r) for hydrogen-like atom (in atomic units by default: a0=1).
    Uses associated Laguerre polynomials.

    Formula (one common normalization):
      rho = 2 Z r / (n a0)
      R_{nl} = (2Z/(n a0))^(3/2) * sqrt((n-l-1)! / (2n (n+l)!))
               * exp(-rho/2) * rho^l * L_{n-l-1}^{2l+1}(rho)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if l < 0 or l >= n:
        raise ValueError("l must satisfy 0 <= l < n")
    if Z <= 0:
        raise ValueError("Z must be > 0")

    rho = 2.0 * Z * r / (n * a0)

    # Associated Laguerre: L_{n-l-1}^{2l+1}(rho)
    k = n - l - 1
    alpha = 2 * l + 1
    L = genlaguerre(k, alpha)(rho)

    # factorials via scipy.special.factorial (exact integer then float)
    num = factorial(n - l - 1, exact=False)
    den = factorial(n + l, exact=False)

    pref = (2.0 * Z / (n * a0)) ** 1.5
    norm = np.sqrt(num / (2.0 * n * den))

    R = pref * norm * np.exp(-rho / 2.0) * (rho ** l) * L
    return R


def psi_hydrogen_spherical(n: int, l: int, m: int, r, theta, phi, nuclear_charge: float = 1.0):
    """
    Complex hydrogen-like wavefunction psi_{nlm}(r,theta,phi) = R_{nl}(r) Y_l^m(theta,phi).
    scipy.special.sph_harm signature: sph_harm(m, l, phi, theta)
    """
    R = radial_R_nl(n, l, r, Z=nuclear_charge)
    Y = sph_harm(m, l, phi, theta)  # complex
    return R * Y


def psi_hydrogen_cartesian(n: int, l: int, m: int, X, Y, Z, nuclear_charge: float = 1.0):
    r, theta, phi = cartesian_to_spherical(X, Y, Z)
    return psi_hydrogen_spherical(n, l, m, r, theta, phi, nuclear_charge=nuclear_charge)