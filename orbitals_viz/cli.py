import argparse
import sys
import numpy as np

from .grid import make_cartesian_grid
from .hydrogen import psi_hydrogen_cartesian
from .sampler import metropolis_sample_positions
from .viz import make_figure, make_cloud_figure


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="orbitals_viz",
        description="3D visualization of hydrogen-like atomic orbitals (Plotly).",
    )
    p.add_argument("--n", type=int, default=2, help="Principal quantum number n (>=1)")
    p.add_argument("--l", type=int, default=1, help="Azimuthal quantum number l (0<=l<n)")
    p.add_argument("--m", type=int, default=0, help="Magnetic quantum number m (-l<=m<=l)")
    p.add_argument("--Z", type=float, default=1.0, help="Nuclear charge Z (hydrogen-like)")

    p.add_argument(
        "--mode",
        choices=["density", "signed", "cloud"],
        default="cloud",
        help="density: |psi|^2 isosurface, signed: Re(psi) +/- isosurfaces, cloud: Monte-Carlo point cloud",
    )

    # grid params (for isosurface modes)
    p.add_argument("--grid", type=int, default=85, help="Grid size N (e.g. 70..120)")
    p.add_argument("--extent", type=float, default=14.0, help="Cube extent in a0: [-E, E]")
    p.add_argument("--iso_quantile", type=float, default=0.985, help="Quantile for iso threshold")

    # slice
    p.add_argument("--slice", choices=["xy", "xz", "yz"], default="xy", help="Slice plane")
    p.add_argument("--slice_pos", type=float, default=0.0, help="Slice position (a0)")

    # cloud params
    p.add_argument("--points", type=int, default=60000, help="Number of points in cloud (e.g. 20000..200000)")
    p.add_argument("--burnin", type=float, default=0.15, help="Burn-in fraction for Metropolis")
    p.add_argument("--step", type=float, default=1.2, help="Metropolis proposal step size")
    p.add_argument("--thin", type=int, default=1, help="Keep every thin-th state after burn-in")
    p.add_argument("--init_box", type=float, default=0.5, help="Initial position box size")
    p.add_argument("--cloud_extent", type=float, default=16.0, help="Soft display limit (just for title/range)")

    p.add_argument(
        "--cloud_color",
        choices=["none", "sign", "re"],
        default="sign",
        help="Color points: none, sign (sign(Re(psi))), re (Re(psi) value)",
    )

    p.add_argument("--out", type=str, default="orbitals.html", help="Output HTML file")
    return p


def validate_quantum_numbers(n: int, l: int, m: int) -> None:
    if n < 1:
        raise ValueError("n must be >= 1")
    if l < 0 or l >= n:
        raise ValueError("l must satisfy 0 <= l < n")
    if m < -l or m > l:
        raise ValueError("m must satisfy -l <= m <= l")


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_quantum_numbers(args.n, args.l, args.m)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    title = f"Hydrogen-like orbital n={args.n}, l={args.l}, m={args.m}, mode={args.mode}"

    if args.mode == "cloud":
        # log probability function: log(|psi|^2 + eps)
        eps = 1e-300  # prevents log(0)

        def logp(r3: np.ndarray) -> float:
            x, y, z = r3[0], r3[1], r3[2]
            psi = psi_hydrogen_cartesian(args.n, args.l, args.m, x, y, z, nuclear_charge=args.Z)
            p = (abs(psi) ** 2) + eps
            return float(np.log(p))

        samples, acc = metropolis_sample_positions(
            logp,
            n_samples=args.points,
            burnin=args.burnin,
            step_size=args.step,
            init_box=args.init_box,
            thin=args.thin,
        )

        # Color points
        color_value = None
        if args.cloud_color in ("sign", "re"):
            # compute Re(psi) for each sample (vectorized loop to keep code simple/robust)
            re_vals = np.empty(samples.shape[0], dtype=np.float64)
            for i in range(samples.shape[0]):
                x, y, z = samples[i]
                re_vals[i] = float(np.real(psi_hydrogen_cartesian(args.n, args.l, args.m, x, y, z, nuclear_charge=args.Z)))

            if args.cloud_color == "sign":
                color_value = np.sign(re_vals)
            else:
                color_value = re_vals

        fig = make_cloud_figure(
            samples,
            color_value=color_value,
            title=title + f" | Metropolis acc≈{acc:.1f}%",
            point_size=2.0,
            opacity=0.55,
        )
        fig.write_html(args.out, auto_open=True)
        print(f"Saved: {args.out}")
        return

    # isosurface modes (как раньше)
    X, Y, Z, coords = make_cartesian_grid(args.grid, args.extent)
    psi = psi_hydrogen_cartesian(args.n, args.l, args.m, X, Y, Z, nuclear_charge=args.Z)

    fig = make_figure(
        psi=psi,
        coords=coords,
        mode=args.mode,
        iso_quantile=args.iso_quantile,
        slice_plane=args.slice,
        slice_pos=args.slice_pos,
        title=title,
    )

    fig.write_html(args.out, auto_open=True)
    print(f"Saved: {args.out}")