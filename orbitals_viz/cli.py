import argparse
import sys

from .grid import make_cartesian_grid
from .hydrogen import psi_hydrogen_cartesian
from .viz import make_figure


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="orbitals_viz",
        description="3D visualization of hydrogen-like atomic orbitals (Plotly).",
    )
    p.add_argument("--n", type=int, default=2, help="Principal quantum number n (>=1)")
    p.add_argument("--l", type=int, default=1, help="Azimuthal quantum number l (0<=l<n)")
    p.add_argument("--m", type=int, default=0, help="Magnetic quantum number m (-l<=m<=l)")
    p.add_argument("--Z", type=float, default=1.0, help="Nuclear charge Z (hydrogen-like)")

    p.add_argument("--mode", choices=["density", "signed"], default="density",
                   help="density: |psi|^2, signed: Re(psi) with +/- isosurfaces")
    p.add_argument("--grid", type=int, default=85, help="Grid size N (e.g. 70..120)")
    p.add_argument("--extent", type=float, default=14.0, help="Cube extent in a0: [-E, E]")

    p.add_argument("--iso_quantile", type=float, default=0.985,
                   help="Quantile for iso threshold (0.97..0.995 recommended)")

    p.add_argument("--slice", choices=["xy", "xz", "yz"], default="xy",
                   help="Slice plane for 2D heatmap")
    p.add_argument("--slice_pos", type=float, default=0.0,
                   help="Slice position along the remaining axis (in a0), e.g. 0")

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

    # Create grid
    X, Y, Z, coords = make_cartesian_grid(args.grid, args.extent)

    # Compute psi on grid (complex)
    psi = psi_hydrogen_cartesian(args.n, args.l, args.m, X, Y, Z, nuclear_charge=args.Z)

    # Build figure
    fig = make_figure(
        psi=psi,
        coords=coords,
        mode=args.mode,
        iso_quantile=args.iso_quantile,
        slice_plane=args.slice,
        slice_pos=args.slice_pos,
        title=f"Hydrogen-like orbital n={args.n}, l={args.l}, m={args.m}, mode={args.mode}",
    )

    fig.write_html(args.out, auto_open=True)
    print(f"Saved: {args.out}")