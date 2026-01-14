import numpy as np
import plotly.graph_objects as go

from .grid import GridCoords


def _pick_iso_from_quantile(values: np.ndarray, q: float) -> float:
    q = float(q)
    if not (0.0 < q < 1.0):
        raise ValueError("--iso_quantile must be between 0 and 1 (e.g. 0.985).")
    flat = values.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0
    return float(np.quantile(flat, q))


def _nearest_index(vec: np.ndarray, x: float) -> int:
    return int(np.argmin(np.abs(vec - x)))


def _make_slice(psi: np.ndarray, coords: GridCoords, plane: str, pos: float, mode: str):
    if mode == "density":
        field = np.abs(psi) ** 2
        label = "|ψ|²"
    else:
        field = np.real(psi)
        label = "Re(ψ)"

    x, y, z = coords.x, coords.y, coords.z

    if plane == "xy":
        k = _nearest_index(z, pos)
        img = field[:, :, k]
        trace = go.Heatmap(x=x, y=y, z=img.T)
        txt = f"Slice: z={z[k]:.3g}, field={label}"
        return trace, txt

    if plane == "xz":
        j = _nearest_index(y, pos)
        img = field[:, j, :]
        trace = go.Heatmap(x=x, y=z, z=img.T)
        txt = f"Slice: y={y[j]:.3g}, field={label}"
        return trace, txt

    if plane == "yz":
        i = _nearest_index(x, pos)
        img = field[i, :, :]
        trace = go.Heatmap(x=y, y=z, z=img.T)
        txt = f"Slice: x={x[i]:.3g}, field={label}"
        return trace, txt

    raise ValueError("Unknown slice plane. Use xy/xz/yz.")


def make_cloud_figure(
    samples_xyz: np.ndarray,
    *,
    color_value: np.ndarray | None = None,
    title: str,
    point_size: float = 2.0,
    opacity: float = 0.6,
):
    """
    samples_xyz: (N,3)
    color_value:
      - None => single-color markers
      - array (N,) => marker color mapped by Plotly
    """
    x, y, z = samples_xyz[:, 0], samples_xyz[:, 1], samples_xyz[:, 2]

    marker = dict(size=point_size, opacity=opacity)
    if color_value is not None:
        marker["color"] = color_value
        marker["colorscale"] = "RdBu"
        marker["showscale"] = True

    trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=marker,
    )

    fig = go.Figure(data=[trace])
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="cube"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def make_figure(
    psi: np.ndarray,
    coords: GridCoords,
    mode: str,
    iso_quantile: float,
    slice_plane: str,
    slice_pos: float,
    title: str,
):
    x, y, z = coords.x, coords.y, coords.z

    if mode == "density":
        density = np.abs(psi) ** 2
        t = _pick_iso_from_quantile(density, iso_quantile)

        iso = go.Isosurface(
            x=np.repeat(x, coords.N * coords.N),
            y=np.tile(np.repeat(y, coords.N), coords.N),
            z=np.tile(z, coords.N * coords.N),
            value=density.ravel(),
            isomin=t,
            isomax=float(density.max()),
            surface_count=2,
            caps=dict(x_show=False, y_show=False, z_show=False),
        )

        slice_trace, slice_txt = _make_slice(psi, coords, slice_plane, slice_pos, mode="density")

        fig = go.Figure(data=[iso])
        fig.update_layout(
            title=title,
            scene=dict(aspectmode="cube"),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        fig2 = go.Figure(fig)
        fig2.add_trace(slice_trace)
        fig2.data[-1].update(xaxis="x", yaxis="y")
        fig2.update_layout(
            xaxis=dict(domain=[0, 1], anchor="y"),
            yaxis=dict(domain=[0, 0.35]),
            scene=dict(domain=dict(x=[0, 1], y=[0.4, 1])),
            annotations=[dict(text=slice_txt, x=0.5, y=0.37, xref="paper", yref="paper", showarrow=False)],
        )
        return fig2

    # signed mode
    signed = np.real(psi)
    t = _pick_iso_from_quantile(np.abs(signed), iso_quantile)
    vmax = float(np.max(np.abs(signed)))

    iso_pos = go.Isosurface(
        x=np.repeat(x, coords.N * coords.N),
        y=np.tile(np.repeat(y, coords.N), coords.N),
        z=np.tile(z, coords.N * coords.N),
        value=signed.ravel(),
        isomin=t,
        isomax=vmax,
        surface_count=2,
        caps=dict(x_show=False, y_show=False, z_show=False),
    )
    iso_neg = go.Isosurface(
        x=np.repeat(x, coords.N * coords.N),
        y=np.tile(np.repeat(y, coords.N), coords.N),
        z=np.tile(z, coords.N * coords.N),
        value=signed.ravel(),
        isomin=-vmax,
        isomax=-t,
        surface_count=2,
        caps=dict(x_show=False, y_show=False, z_show=False),
    )

    slice_trace, slice_txt = _make_slice(psi, coords, slice_plane, slice_pos, mode="signed")

    fig = go.Figure(data=[iso_pos, iso_neg])
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="cube"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig2 = go.Figure(fig)
    fig2.add_trace(slice_trace)
    fig2.data[-1].update(xaxis="x", yaxis="y")
    fig2.update_layout(
        xaxis=dict(domain=[0, 1], anchor="y"),
        yaxis=dict(domain=[0, 0.35]),
        scene=dict(domain=dict(x=[0, 1], y=[0.4, 1])),
        annotations=[dict(text=slice_txt, x=0.5, y=0.37, xref="paper", yref="paper", showarrow=False)],
    )
    return fig2