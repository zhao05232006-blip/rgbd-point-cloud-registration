"""Small, deterministic static figures; all measurements come from the caller."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOURCE, TARGET = "#d9832e", "#3278ad"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})


def _sample(cloud, limit=6000):
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None
    valid = np.flatnonzero(np.isfinite(points).all(axis=1))
    if len(valid) > limit:
        valid = valid[np.linspace(0, len(valid) - 1, limit, dtype=int)]
    return points[valid], colors[valid] if colors is not None else TARGET


def _move(points, transform):
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4x4 source-to-target transform")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _bounds(*clouds):
    points = np.concatenate(clouds, axis=0)
    if len(points) == 0:
        return np.zeros(3), 1.0
    low, high = points.min(axis=0), points.max(axis=0)
    return (low + high) / 2, max(float((high - low).max()) * 0.55, 0.01)


def _axes3d(ax, bounds):
    center, radius = bounds
    for index, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setter(center[index] - radius, center[index] + radius)
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-65)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis._axinfo["grid"]["color"] = (0.86, 0.86, 0.86, 0.6)
    ax.tick_params(labelsize=8)


def _xz(ax, source, target, bounds):
    for points, color, label in ((target, TARGET, "Target"), (source, SOURCE, "Source")):
        ax.scatter(points[:, 0], points[:, 2], s=2, c=color, alpha=0.5,
                   linewidths=0, label=label, rasterized=True)
    center, radius = bounds
    ax.set(xlim=(center[0] - radius, center[0] + radius),
           ylim=(center[2] - radius, center[2] + radius), xlabel="x (m)", ylabel="z (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right", markerscale=3, framealpha=0.9)


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_inputs(color, depth, cloud, out_dir):
    """Show aligned RGB/depth plus two views of the back-projected cloud."""
    points, colors = _sample(cloud)
    fig = plt.figure(figsize=(12, 9), layout="constrained")
    rgb_ax = fig.add_subplot(221)
    rgb_ax.imshow(color)
    rgb_ax.set_title("RGB input")
    rgb_ax.axis("off")
    depth_ax = fig.add_subplot(222)
    depth_m = np.asarray(depth, dtype=float) / 1000.0
    image = depth_ax.imshow(np.ma.masked_where((depth_m <= 0) | ~np.isfinite(depth_m), depth_m),
                            cmap="viridis")
    depth_ax.set_title("Depth input (missing pixels shown white)")
    depth_ax.axis("off")
    fig.colorbar(image, ax=depth_ax, shrink=0.8, label="Depth (m)")
    flat_ax = fig.add_subplot(223)
    flat_ax.scatter(points[:, 0], points[:, 2], c=colors, s=2, linewidths=0)
    flat_ax.set(xlabel="x (m)", ylabel="z (m)", title="Back-projected cloud: x-z view")
    flat_ax.set_aspect("equal", adjustable="box")
    flat_ax.grid(alpha=0.15)
    space_ax = fig.add_subplot(224, projection="3d")
    space_ax.scatter(*points.T, c=colors, s=2, linewidths=0, depthshade=False)
    _axes3d(space_ax, _bounds(points))
    space_ax.set_title("Back-projected cloud: 3D view")
    fig.suptitle("RGB-D input and geometry | display sample: at most 6,000 points", fontsize=14)
    return _save(fig, Path(out_dir) / "01_rgbd_and_cloud.png")


def save_alignment(source, target, T, out_path, title):
    """Compare the original and transformed source with identical 3D limits."""
    source_points, _ = _sample(source)
    target_points, _ = _sample(target)
    aligned = _move(source_points, T)
    bounds = _bounds(source_points, target_points, aligned)
    fig = plt.figure(figsize=(13, 6), layout="constrained")
    for column, (points, label) in enumerate(((source_points, "Before"), (aligned, "After")), 1):
        ax = fig.add_subplot(1, 2, column, projection="3d")
        for data, color, name in ((target_points, TARGET, "Target"), (points, SOURCE, "Source")):
            ax.scatter(*data.T, c=color, s=2, alpha=0.55, linewidths=0,
                       depthshade=False, label=name)
        _axes3d(ax, bounds)
        ax.set_title(label)
        ax.legend(markerscale=3, loc="upper right")
    fig.suptitle(f"{title}\nSource transformed into target coordinates; identical axes", fontsize=14)
    return _save(fig, out_path)


def save_sweep(rows, out_dir):
    """Render one voxel-by-ICP-threshold heatmap for each supplied metric."""
    voxels = sorted({float(row["voxel_m"]) for row in rows})
    thresholds = sorted({float(row["threshold_m"]) for row in rows})
    if not voxels or not thresholds:
        raise ValueError("The parameter sweep must contain at least one row")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    metrics = (("eval_fitness", "Evaluation fitness", 1, "viridis", ".4f"),
               ("eval_rmse_m", "Inlier RMSE (mm)", 1000, "magma_r", ".3f"),
               ("icp_seconds", "ICP time (s)", 1, "magma_r", ".3f"))
    for ax, (key, label, scale, cmap, fmt) in zip(axes, metrics):
        values = np.full((len(voxels), len(thresholds)), np.nan)
        for row in rows:
            value = row[key]
            values[voxels.index(float(row["voxel_m"])), thresholds.index(float(row["threshold_m"]))] = (
                np.nan if value is None else float(value) * scale)
        kwargs = {"vmin": 0, "vmax": 1} if key == "eval_fitness" else {}
        image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, aspect="equal", **kwargs)
        ax.set_xticks(range(len(thresholds)), [f"{v * 1000:g}" for v in thresholds])
        ax.set_yticks(range(len(voxels)), [f"{v * 1000:g}" for v in voxels])
        ax.set(xlabel="ICP correspondence distance (mm)", ylabel="Voxel size (mm)", title=label)
        for (y, x), value in np.ndenumerate(values):
            ax.text(x, y, format(value, fmt) if np.isfinite(value) else "N/A",
                    ha="center", va="center", color="black", fontsize=11,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2})
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.suptitle("Parameter sweep | fixed evaluation: 10 mm voxel, 30 mm distance\n"
                 "Small metric differences; single-run time only", fontsize=13)
    return _save(fig, Path(out_dir) / "03_parameter_sweep.png")


def save_cases(cases, out_dir):
    """Show each stress case's initial/final x-z projection and supplied scores."""
    if not cases:
        raise ValueError("At least one stress case is required")
    fig, axes = plt.subplots(len(cases), 2, figsize=(12, 4.3 * len(cases)),
                             squeeze=False, layout="constrained")
    for index, case in enumerate(cases):
        source, _ = _sample(case["source"])
        target, _ = _sample(case["target"])
        initial, final = _move(source, case["initial"]), _move(source, case["transform"])
        bounds = _bounds(initial, final, target)
        _xz(axes[index, 0], initial, target, bounds)
        _xz(axes[index, 1], final, target, bounds)
        axes[index, 0].set_title(f"{case['name']} | Initial pose")
        rmse = case["rmse_m"]
        rmse_text = "N/A" if rmse is None or not np.isfinite(rmse) else f"{rmse * 1000:.1f} mm"
        axes[index, 1].set_title(
            f"Final | fitness {case['fitness']:.3f}, reverse {case['reverse_fitness']:.3f}\n"
            f"Inlier RMSE: {rmse_text}")
    fig.suptitle("Stress cases | x-z projection hides y error and can hide misalignment\n"
                 "Same limits within each row; limits may differ between rows", fontsize=13)
    return _save(fig, Path(out_dir) / "04_stress_cases.png")
