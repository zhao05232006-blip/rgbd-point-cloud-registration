"""Run a small parameter study. All reported numbers are computed from the files."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib"))

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d

from pipeline import DATA, align, load_frame, pose_difference, preprocess, reference_transform, score, transformed
from figures import save_alignment, save_cases, save_inputs, save_sweep

ROOT = Path(__file__).resolve().parent
EVAL_VOXEL = 0.01
EVAL_THRESHOLD = 0.03


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def verify_data():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for frame in manifest["frames"]:
        for kind in ("color", "depth"):
            digest = hashlib.sha256((DATA / frame[kind]).read_bytes()).hexdigest()
            if digest != frame[f"{kind}_sha256"]:
                raise ValueError(f"Data changed: {frame[kind]}")
    return manifest


def evaluate_case(raw_source, raw_target, manifest, si, ti, voxel, threshold, initial, label):
    source, ns = preprocess(raw_source, voxel)
    target, nt = preprocess(raw_target, voxel)
    # The evaluation clouds do not change with the optimizer's voxel size.
    eval_s, _ = preprocess(raw_source, EVAL_VOXEL)
    eval_t, _ = preprocess(raw_target, EVAL_VOXEL)
    before = score(eval_s, eval_t, initial, EVAL_THRESHOLD)
    start = time.perf_counter()
    result = align(source, target, threshold, initial)
    seconds = time.perf_counter() - start
    after = score(eval_s, eval_t, result.transformation, EVAL_THRESHOLD)
    reverse = score(eval_t, eval_s, np.linalg.inv(result.transformation), EVAL_THRESHOLD)
    ref = reference_transform(manifest, si, ti)
    ref_score = score(eval_s, eval_t, ref, EVAL_THRESHOLD)
    record = {
        "case": label, "source_frame": si, "target_frame": ti,
        "voxel_m": voxel, "threshold_m": threshold,
        "source_raw_points": ns["raw"], "source_voxel_points": ns["voxel"],
        "source_filtered_points": ns["filtered"], "target_filtered_points": nt["filtered"],
        "initial_fitness": before["fitness"], "initial_rmse_m": before["inlier_rmse_m"],
        "eval_fitness": after["fitness"], "eval_rmse_m": after["inlier_rmse_m"],
        "eval_correspondences": after["correspondences"], "reverse_fitness": reverse["fitness"],
        "icp_seconds": seconds,
        "reference_fitness": ref_score["fitness"],
        **pose_difference(result.transformation, ref),
        "initial_transform": initial.tolist(), "transform": result.transformation.tolist(),
        "reference_transform": ref.tolist(),
    }
    plot_case = {"name": label, "source": source, "target": target, "initial": initial,
                 "transform": result.transformation, "fitness": after["fitness"],
                 "reverse_fitness": reverse["fitness"], "rmse_m": after["inlier_rmse_m"]}
    print(f"{label:28s} voxel={voxel:.3f} threshold={threshold:.3f} fitness={after['fitness']:.4f} RMSE={after['inlier_rmse_m']}", flush=True)
    return record, plot_case


def make_report(output, records, mode):
    baseline = next((r for r in records if r["case"] == "identity_baseline"), records[0])
    def fmt(number, scale=1, digits=3):
        return "no matches" if number is None else f"{number * scale:.{digits}f}"
    rows = "".join(
        f"<tr><td>{html.escape(r['case'])}</td><td>{r['source_frame']} → {r['target_frame']}</td>"
        f"<td>{r['voxel_m']:.2f}</td><td>{r['threshold_m']:.2f}</td>"
        f"<td>{r['eval_fitness']:.3f}</td><td>{r['reverse_fitness']:.3f}</td>"
        f"<td>{fmt(r['eval_rmse_m'], 1000)}</td>"
        f"<td>{r['reference_rotation_difference_deg']:.2f}</td>"
        f"<td>{r['reference_translation_difference_m']*100:.2f}</td></tr>" for r in records)
    figures = ["01_rgbd_and_cloud.png", "02_alignment.png"]
    if mode == "all":
        figures += ["03_parameter_sweep.png", "04_stress_cases.png"]
    images = "".join(f'<figure><img src="figures/{f}" alt="{f}"><figcaption>{f}</figcaption></figure>' for f in figures)
    observations = ""
    if mode == "all":
        crop = next(r for r in records if r["case"] == "restricted_image_coverage")
        bad = next(r for r in records if r["case"] == "poor_initial_guess")
        p23 = next(r for r in records if r["source_frame"] == 2)
        p45 = next(r for r in records if r["source_frame"] == 4)
        sweep = [r for r in records if r["case"] in ("identity_baseline", "parameter_sweep")]
        observations = f"""<h2>Observations from this run</h2><ul>
<li>The nine sweep configurations are close: fitness {min(r['eval_fitness'] for r in sweep):.5f}–{max(r['eval_fitness'] for r in sweep):.5f}, RMSE {min(r['eval_rmse_m'] for r in sweep)*1000:.3f}–{max(r['eval_rmse_m'] for r in sweep)*1000:.3f} mm. This small experiment does not establish a significantly better parameter choice.</li>
<li>The poor initial guess ends with fitness {bad['eval_fitness']:.3f} and a {bad['reference_translation_difference_m']*100:.1f} cm translation difference from the reference. The alignment plot confirms substantial remaining mismatch.</li>
<li>The crop test ends with fitness {crop['eval_fitness']:.3f}; the reference transform itself gives {crop['reference_fitness']:.3f}. Its reference translation difference is only {crop['reference_translation_difference_m']*100:.2f} cm. Low coverage in this case is not evidence that registration failed.</li>
<li>Pair 2 → 3 gives ICP fitness {p23['eval_fitness']:.3f}, while the reference transform gives {p23['reference_fitness']:.3f}. A {p23['reference_translation_difference_m']*100:.2f} cm translation discrepancy can cross the 3 cm matching cutoff. This also illustrates why the estimated reference is not exact ground truth.</li>
<li>Pair 4 → 5 gives fitness {p45['eval_fitness']:.3f}, but differs from the reference translation by {p45['reference_translation_difference_m']*100:.2f} cm. A local alignment error is possible; this is not counted as independently verified pose success.</li></ul>"""
    report = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>RGB-D point-cloud registration — experiment report</title>
<style>body{{max-width:1100px;margin:40px auto;padding:0 22px;font:16px/1.6 Arial,sans-serif;color:#222}}h1{{font-size:30px}}h2{{margin-top:36px}}img{{max-width:100%;height:auto}}figure{{margin:24px 0}}figcaption{{color:#555;font-size:13px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #ccc}}.scroll{{overflow-x:auto}}code{{background:#eee;padding:2px 5px}}a{{color:#255573}}</style>
<h1>RGB-D point-cloud processing and registration</h1>
<p>A small offline experiment with six real Fountain RGB-D keyframes. Prepared {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.</p>
<p><a href="../START_HERE.html">中文学习入口</a> · <a href="metrics.csv">Raw metrics</a> · <a href="experiments.json">Matrices and full records</a> · <a href="run_metadata.json">Run details</a></p>
<h2>Question</h2><p>How do voxel size, correspondence distance, initialization and restricted image coverage affect pairwise ICP?</p>
<h2>Method</h2><p>NumPy pinhole back-projection; Open3D voxel downsampling, statistical outlier removal (20 neighbours, 2.0 standard-deviation ratio), normal estimation (radius = 3 × voxel size), and point-to-plane ICP (at most 80 iterations). ICP is supplied by Open3D, not implemented from scratch here. No learned model is trained.</p>
<p>The default pair is keyframe 0 → 1. These are sampled keyframes, not consecutive raw video frames. Baseline initialization is identity. Dataset poses are used only for the diagnostic comparison after registration.</p>
<h2>Baseline result</h2><p>For keyframe {baseline['source_frame']} → {baseline['target_frame']}, at voxel {baseline['voxel_m']:.2f} m and correspondence distance {baseline['threshold_m']:.2f} m, fixed-evaluation source coverage changed from {baseline['initial_fitness']:.3f} to {baseline['eval_fitness']:.3f}. Inlier RMSE changed from {fmt(baseline['initial_rmse_m'],1000)} to {fmt(baseline['eval_rmse_m'],1000)} mm. These two statistics must be read together.</p>
{images}
{observations}
<h2>All runs</h2><p>Every row is evaluated at 1 cm voxel size and 3 cm matching distance. Sweep rows use exactly the same evaluation point sets. Cropped cases have different point sets, so their fitness values are not a direct ranking against the full-view case. The reverse column measures target → source coverage. Times are single-run observations, not a rigorous speed benchmark.</p>
<div class="scroll"><table><thead><tr><th>Case</th><th>Pair</th><th>Voxel m</th><th>ICP limit m</th><th>Fitness</th><th>Reverse</th><th>RMSE mm</th><th>Ref. Δrot °</th><th>Ref. Δtrans cm</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>What the numbers do and do not mean</h2><ul>
<li>Fitness is the fraction of source points with a target neighbour within 3 cm. A large value is not proof of a correct pose.</li>
<li>RMSE uses only those matching points. A tiny RMSE with very few matches can be misleading. Empty matches are saved as null, not zero.</li>
<li>Reference differences compare against dataset-estimated camera poses, not independent ground truth.</li>
<li>The crop test retains source columns 0–383 and target columns 256–639; pixels outside those ranges are masked. Pixel retention is not a measured 3D overlap percentage.</li>
<li>The poor initial guess rotates 25° around camera y and translates +0.30 m along x. A separate +5 m x test checks the no-correspondence case.</li>
<li>Only one small scene is studied. No camera calibration, LiDAR acquisition, Jetson deployment, object detector or robot manipulation is demonstrated.</li></ul>
<h2>Data and reference</h2><p>Qian-Yi Zhou and Vladlen Koltun, <i>Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras</i>, SIGGRAPH 2014. <a href="https://www.qianyi.info/scenedata.html">Author's data page and usage terms</a>. Fountain was recorded with an Asus Xtion Pro Live; original image files are unchanged. This project uses the documented PrimeSense pinhole parameters, rather than performing its own calibration.</p>
<p><a href="https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html">Open3D ICP documentation</a>. Open3D: Qian-Yi Zhou, Jaesik Park and Vladlen Koltun, 2018.</p></html>"""
    (output / "report.html").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "single"], default="all")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    if not 0 <= args.source < 6 or not 0 <= args.target < 6 or args.source == args.target:
        parser.error("Choose two different frame indices between 0 and 5")
    if args.voxel <= 0 or args.threshold <= 0:
        parser.error("Voxel size and threshold must be positive")
    if args.mode == "all" and (args.source, args.target, args.voxel, args.threshold) != (0, 1, 0.02, 0.05):
        parser.error("Use --mode single to change the baseline parameters")
    manifest = verify_data()
    output = ROOT / args.output
    plots = output / "figures"
    plots.mkdir(parents=True, exist_ok=True)
    source, color, depth, _ = load_frame(args.source)
    target, _, _, _ = load_frame(args.target)
    save_inputs(color, depth, source, plots)
    baseline, base_plot = evaluate_case(source, target, manifest, args.source, args.target,
                                        args.voxel, args.threshold, np.eye(4), "identity_baseline")
    records = [baseline]
    save_alignment(base_plot["source"], base_plot["target"], base_plot["transform"],
                   plots / "02_alignment.png", "Fountain: baseline pairwise ICP")
    cloud_dir = output / "clouds"
    cloud_dir.mkdir(exist_ok=True)
    o3d.io.write_point_cloud(str(cloud_dir / "source_clean.ply"), base_plot["source"])
    o3d.io.write_point_cloud(str(cloud_dir / "target_clean.ply"), base_plot["target"])
    o3d.io.write_point_cloud(str(cloud_dir / "source_aligned.ply"), transformed(base_plot["source"], base_plot["transform"]))
    if args.mode == "all":
        sweep = []
        for voxel in [0.01, 0.02, 0.04]:
            for threshold in [0.02, 0.05, 0.10]:
                if (voxel, threshold) == (0.02, 0.05):
                    sweep.append(baseline)
                    continue
                row, _ = evaluate_case(source, target, manifest, 0, 1, voxel, threshold, np.eye(4), "parameter_sweep")
                records.append(row)
                sweep.append(row)
        save_sweep(sweep, plots)
        cases = [base_plot]
        bad = np.eye(4)
        bad[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([0, np.radians(25), 0])
        bad[0, 3] = 0.30
        cropped_s, _, _, _ = load_frame(0, (0, 384))
        cropped_t, _, _, _ = load_frame(1, (256, 640))
        far = np.eye(4)
        far[0, 3] = 5.0
        for raw_s, raw_t, init, label in [(source, target, bad, "poor_initial_guess"),
                                         (cropped_s, cropped_t, np.eye(4), "restricted_image_coverage"),
                                         (source, target, far, "no_matches_sanity_check")]:
            row, plot_case = evaluate_case(raw_s, raw_t, manifest, 0, 1, .02, .05, init, label)
            records.append(row)
            cases.append(plot_case)
        save_cases(cases, plots)
        for si, ti in [(2, 3), (4, 5)]:
            raw_s, _, _, _ = load_frame(si)
            raw_t, _, _, _ = load_frame(ti)
            row, _ = evaluate_case(raw_s, raw_t, manifest, si, ti, .02, .05, np.eye(4), "additional_pair")
            records.append(row)
    write_json(output / "experiments.json", records)
    scalar_keys = [k for k in records[0] if not isinstance(records[0][k], list)]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    metadata = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "platform": platform.platform(), "settings": vars(args),
                "evaluation_voxel_m": EVAL_VOXEL, "evaluation_distance_m": EVAL_THRESHOLD,
                "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
                "packages": {name: importlib.metadata.version(name) for name in ["numpy", "open3d", "matplotlib", "pillow"]},
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")},
                "data_manifest_sha256": hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest(),
                "number_of_runs": len(records), "reference_pose_used_for_initialization": False}
    write_json(output / "run_metadata.json", metadata)
    make_report(output, records, args.mode)
    print(f"Wrote {len(records)} experiments and report: {output / 'report.html'}")


if __name__ == "__main__":
    main()
