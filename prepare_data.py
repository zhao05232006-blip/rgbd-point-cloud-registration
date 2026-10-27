"""Download the official Fountain sample and keep six unchanged RGB-D pairs."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import open3d as o3d

ROOT = Path(__file__).resolve().parent


def main():
    dataset = o3d.data.SampleFountainRGBDImages(data_root=str(ROOT / ".cache"))
    trajectory = o3d.io.read_pinhole_camera_trajectory(dataset.keyframe_poses_log_path)
    out = ROOT / "data" / "fountain"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(6):
        files = {}
        for kind, original in [("color", dataset.color_paths[i]), ("depth", dataset.depth_paths[i])]:
            original = Path(original)
            destination = out / f"frame_{i:02d}_{kind}{original.suffix}"
            shutil.copy2(original, destination)
            files[kind] = destination.name
            files[f"{kind}_original"] = original.name
            files[f"{kind}_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        files["index"] = i
        # Open3D inverts key.log's camera-to-world matrices on import.
        files["world_to_camera"] = trajectory.parameters[i].extrinsic.tolist()
        frames.append(files)
    shutil.copy2(dataset.keyframe_poses_log_path, out / "key.log")
    intrinsic = trajectory.parameters[0].intrinsic
    manifest = {
        "dataset": "Open3D SampleFountainRGBDImages (six selected keyframes)",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://www.qianyi.info/scenedata.html",
        "download": "https://github.com/isl-org/open3d_downloads/releases/download/20220201-data/SampleFountainRGBDImages.zip",
        "credit": "Qian-Yi Zhou and Vladlen Koltun, Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras, SIGGRAPH 2014.",
        "intrinsic_note": "PrimeSenseDefault assumed by Open3D's LOG reader; not a calibration performed in this project.",
        "pose_note": "Dataset-estimated poses, not independent ground truth. Used for diagnostics only, never ICP initialization.",
        "width": intrinsic.width,
        "height": intrinsic.height,
        "K": intrinsic.intrinsic_matrix.tolist(),
        "depth_scale": 1000.0,
        "depth_max_m": 3.0,
        "frames": frames,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(frames)} RGB-D pairs to {out}")


if __name__ == "__main__":
    main()
