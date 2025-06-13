import os
from PIL import Image
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
from scipy.ndimage import label
import glob
from typing import List, Tuple
from pathlib import Path
import os
import numpy as np
import matplotlib.cm as cm                 # → colormap machinery :contentReference[oaicite:0]{index=0}
from PIL import Image

# Input/output paths
input_dir = "/home/jordanprescott/shiv_research/tnt_data/rgbd_scenenet/train/0/1/photo"
output_gif = "/home/jordanprescott/shiv_research/tnt_data/rgbd_scenenet/train/0/1/photo.gif"

def make_gif(input_dir, output_path):
    images = []
    # Sort filenames numerically by stripping extension and converting to int
    for filename in sorted(
        os.listdir(input_dir),
        key=lambda x: int(os.path.splitext(x)[0])
    ):
        if filename.endswith(('.png', '.jpg')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert("RGBA")
            images.append(img)

    if not images:
        print("No images found in", input_dir)
        return

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=100
    )
    print(f"GIF saved to {output_path}")


def make_depth_gif(
        input_dir,
        output_path,
        cmap_name = "plasma",          # any Matplotlib cmap  :contentReference[oaicite:1]{index=1}
        duration_ms = 100,
    ):
    """
    Read every *.png / *.jpg* in `input_dir`, assume it encodes a depth map
    (uint8/uint16/float), colourise it via `cmap_name`, and save an animated
    GIF to `output_path`.
    """
    input_dir  = Path(input_dir)
    cmap       = cm.get_cmap(cmap_name)     # returns callable v∈[0,1] → RGBA  :contentReference[oaicite:2]{index=2}
    images     = []

    for file in sorted(input_dir.iterdir(), key=lambda p: int(p.stem)):
        if file.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            continue

        depth = np.array(Image.open(file))  # keeps original dtype  :contentReference[oaicite:3]{index=3}

        # -------------------------------- normalise depth to [0,1]
        d = depth.astype(np.float32)
        vmin, vmax = np.percentile(d, 1), np.percentile(d, 99)  # robust min/max
        d_norm = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)

        # -------------------------------- apply colormap  (H,W,4) float
        rgba = cmap(d_norm)
        rgb  = (rgba[..., :3] * 255).astype(np.uint8)           # drop alpha

        # -------------------------------- PIL → 'P' so GIF palette behaves
        img = Image.fromarray(rgb, mode="RGB").convert("P", palette=Image.ADAPTIVE)
        images.append(img)

    if not images:
        raise FileNotFoundError(f"No depth files found in {input_dir}")

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=duration_ms,
        optimize=False
    )
    print(f"GIF saved → {output_path}")


import os, glob
import numpy as np
from PIL import Image
from typing import List, Tuple

def zeros_in_image(path: str) -> int:
    """Count pixels that are exactly 0 (all channels)."""
    return int((np.asarray(Image.open(path)) == 0).sum())

def classify_folders(base_dir: str, *, max_zeros: int = 0) -> Tuple[List[int], List[int]]:
    good_with_counts: List[Tuple[int, int]] = []   # (folder_id, zero_count)
    bad_folders: List[int] = []

    for folder_name in os.listdir(base_dir):
        print(folder_name)
        folder_id  = int(folder_name)                              # keep as int once
        photo_glob = os.path.join(base_dir, folder_name, "photo", "*.*")

        total = 0
        for img in glob.glob(photo_glob):
            total += zeros_in_image(img)
            if total > max_zeros:              # early exit → folder is bad
                bad_folders.append(folder_id)
                break
        else:                                  # only runs if no break
            good_with_counts.append((folder_id, total))

    # ---- SORT the list in ascending order of zero_count (then folder_id) ---
    good_with_counts.sort(key=lambda t: (t[1], t[0]))   # in-place & stable

    good_sorted = [fid for fid, _ in good_with_counts]  # strip counts
    return good_sorted, bad_folders, good_with_counts


root = "/home/jordanprescott/shiv_research/tnt_data/rgbd_scenenet/train/0"
# good, bad, gwc = classify_folders(root, max_zeros=10000)
# 
# print("good folders:", good)          # ascending by zero count
# print()
# print("bad  folders:", bad)
# print()
# print("good folders with counts:")
# print(gwc)                             # ascending by zero count




input_dir = "/home/jordanprescott/shiv_research/tnt_data/rgbd_scenenet/train/0"
good_folders = [898, 657, 742, 809, 830, 604, 117, 797, 259, 47, 711, 4, 413, 450, 345, 792, 18, 555, 614, 70, 818, 3, 511, 275, 583, 182, 696, 459, 481, 980, 772, 966, 483, 641, 389, 330, 49, 437, 286]
for folder in sorted(good_folders[:10]):
    print(folder)
    make_gif(os.path.join(input_dir, str(folder), "photo"), os.path.join(input_dir, str(folder), "photo.gif"))
    make_depth_gif(os.path.join(input_dir, str(folder), "da2"), os.path.join(input_dir, str(folder), "da2.gif"))