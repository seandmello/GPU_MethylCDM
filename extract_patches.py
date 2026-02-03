import argparse
import glob
import json
import os
import shutil
import h5py
import numpy as np
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from shapely.geometry import shape, box
from shapely.ops import unary_union

try:
    import openslide
except ImportError:
    raise ImportError("openslide-python is required. Install with: pip install openslide-python")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_magnification(slide):
    """Extract the native magnification from SVS metadata."""
    for key in ['openslide.objective-power', 'aperio.AppMag']:
        mag = slide.properties.get(key)
        if mag is not None:
            return float(mag)
    return None


def load_tissue_mask(geojson_path):
    """Load tissue polygons from a GeoJSON file and return a single unified geometry."""
    with open(geojson_path, "r") as f:
        data = json.load(f)
    polygons = []
    for feature in data["features"]:
        geom = shape(feature["geometry"])
        if geom.is_valid:
            polygons.append(geom)
        else:
            polygons.append(geom.buffer(0))
    if not polygons:
        raise ValueError(f"No valid tissue polygons found in {geojson_path}")
    return unary_union(polygons)


def tissue_proportion(coord, read_size, tissue_geom):
    """Compute the fraction of a patch that overlaps with the tissue mask."""
    x, y = int(coord[0]), int(coord[1])
    tl_x = x - read_size // 2
    tl_y = y - read_size // 2
    patch_box = box(tl_x, tl_y, tl_x + read_size, tl_y + read_size)
    intersection = patch_box.intersection(tissue_geom)
    return intersection.area / patch_box.area


def extract_patch(slide, coord, patch_size, scale_factor):
    """Extract a single patch centred on coord at the target magnification."""
    x, y = int(coord[0]), int(coord[1])
    read_size = int(patch_size * scale_factor)

    top_left_x = x - read_size // 2
    top_left_y = y - read_size // 2

    dims = slide.dimensions
    top_left_x = max(0, min(top_left_x, dims[0] - read_size))
    top_left_y = max(0, min(top_left_y, dims[1] - read_size))

    region = slide.read_region((top_left_x, top_left_y), 0, (read_size, read_size))
    region = region.convert("RGB")

    if read_size != patch_size:
        region = region.resize((patch_size, patch_size), Image.LANCZOS)

    return region


def create_visualisation(slide, coords, patch_size, scale_factor, save_path):
    """Save a low-res thumbnail with patch locations marked."""
    thumb_max = 2048
    thumb = slide.get_thumbnail((thumb_max, thumb_max))
    thumb_np = np.array(thumb)

    w, h = slide.dimensions
    sx = thumb_np.shape[1] / w
    sy = thumb_np.shape[0] / h

    read_size = int(patch_size * scale_factor)

    for coord in coords:
        x, y = int(coord[0]), int(coord[1])
        tl_x = int((x - read_size // 2) * sx)
        tl_y = int((y - read_size // 2) * sy)
        br_x = int((x + read_size // 2) * sx)
        br_y = int((y + read_size // 2) * sy)
        thumb_np[tl_y:tl_y+2, tl_x:br_x] = [255, 0, 0]
        thumb_np[br_y:br_y+2, tl_x:br_x] = [255, 0, 0]
        thumb_np[tl_y:br_y, tl_x:tl_x+2] = [255, 0, 0]
        thumb_np[tl_y:br_y, br_x:br_x+2] = [255, 0, 0]

    Image.fromarray(thumb_np).save(save_path)


def process_single(args_tuple):
    """Worker function for extracting and saving one patch."""
    idx, coord, slide_path, patch_size, scale_factor, save_dir = args_tuple
    slide = openslide.OpenSlide(slide_path)
    patch = extract_patch(slide, coord, patch_size, scale_factor)
    patch.save(os.path.join(save_dir, f"patch_{idx:06d}.png"))
    slide.close()
    return idx


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def clear_patches(save_dir):
    """Remove all patch PNG files from a directory."""
    removed = 0
    for f in glob.glob(os.path.join(save_dir, "patch_*.png")):
        os.remove(f)
        removed += 1
    vis = os.path.join(save_dir, "patch_locations.png")
    if os.path.exists(vis):
        os.remove(vis)
        removed += 1
    print(f"Cleared {removed} files from {save_dir}")


def clear_all_patches(save_root):
    """Remove all per-slide patch directories under save_root."""
    if not os.path.isdir(save_root):
        print(f"Nothing to clear – {save_root} does not exist")
        return
    count = 0
    for entry in os.listdir(save_root):
        entry_path = os.path.join(save_root, entry)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
            count += 1
    print(f"Removed {count} slide directories from {save_root}")


# ---------------------------------------------------------------------------
# Per-slide extraction
# ---------------------------------------------------------------------------

def process_slide(svs_path, h5_path, save_dir, patch_size, target_mag,
                  max_workers, geojson_path=None, min_tissue=0.0,
                  num_patches=None, visualize=False, skip_errors=False,
                  clear_cache=False):
    """Extract patches for a single SVS slide."""

    if clear_cache and os.path.isdir(save_dir):
        clear_patches(save_dir)

    slide = openslide.OpenSlide(svs_path)
    native_mag = get_magnification(slide)
    if native_mag is None:
        print("  Warning: Could not detect native magnification, assuming 40x")
        native_mag = 40.0
    scale_factor = native_mag / target_mag

    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
    print(f"  Loaded {len(coords)} coordinates")

    # --- Tissue mask filtering ---
    read_size = int(patch_size * scale_factor)
    if geojson_path is not None:
        tissue_geom = load_tissue_mask(geojson_path)
        keep = []
        for coord in tqdm(coords, desc="  Filtering by tissue mask", leave=False):
            if tissue_proportion(coord, read_size, tissue_geom) >= min_tissue:
                keep.append(coord)
        coords = np.array(keep) if keep else np.empty((0, 2))
        print(f"  {len(coords)} patches passed tissue filter (>= {min_tissue * 100:.0f}%)")

    # --- Random sub-sampling ---
    if num_patches is not None and num_patches < len(coords):
        indices = np.random.choice(len(coords), size=num_patches, replace=False)
        indices.sort()
        coords = coords[indices]
        print(f"  Randomly sampled {num_patches} patches")

    if len(coords) == 0:
        print("  No patches to extract – skipping")
        slide.close()
        return

    os.makedirs(save_dir, exist_ok=True)

    if visualize:
        vis_path = os.path.join(save_dir, "patch_locations.png")
        create_visualisation(slide, coords, patch_size, scale_factor, vis_path)

    slide.close()

    # --- Multi-threaded extraction ---
    tasks = [
        (i, coord, svs_path, patch_size, scale_factor, save_dir)
        for i, coord in enumerate(coords)
    ]

    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_single, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  Extracting patches", leave=False):
            try:
                future.result()
            except Exception as e:
                idx = futures[future]
                if skip_errors:
                    failed += 1
                    tqdm.write(f"  Warning: patch {idx} failed – {e}")
                else:
                    raise

    total = len(coords)
    print(f"  Extracted {total - failed}/{total} patches")


# ---------------------------------------------------------------------------
# Batch processing – loop over all slides in a folder
# ---------------------------------------------------------------------------

def process_all_slides(svs_dir, h5_dir, geojson_dir, save_root, patch_size,
                       target_mag, max_workers, min_tissue=0.0,
                       num_patches=None, visualize=False, skip_errors=False,
                       clear_cache=False):
    """Loop over every .svs in svs_dir and extract patches."""

    svs_files = sorted(glob.glob(os.path.join(svs_dir, "*.svs")))
    if not svs_files:
        print(f"No .svs files found in {svs_dir}")
        return

    print(f"Found {len(svs_files)} SVS files in {svs_dir}\n")

    processed, skipped = 0, 0

    for svs_path in svs_files:
        stem = os.path.splitext(os.path.basename(svs_path))[0]
        h5_path = os.path.join(h5_dir, f"{stem}_patches.h5")
        geojson_path = os.path.join(geojson_dir, f"{stem}.geojson") if geojson_dir else None

        if not os.path.isfile(h5_path):
            print(f"SKIP {stem}: H5 not found at {h5_path}")
            skipped += 1
            continue

        if geojson_path and not os.path.isfile(geojson_path):
            print(f"  Warning: no GeoJSON for {stem}, extracting without tissue filter")
            geojson_path = None

        slide_save_dir = os.path.join(save_root, stem)

        processed += 1
        print(f"=== [{processed}] {stem} ===")

        process_slide(
            svs_path=svs_path,
            h5_path=h5_path,
            save_dir=slide_save_dir,
            patch_size=patch_size,
            target_mag=target_mag,
            max_workers=max_workers,
            geojson_path=geojson_path,
            min_tissue=min_tissue,
            num_patches=num_patches,
            visualize=visualize,
            skip_errors=skip_errors,
            clear_cache=clear_cache,
        )
        print()

    print("========================================")
    print(f"Finished. Processed {processed}, skipped {skipped}.")
    print(f"Patches saved to {save_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract patches from SVS slides using coordinates from HDF5 files."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- extract (single slide) ----
    p_single = sub.add_parser("extract", help="Extract patches from a single SVS slide")
    p_single.add_argument("--svs", type=str, required=True)
    p_single.add_argument("--h5", type=str, required=True)
    p_single.add_argument("--save_dir", type=str, default="patches_output")
    p_single.add_argument("--patch_size", type=int, default=256)
    p_single.add_argument("--target_magnification", type=float, default=20.0)
    p_single.add_argument("--max_workers", type=int, default=8)
    p_single.add_argument("--visualize", action="store_true")
    p_single.add_argument("--skip_errors", action="store_true")
    p_single.add_argument("--num_patches", type=int, default=None)
    p_single.add_argument("--geojson", type=str, default=None)
    p_single.add_argument("--min_tissue", type=float, default=0.0)
    p_single.add_argument("--clear_cache", action="store_true")

    # ---- batch (all slides in a folder) ----
    p_batch = sub.add_parser("batch", help="Extract patches from all SVS slides in a directory")
    p_batch.add_argument("--svs_dir", type=str, required=True)
    p_batch.add_argument("--h5_dir", type=str, required=True)
    p_batch.add_argument("--geojson_dir", type=str, default=None)
    p_batch.add_argument("--save_dir", type=str, required=True)
    p_batch.add_argument("--patch_size", type=int, default=256)
    p_batch.add_argument("--target_magnification", type=float, default=20.0)
    p_batch.add_argument("--max_workers", type=int, default=8)
    p_batch.add_argument("--visualize", action="store_true")
    p_batch.add_argument("--skip_errors", action="store_true")
    p_batch.add_argument("--num_patches", type=int, default=None)
    p_batch.add_argument("--min_tissue", type=float, default=0.0)
    p_batch.add_argument("--clear_cache", action="store_true")

    # ---- clean (remove extracted patches) ----
    p_clean = sub.add_parser("clean", help="Remove extracted patches")
    p_clean.add_argument("--save_dir", type=str, required=True,
                         help="Path to a single slide patch dir or the batch save root")
    p_clean.add_argument("--all", action="store_true",
                         help="Remove all slide subdirectories (batch mode cleanup)")

    args = parser.parse_args()

    if args.command == "extract":
        process_slide(
            svs_path=args.svs,
            h5_path=args.h5,
            save_dir=args.save_dir,
            patch_size=args.patch_size,
            target_mag=args.target_magnification,
            max_workers=args.max_workers,
            geojson_path=args.geojson,
            min_tissue=args.min_tissue,
            num_patches=args.num_patches,
            visualize=args.visualize,
            skip_errors=args.skip_errors,
            clear_cache=args.clear_cache,
        )

    elif args.command == "batch":
        process_all_slides(
            svs_dir=args.svs_dir,
            h5_dir=args.h5_dir,
            geojson_dir=args.geojson_dir,
            save_root=args.save_dir,
            patch_size=args.patch_size,
            target_mag=args.target_magnification,
            max_workers=args.max_workers,
            min_tissue=args.min_tissue,
            num_patches=args.num_patches,
            visualize=args.visualize,
            skip_errors=args.skip_errors,
            clear_cache=args.clear_cache,
        )

    elif args.command == "clean":
        if args.all:
            clear_all_patches(args.save_dir)
        else:
            clear_patches(args.save_dir)


if __name__ == "__main__":
    main()
