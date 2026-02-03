import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
from PIL import Image
from shapely.geometry import box as shapely_box

from extract_patches import (
    clear_all_patches,
    clear_patches,
    extract_patch,
    load_tissue_mask,
    process_all_slides,
    process_slide,
    tissue_proportion,
)


# ---------------------------------------------------------------------------
# Helpers for building test fixtures
# ---------------------------------------------------------------------------

def _make_h5(path, coords):
    """Write an HDF5 file with a 'coords' dataset."""
    with h5py.File(path, "w") as f:
        f.create_dataset("coords", data=np.array(coords, dtype=np.int64),
                         maxshape=(None, 2))


def _make_geojson(path, polygons):
    """Write a minimal GeoJSON FeatureCollection.

    polygons: list of (minx, miny, maxx, maxy) bounding boxes that become
              Polygon features.
    """
    features = []
    for i, (minx, miny, maxx, maxy) in enumerate(polygons):
        features.append({
            "type": "Feature",
            "properties": {"tissue_id": i},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [minx, miny], [maxx, miny],
                    [maxx, maxy], [minx, maxy],
                    [minx, miny],
                ]],
            },
        })
    data = {"type": "FeatureCollection", "features": features}
    with open(path, "w") as f:
        json.dump(data, f)


def _mock_slide(width=10000, height=10000, mag=40.0):
    """Return a mock that behaves like openslide.OpenSlide."""
    slide = MagicMock()
    slide.dimensions = (width, height)
    slide.properties = {"openslide.objective-power": str(mag)}

    # read_region returns an RGBA PIL image
    def _read_region(location, level, size):
        return Image.new("RGBA", size, color=(128, 64, 32, 255))

    slide.read_region.side_effect = _read_region

    # thumbnail for visualisation
    def _get_thumbnail(size):
        return Image.new("RGB", size, color=(200, 200, 200))

    slide.get_thumbnail.side_effect = _get_thumbnail
    return slide


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTissueProportion(unittest.TestCase):

    def test_fully_inside(self):
        """Patch fully inside the tissue polygon -> 1.0."""
        tissue = shapely_box(0, 0, 1000, 1000)
        prop = tissue_proportion([500, 500], read_size=100, tissue_geom=tissue)
        self.assertAlmostEqual(prop, 1.0)

    def test_fully_outside(self):
        """Patch fully outside -> 0.0."""
        tissue = shapely_box(0, 0, 100, 100)
        prop = tissue_proportion([500, 500], read_size=50, tissue_geom=tissue)
        self.assertAlmostEqual(prop, 0.0)

    def test_partial_overlap(self):
        """Patch half inside -> ~0.5."""
        # tissue covers x=[0..550], patch centred at 500 with read_size=100
        # patch box = [450..550, 450..550], tissue covers x up to 550 fully
        # but let's make it cleaner: tissue = [0..500, 0..1000]
        # patch = [450..550, 450..550] -> overlap x=[450..500]=50, full y=100
        # overlap area = 50*100 = 5000, patch area = 100*100 = 10000 -> 0.5
        tissue = shapely_box(0, 0, 500, 1000)
        prop = tissue_proportion([500, 500], read_size=100, tissue_geom=tissue)
        self.assertAlmostEqual(prop, 0.5)


class TestLoadTissueMask(unittest.TestCase):

    def test_loads_valid_geojson(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mask.geojson")
            _make_geojson(path, [(0, 0, 100, 100)])
            geom = load_tissue_mask(path)
            self.assertAlmostEqual(geom.area, 100 * 100)

    def test_multiple_polygons_unified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mask.geojson")
            _make_geojson(path, [(0, 0, 50, 50), (50, 50, 100, 100)])
            geom = load_tissue_mask(path)
            # Two non-overlapping boxes, total area = 2 * 2500
            self.assertAlmostEqual(geom.area, 5000)

    def test_empty_features_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.geojson")
            with open(path, "w") as f:
                json.dump({"type": "FeatureCollection", "features": []}, f)
            with self.assertRaises(ValueError):
                load_tissue_mask(path)


class TestExtractPatch(unittest.TestCase):

    def test_returns_correct_size_same_mag(self):
        """scale_factor=1 -> read_size == patch_size, no resize."""
        slide = _mock_slide(mag=20.0)
        patch = extract_patch(slide, [500, 500], patch_size=256, scale_factor=1.0)
        self.assertEqual(patch.size, (256, 256))
        self.assertEqual(patch.mode, "RGB")

    def test_returns_correct_size_downscale(self):
        """40x native -> 20x target, scale_factor=2 -> reads 512 then resizes to 256."""
        slide = _mock_slide(mag=40.0)
        patch = extract_patch(slide, [1000, 1000], patch_size=256, scale_factor=2.0)
        self.assertEqual(patch.size, (256, 256))
        # Verify it read a 512x512 region
        call_args = slide.read_region.call_args
        self.assertEqual(call_args[0][2], (512, 512))

    def test_clamps_to_bounds(self):
        """Coordinate near edge should clamp so we don't read outside the slide."""
        slide = _mock_slide(width=300, height=300, mag=20.0)
        patch = extract_patch(slide, [10, 10], patch_size=256, scale_factor=1.0)
        call_args = slide.read_region.call_args
        top_left = call_args[0][0]
        self.assertGreaterEqual(top_left[0], 0)
        self.assertGreaterEqual(top_left[1], 0)


class TestClearPatches(unittest.TestCase):

    def test_clear_patches_removes_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some fake patch files
            for i in range(5):
                open(os.path.join(tmpdir, f"patch_{i:06d}.png"), "w").close()
            open(os.path.join(tmpdir, "patch_locations.png"), "w").close()
            # Also a non-patch file that should NOT be removed
            open(os.path.join(tmpdir, "keep_me.txt"), "w").close()

            clear_patches(tmpdir)

            remaining = os.listdir(tmpdir)
            self.assertEqual(remaining, ["keep_me.txt"])

    def test_clear_all_patches_removes_subdirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "slide_A"))
            os.makedirs(os.path.join(tmpdir, "slide_B"))
            open(os.path.join(tmpdir, "keep.txt"), "w").close()

            clear_all_patches(tmpdir)

            remaining = os.listdir(tmpdir)
            self.assertEqual(remaining, ["keep.txt"])

    def test_clear_all_nonexistent_dir(self):
        """Should not raise on a missing directory."""
        clear_all_patches("/tmp/_nonexistent_test_dir_12345")


class TestProcessSlide(unittest.TestCase):

    @patch("extract_patches.openslide")
    def test_extracts_correct_number(self, mock_openslide):
        """process_slide should create one PNG per coordinate."""
        mock_openslide.OpenSlide.return_value = _mock_slide(mag=20.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "test_patches.h5")
            save_dir = os.path.join(tmpdir, "output")
            _make_h5(h5_path, [[500, 500], [600, 600], [700, 700]])

            process_slide(
                svs_path="dummy.svs",
                h5_path=h5_path,
                save_dir=save_dir,
                patch_size=256,
                target_mag=20.0,
                max_workers=1,
            )

            pngs = [f for f in os.listdir(save_dir) if f.endswith(".png")]
            self.assertEqual(len(pngs), 3)

    @patch("extract_patches.openslide")
    def test_num_patches_limits_output(self, mock_openslide):
        mock_openslide.OpenSlide.return_value = _mock_slide(mag=20.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "test_patches.h5")
            save_dir = os.path.join(tmpdir, "output")
            coords = [[100 * i, 100 * i] for i in range(20)]
            _make_h5(h5_path, coords)

            process_slide(
                svs_path="dummy.svs",
                h5_path=h5_path,
                save_dir=save_dir,
                patch_size=256,
                target_mag=20.0,
                max_workers=1,
                num_patches=5,
            )

            pngs = [f for f in os.listdir(save_dir) if f.endswith(".png")]
            self.assertEqual(len(pngs), 5)

    @patch("extract_patches.openslide")
    def test_tissue_filter_excludes_patches(self, mock_openslide):
        mock_openslide.OpenSlide.return_value = _mock_slide(mag=20.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "test_patches.h5")
            geojson_path = os.path.join(tmpdir, "mask.geojson")
            save_dir = os.path.join(tmpdir, "output")

            # Tissue only covers [0..600, 0..600]
            _make_geojson(geojson_path, [(0, 0, 600, 600)])
            # Two coords inside tissue, one outside
            _make_h5(h5_path, [[300, 300], [500, 500], [5000, 5000]])

            process_slide(
                svs_path="dummy.svs",
                h5_path=h5_path,
                save_dir=save_dir,
                patch_size=256,
                target_mag=20.0,
                max_workers=1,
                geojson_path=geojson_path,
                min_tissue=0.5,
            )

            pngs = [f for f in os.listdir(save_dir) if f.endswith(".png")]
            self.assertEqual(len(pngs), 2)

    @patch("extract_patches.openslide")
    def test_clear_cache_removes_old(self, mock_openslide):
        mock_openslide.OpenSlide.return_value = _mock_slide(mag=20.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "test_patches.h5")
            save_dir = os.path.join(tmpdir, "output")
            os.makedirs(save_dir)
            # Pre-existing stale patches
            for i in range(3):
                open(os.path.join(save_dir, f"patch_{i:06d}.png"), "w").close()

            _make_h5(h5_path, [[500, 500]])

            process_slide(
                svs_path="dummy.svs",
                h5_path=h5_path,
                save_dir=save_dir,
                patch_size=256,
                target_mag=20.0,
                max_workers=1,
                clear_cache=True,
            )

            pngs = [f for f in os.listdir(save_dir) if f.endswith(".png")]
            # Old 3 cleared, 1 new extracted
            self.assertEqual(len(pngs), 1)


class TestProcessAllSlides(unittest.TestCase):

    @patch("extract_patches.openslide")
    def test_batch_processes_matching_slides(self, mock_openslide):
        mock_openslide.OpenSlide.return_value = _mock_slide(mag=20.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            svs_dir = os.path.join(tmpdir, "raw")
            h5_dir = os.path.join(tmpdir, "patches")
            save_root = os.path.join(tmpdir, "output")
            os.makedirs(svs_dir)
            os.makedirs(h5_dir)

            # Create 2 slides, but only 1 has a matching H5
            open(os.path.join(svs_dir, "slide_A.svs"), "w").close()
            open(os.path.join(svs_dir, "slide_B.svs"), "w").close()
            _make_h5(os.path.join(h5_dir, "slide_A_patches.h5"),
                     [[200, 200], [400, 400]])
            # slide_B has no H5 -> should be skipped

            process_all_slides(
                svs_dir=svs_dir,
                h5_dir=h5_dir,
                geojson_dir=None,
                save_root=save_root,
                patch_size=256,
                target_mag=20.0,
                max_workers=1,
            )

            # Only slide_A should have output
            self.assertTrue(os.path.isdir(os.path.join(save_root, "slide_A")))
            self.assertFalse(os.path.isdir(os.path.join(save_root, "slide_B")))

            pngs = os.listdir(os.path.join(save_root, "slide_A"))
            self.assertEqual(len(pngs), 2)

    @patch("extract_patches.openslide")
    def test_batch_with_geojson(self, mock_openslide):
        mock_openslide.OpenSlide.return_value = _mock_slide(mag=20.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            svs_dir = os.path.join(tmpdir, "raw")
            h5_dir = os.path.join(tmpdir, "patches")
            geojson_dir = os.path.join(tmpdir, "contours")
            save_root = os.path.join(tmpdir, "output")
            os.makedirs(svs_dir)
            os.makedirs(h5_dir)
            os.makedirs(geojson_dir)

            open(os.path.join(svs_dir, "slide_A.svs"), "w").close()
            _make_h5(os.path.join(h5_dir, "slide_A_patches.h5"),
                     [[300, 300], [5000, 5000]])
            _make_geojson(os.path.join(geojson_dir, "slide_A.geojson"),
                          [(0, 0, 600, 600)])

            process_all_slides(
                svs_dir=svs_dir,
                h5_dir=h5_dir,
                geojson_dir=geojson_dir,
                save_root=save_root,
                patch_size=256,
                target_mag=20.0,
                max_workers=1,
                min_tissue=0.5,
            )

            pngs = os.listdir(os.path.join(save_root, "slide_A"))
            # Only [300,300] is inside the tissue mask
            self.assertEqual(len(pngs), 1)


if __name__ == "__main__":
    unittest.main()
