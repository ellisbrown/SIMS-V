import random
from typing import Dict

import numpy as np
import cv2
from scipy.ndimage import binary_erosion


class Digitizer:
    def __init__(self, bins_x, bins_z, grid_spacing):
        self.bins_x = bins_x
        self.bins_z = bins_z
        self.grid_spacing = grid_spacing

    def row_col(self, x):
        if x is None:
            return None

        if isinstance(x, Dict) and "position" in x:
            x = x["position"]

        if isinstance(x, Dict):
            if "z" in x:
                x = [x["x"], x["z"]]
            elif "y" in x:
                x = [x["x"], x["y"]]

        if not isinstance(x, np.ndarray):
            x = np.array(x)

        return (
            np.digitize(x[..., 0], bins=self.bins_x),
            np.digitize(x[..., -1], bins=self.bins_z),
        )


def estimate_axis_offset(xs, grid_spacing, rtol=1e-2, atol=1e-3):
    try:
        # Estimate the input data spacing
        xsorted = np.sort(np.unique(xs))
        difs = np.sort(xsorted[1:] - xsorted[:-1])
        # First approx: inter-quartile mean
        q1 = int(np.round((len(difs) - 1) * 0.25))
        q3 = int(np.round((len(difs) - 1) * 0.75))
        # These tolerances seem to make sense when comparing distances in the order of cm (scale 1e-2)
        close_vals = np.isclose(difs, difs[q1 : q3 + 1].mean(), rtol=rtol, atol=atol)

        if len(close_vals) > q3 - q1:
            # Second approx: mean of data close to inter-quartile mean
            data_spacing = difs[close_vals].mean()
            if not np.isclose(data_spacing, grid_spacing, rtol=rtol, atol=atol):
                return grid_spacing - data_spacing / 2
            offset = (
                grid_spacing - data_spacing / 2
                if not np.isclose(data_spacing, grid_spacing, rtol=rtol, atol=atol)
                else grid_spacing / 2
            )
        else:
            # The data seem to be dispersed, so just assume them to be uniformly sampled
            offset = grid_spacing / 2
    except Exception:
        offset = grid_spacing / 2

    return offset


def locs2grids(
    locations,
    grid_spacing,
    return_digitizer=False,
    mask_locs=True,
    use_mesh_grid=False,
    rtol=1e-2,
    atol=1e-3,
):
    if len(locations) < 2:
        if not return_digitizer:
            return None, None
        else:
            return None, None, None

    if return_digitizer:
        mask_locs = False
        use_mesh_grid = True

    # Shuffle order to make each row/column contain different x/z values if using wide grid spacings
    locs = locations[:]
    random.shuffle(locs)

    xs = np.array([loc["x"] for loc in locs])
    zs = np.array([loc["z"] for loc in locs])

    # Assume the offset is the same for both axes
    offset = estimate_axis_offset(xs, grid_spacing, rtol=rtol, atol=atol)

    # make {x,z}s.min() -> 0 and {x,z}s.max() -> len(bins)
    bins_x = np.arange(start=xs.min() + offset, stop=xs.max(), step=grid_spacing)
    bins_z = np.arange(start=zs.min() + offset, stop=zs.max(), step=grid_spacing)
    # TODO should we leave headroom? currently, if we're closer to an obstacle at the edge of the scene than what the
    #  reachability grid provides, we'll plan assuming we're in safe area. also, handle those in the digitizer.

    rows = np.digitize(xs, bins=bins_x)
    cols = np.digitize(zs, bins=bins_z)

    imsize = (rows.max() + 1, cols.max() + 1)

    valid_grid = np.zeros(imsize, dtype=bool)
    valid_grid[rows, cols] = True

    if not use_mesh_grid:
        locs_grid = np.zeros(imsize + (3,), dtype=np.float32)
        locs_grid[rows, cols] = [[loc["x"], loc["y"], loc["z"]] for loc in locs]

        # Average x/z values in each row/column, and y overall
        locs_grid[:, :, 0] = locs_grid[:, :, 0].sum(
            axis=1, keepdims=True
        ) / valid_grid.sum(axis=1, keepdims=True)
        locs_grid[:, :, 1] = locs_grid[:, :, 1].sum() / valid_grid.sum()
        locs_grid[:, :, 2] = locs_grid[:, :, 2].sum(
            axis=0, keepdims=True
        ) / valid_grid.sum(axis=0, keepdims=True)

        if mask_locs:
            locs_grid *= valid_grid[..., np.newaxis]
    else:
        # make a full meshgrid with the digitizer bin centroids...
        row_centroids = np.concatenate(
            [bins_x - grid_spacing / 2, bins_x[-1:] + grid_spacing / 2]
        )
        col_centroids = np.concatenate(
            [bins_z - grid_spacing / 2, bins_z[-1:] + grid_spacing / 2]
        )
        x_full, z_full = np.meshgrid(row_centroids, col_centroids, indexing="ij")
        y_full = np.mean([loc["y"] for loc in locs]) * np.ones(
            (len(row_centroids), len(col_centroids)), dtype=row_centroids.dtype
        )
        locs_grid = np.stack((x_full, y_full, z_full), axis=2)

    if not return_digitizer:
        return valid_grid, locs_grid
    else:
        return valid_grid, locs_grid, Digitizer(bins_x, bins_z, grid_spacing)


def grids2locs(valid_grid, locs_grid):
    locs = locs_grid[np.nonzero(valid_grid)]
    return [dict(x=loc[0], y=loc[1], z=loc[2]) for loc in locs]


def upscale_grids(valid_grid, locs_grid, digitizer, factor=3, optimistic=False):
    assert factor % 2 != 0, "Only odd upscaling factors supported"

    if factor == 1:
        return valid_grid, locs_grid, digitizer

    # Avoid default constant padding in INTER_LINEAR, and provide extrapolation

    def resize_with_padding(data):
        padding = np.array([digitizer.grid_spacing])
        data = np.concatenate([data[0] - padding, data, data[-1] + padding])
        return cv2.resize(
            data.reshape(1, -1),
            (len(data) * factor, 1),
            interpolation=cv2.INTER_LINEAR,
        )[0][factor:-factor]

    # upscale dense x locations (assuming ~constant x along columns)
    row_vals = resize_with_padding(
        (locs_grid[:, :, 0] * valid_grid).sum(axis=1) / valid_grid.sum(axis=1)
    )

    # upscale dense z locations (assuming ~constant z along rows)
    col_vals = resize_with_padding(
        (locs_grid[:, :, 2] * valid_grid).sum(axis=0) / valid_grid.sum(axis=0)
    )

    # make a full meshgrid with the upscaled locations...
    x_full, z_full = np.meshgrid(row_vals, col_vals, indexing="ij")

    # upscale valid grid
    valid_grid = cv2.resize(
        valid_grid.astype(np.uint8),
        (valid_grid.shape[1] * factor, valid_grid.shape[0] * factor),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    # discard locations beyond original valid ones via erosion
    if not optimistic:
        valid_grid = binary_erosion(valid_grid, iterations=(factor - 1) // 2)

    # assume constant y
    y_grid = np.ones(valid_grid.shape, dtype=locs_grid.dtype) * np.max(
        locs_grid[:, :, 1]
    )

    # ...and stack the grids
    locs_grid = np.stack([x_full, y_grid, z_full], axis=2)

    # Remove output borders, if any
    if not optimistic:
        valid_rows = np.max(valid_grid, axis=1)
        valid_cols = np.max(valid_grid, axis=0)

        valid_grid = valid_grid[valid_rows, :]
        valid_grid = valid_grid[:, valid_cols]

        locs_grid = locs_grid[valid_rows, :, :]
        locs_grid = locs_grid[:, valid_cols, :]

    grid_spacing = digitizer.grid_spacing / factor

    # make {x,z}s.min() -> 0 and {x,z}s.max() -> len(bins)
    bins_x = np.arange(
        start=locs_grid[0, 0, 0] + grid_spacing / 2,
        stop=locs_grid[-1, 0, 0],
        step=grid_spacing,
    )
    bins_z = np.arange(
        start=locs_grid[0, 0, 2] + grid_spacing / 2,
        stop=locs_grid[0, -1, 2],
        step=grid_spacing,
    )

    return valid_grid, locs_grid, Digitizer(bins_x, bins_z, grid_spacing)
