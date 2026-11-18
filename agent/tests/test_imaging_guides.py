import numpy as np

from app.imaging.guides import depth_guide, line_guide


def test_line_guide_marks_tone_boundaries_only():
    g = np.full((100, 150, 3), 235, np.uint8)  # sky
    g[40:] = 140  # ground
    g[70:, 50:100] = 56  # road
    out = line_guide(g)
    assert out.shape == (100, 150) and out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 255}
    assert out[39:41, 10].max() == 255  # horizon
    assert out[80, 49:51].max() == 255  # road edge
    assert out[10:30].max() == 0 and out[50:65, 10:40].max() == 0  # flat regions stay black
    assert line_guide(g[..., 0]).shape == (100, 150)  # single channel accepted


def test_depth_guide_sky_black_near_bright():
    z = np.tile(np.linspace(3.0, 400.0, 80, dtype=np.float32)[::-1, None], (1, 60))  # far at top, near at bottom
    z[:20] = np.inf
    z[20:25, :30] = 1e10  # Blender's huge sky value
    z[25, 0] = np.nan
    out = depth_guide(z)
    assert out.shape == (80, 60) and out.dtype == np.uint8
    assert out[:20].max() == 0 and out[20:25, :30].max() == 0 and out[25, 0] == 0
    assert out[-1, 5] > out[40, 5] > out[26, 5] > 0  # nearer = brighter, far ground still > sky
    assert out[-1].min() == 255


def test_depth_guide_all_sky_and_flat():
    assert depth_guide(np.full((4, 4), np.inf, np.float32)).max() == 0
    flat = depth_guide(np.full((4, 4), 10.0, np.float32))
    assert flat.min() > 0
