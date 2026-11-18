"""Lens geometry: camera rotation convention, rolling-shot blur direction and depth falloff."""

import numpy as np

from app.imaging.lens import Intrinsics, camera_displacement, tracking_blur, world_to_camera


def test_camera_looking_along_world_y_sees_forward_motion_as_minus_z():
    # Blender camera rotated 90 deg about X looks along +Y; travelling along +Y is "into the image".
    np.testing.assert_allclose(world_to_camera((90.0, 0.0, 0.0)) @ np.array([0.0, 1.0, 0.0]), [0, 0, -1], atol=1e-9)


def test_sideways_travel_streaks_near_pixels_horizontally_and_leaves_far_ones():
    h, w = 64, 128
    img = np.zeros((h, w, 3), np.uint8)
    img[:, 60:64] = 255  # vertical white stripe
    intr = Intrinsics(fx=100.0, cx=w / 2, cy=h / 2)
    near = tracking_blur(img, np.full((h, w), 5.0, np.float32), intr, (0.5, 0.0, 0.0))
    far = tracking_blur(img, np.full((h, w), 1.0e9, np.float32), intr, (0.5, 0.0, 0.0))
    # 0.5 m sideways at 5 m depth = 10 px of travel (+-5 px): the 4 px stripe at 60..63 covers
    # 55..68 along the row only.
    assert near[32, 56:68, 0].min() > 0 and near[32, :54, 0].max() == 0 and near[32, 70:, 0].max() == 0
    np.testing.assert_array_equal(far, img)


def test_displacement_scales_with_speed_and_shutter():
    d = camera_displacement((90.0, 0.0, 0.0), heading_deg=0.0, speed_kmh=72.0, shutter_s=0.05)
    np.testing.assert_allclose(d, [0.0, 0.0, -1.0], atol=1e-9)  # 20 m/s * 0.05 s straight ahead
