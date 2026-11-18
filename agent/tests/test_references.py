"""Offline tests for app.references: source parsing, dedupe, downscaling, pose grids, recipes."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from app import references as refs
from app.imaging.io import decode_rgb, encode_jpeg, encode_png

COMMONS = {
    "batchcomplete": "",
    "query": {"pages": {
        "-2": {"pageid": 2, "ns": 6, "title": "File:Some Car rear.jpg", "index": 2, "imageinfo": [{
            "size": 900000, "width": 3000, "height": 2000, "mime": "image/jpeg",
            "thumburl": "https://upload.wikimedia.org/thumb/b/bb/Some_Car_rear.jpg/1024px-Some_Car_rear.jpg",
            "url": "https://upload.wikimedia.org/b/bb/Some_Car_rear.jpg",
            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Some_Car_rear.jpg",
            "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"},
                            "Artist": {"value": "<a href=\"//commons.wikimedia.org/wiki/User:X\">Jane &amp; Co</a>"}},
        }]},
        "-1": {"pageid": 1, "ns": 6, "title": "File:Some Car front.jpg", "index": 1, "imageinfo": [{
            "width": 4000, "height": 3000, "mime": "image/jpeg",
            "thumburl": "https://upload.wikimedia.org/thumb/a/aa/Some_Car_front.jpg/1024px-Some_Car_front.jpg",
            "url": "https://upload.wikimedia.org/a/aa/Some_Car_front.jpg",
            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Some_Car_front.jpg",
            "extmetadata": {"LicenseShortName": {"value": "CC BY 2.0"}, "Artist": {"value": "John"}},
        }]},
        "-3": {"pageid": 3, "ns": 6, "title": "File:Tiny.jpg", "index": 3, "imageinfo": [{
            "width": 320, "height": 200, "mime": "image/jpeg", "thumburl": "https://x/tiny.jpg", "url": "https://x/tiny.jpg",
        }]},
        "-4": {"pageid": 4, "ns": 6, "title": "File:Logo.svg", "index": 4, "imageinfo": [{
            "width": 2000, "height": 2000, "mime": "image/svg+xml", "thumburl": "https://x/logo.png", "url": "https://x/logo.svg",
        }]},
    }},
}

OPENVERSE = {
    "result_count": 3, "page_count": 1, "page_size": 20, "page": 1,
    "results": [
        {"id": "a1", "title": "Some Car", "foreign_landing_url": "https://www.flickr.com/photos/u/1",
         "url": "https://live.staticflickr.com/1/1_b.jpg", "creator": "Pat", "license": "by-nd", "license_version": "2.0",
         "provider": "flickr", "source": "flickr", "width": 1024, "height": 683,
         "thumbnail": "https://api.openverse.org/v1/images/a1/thumb/"},
        {"id": "a2", "title": "Some Car front (Commons copy)", "foreign_landing_url": "https://commons.wikimedia.org/w/index.php?curid=1",
         "url": "https://upload.wikimedia.org/a/aa/Some_Car_front.jpg", "creator": "John", "license": "by",
         "license_version": "2.0", "provider": "wikimedia", "source": "wikimedia", "width": 4000, "height": 3000,
         "thumbnail": "https://api.openverse.org/v1/images/a2/thumb/"},
        {"id": "a3", "title": "PD", "foreign_landing_url": "https://example.org/p", "url": "https://example.org/p.jpg",
         "creator": None, "license": "cc0", "license_version": "1.0", "source": "example", "width": 0, "height": 0,
         "thumbnail": "https://api.openverse.org/v1/images/a3/thumb/"},
    ],
}


def test_parse_commons_rank_metadata_and_filters():
    photos = refs.parse_commons(COMMONS)
    assert [p.title for p in photos] == ["Some Car front.jpg", "Some Car rear.jpg"]  # search rank, tiny + svg dropped
    rear = photos[1]
    assert rear.author == "Jane & Co"
    assert rear.license == "CC BY-SA 4.0"
    assert (rear.width, rear.height, rear.source) == (3000, 2000, "commons")
    assert rear.thumb_url.endswith("1024px-Some_Car_rear.jpg")
    assert rear.page_url == "https://commons.wikimedia.org/wiki/File:Some_Car_rear.jpg"


def test_parse_openverse_licenses():
    photos = refs.parse_openverse(OPENVERSE)
    assert photos[0].license == "CC BY-ND 2.0"
    assert photos[0].source == "openverse:flickr"
    assert photos[0].author == "Pat"
    assert photos[2].license == "CC0"
    assert photos[2].author == "unknown"


async def test_search_photos_merges_and_dedupes():
    calls = []

    async def fetch_json(url, params):
        calls.append((url, params))
        return COMMONS if "wikimedia" in url else OPENVERSE

    photos = await refs.search_photos("Some Car", limit=30, fetch_json=fetch_json)
    assert [c[0] for c in calls] == [refs.COMMONS_API, refs.OPENVERSE_API]
    assert "filetype:bitmap" in calls[0][1]["gsrsearch"]
    assert calls[1][1]["page_size"] <= refs.OPENVERSE_MAX_ANON
    # Openverse's Commons copy of "Some_Car_front.jpg" is a duplicate.
    assert [p.source for p in photos] == ["commons", "commons", "openverse:flickr", "openverse:example"]


async def test_search_photos_survives_a_failing_source():
    async def fetch_json(url, params):
        if "wikimedia" in url:
            raise RuntimeError("down")
        return OPENVERSE

    photos = await refs.search_photos("Some Car", limit=2, fetch_json=fetch_json)
    assert len(photos) == 2 and all(p.source.startswith("openverse") for p in photos)


async def test_fetch_thumb_downscales_to_jpeg():
    big = np.zeros((900, 1600, 3), np.uint8)
    big[:, :800] = (200, 30, 30)

    async def fetch_bytes(url):
        return encode_png(big)

    photo = refs.RefPhoto("https://x/t.png", "", "CC0", "a", 1600, 900, "commons")
    data = await refs.fetch_thumb(photo, fetch_bytes=fetch_bytes)
    assert data[:3] == b"\xff\xd8\xff"  # JPEG
    img = decode_rgb(data)
    assert max(img.shape[:2]) == 512 and img.shape[:2] == (288, 512)


def test_downscale_never_upscales():
    small = encode_jpeg(np.full((100, 200, 3), 128, np.uint8))
    assert decode_rgb(refs.downscale_jpeg(small)).shape[:2] == (100, 200)


def _rating(**kw):
    base = dict(index=0, is_the_car=True, is_marketing_quality=5, view="front34", camera_height="low",
                lens_look="tele", car_width_frac=0.6, frame_x=0.5, frame_y=0.6, sun_position="behind_camera_left",
                light_hardness="hard", light_words="low warm sun", motion="static", why="")
    base.update(kw)
    return refs.CarShotRating(**base)


def _pose(az=30.0, el=15.0):
    return refs.PoseMatch(azimuth_deg=az, elevation_deg=el, confidence=0.8, cell_index=1, mirrored=False,
                          nose_points="left")


@pytest.mark.parametrize("lens,motion,frac,focal", [
    ("wide", "static", 0.6, 35.0), ("normal", "static", 0.6, 50.0),
    ("tele", "static", 0.6, 85.0), ("tele", "panning", 0.6, 135.0), ("tele", "static", 0.3, 135.0),
])
def test_camera_recipe_focal(lens, motion, frac, focal):
    assert refs.camera_recipe(_rating(lens_look=lens, motion=motion, car_width_frac=frac), _pose())["focal_mm"] == focal


def test_camera_recipe_from_pose_and_light():
    r = refs.camera_recipe(_rating(), _pose(az=330.0, el=15.0))
    assert r["azimuth_deg"] == 330.0
    assert r["elevation_deg"] == 3.0  # a low camera caps the matched elevation
    assert r["sun_rel_deg"] == 225.0
    assert (r["car_width_frac"], r["frame_x"], r["frame_y"]) == (0.6, 0.5, 0.6)
    eye = refs.camera_recipe(_rating(camera_height="eye", sun_position="overhead"), _pose(el=15.0))
    assert eye["elevation_deg"] == 15.0 and eye["sun_rel_deg"] is None


def test_camera_recipe_without_pose_and_clamps():
    r = refs.camera_recipe(_rating(view="rear34", camera_height="high", car_width_frac=1.4, frame_x=0.0,
                                   frame_y=0.99, sun_position="backlit"), None)
    assert (r["azimuth_deg"], r["elevation_deg"], r["sun_rel_deg"]) == (145.0, 20.0, 0.0)
    assert (r["car_width_frac"], r["frame_x"], r["frame_y"]) == (0.92, 0.2, 0.8)


def test_resolve_pose_mirrors_when_nose_disagrees():
    cell = {"index": 1, "azimuth_deg": 30.0, "elevation_deg": 3.0}
    same = refs.resolve_pose(cell, "left", 0.9)
    assert (same.azimuth_deg, same.mirrored, same.confidence) == (30.0, False, 0.9)
    flipped = refs.resolve_pose(cell, "right", 0.9)
    assert (flipped.azimuth_deg, flipped.mirrored) == (330.0, True)
    assert flipped.confidence < 0.9


def _sheet(cells_n=4, w=120, h=80):
    sheet = np.full((h, cells_n * w, 3), 255, np.uint8)
    cells = []
    for i in range(cells_n):
        cv2.rectangle(sheet, (i * w + 20, 30), (i * w + 100, 70), (90, 90, 90), -1)
        cells.append({"index": i, "azimuth_deg": [0, 90, 0, 90][i], "elevation_deg": [3, 3, 15, 15][i],
                      "x": i * w, "y": 0, "w": w, "h": h})
    return encode_png(sheet), cells


def test_pose_grids_one_per_elevation():
    png, cells = _sheet()
    grids = refs.pose_grids(png, cells)
    assert len(grids) == 2
    assert "elevation 3 degrees (ids 0, 1)" in grids[0][0]
    assert decode_rgb(grids[1][1]).shape[1] == 4 * 128


class FakeVlm:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    async def parse(self, prompt, images, schema, *, label=""):
        self.calls.append((prompt, images, schema))
        return schema.model_validate(self.answer)


async def test_match_poses_one_batched_call():
    png, cells = _sheet()
    photo = encode_jpeg(np.full((300, 400, 3), 100, np.uint8))
    vlm = FakeVlm({"matches": [
        {"photo": 0, "nose_points": "left", "side_visible": "left", "camera_height": "eye", "cell": 3,
         "confidence": 0.7, "why": "side"},
        {"photo": 1, "nose_points": "right", "side_visible": "right", "camera_height": "low", "cell": 1,
         "confidence": 0.5, "why": "mirrored side"},
        {"photo": 2, "nose_points": "left", "side_visible": "left", "camera_height": "low", "cell": 99,
         "confidence": 0.5, "why": "bad cell"},
    ]})
    out = await refs.match_poses([photo, photo, photo], png, cells, vlm=vlm)
    assert len(vlm.calls) == 1 and len(vlm.calls[0][1]) == 2 + 3  # two grids + three photos
    assert (out[0].azimuth_deg, out[0].elevation_deg, out[0].mirrored) == (90.0, 15.0, False)
    assert (out[1].azimuth_deg, out[1].mirrored) == (270.0, True)
    assert out[2] is None


async def test_rate_car_and_location_fill_gaps_and_measure_orientation():
    portrait = encode_jpeg(np.full((400, 300, 3), 100, np.uint8))
    wide = encode_jpeg(np.full((300, 400, 3), 100, np.uint8))
    car = FakeVlm({"photos": [_rating(index=1).model_dump()]})
    rated = await refs.rate_car_photos([wide, wide], "Some Car", vlm=car)
    assert [r.is_the_car for r in rated] == [False, True]
    assert "Some Car" in car.calls[0][0]

    loc_answer = {"index": 0, "matches_location": True, "realistic_photo": True, "has_cars": False,
                  "has_people": False, "has_text": False, "has_road": True, "landscape": True, "score": 9, "why": ""}
    loc = FakeVlm({"photos": [loc_answer, dict(loc_answer, index=1)]})
    rated = await refs.rate_location_photos([wide, portrait], "mountain road", vlm=loc)
    assert rated[0].usable and rated[0].score == 5
    assert not rated[1].landscape and not rated[1].usable


def test_vlm_build_input_downscales_and_uses_low_detail():
    vlm = refs.Vlm(client=object(), model="m")
    big = encode_png(np.zeros((1000, 2000, 3), np.uint8))
    content = vlm.build_input("prompt", [("Photo 0:", big)])[0]["content"]
    image = content[2]
    assert image["detail"] == "low" and image["image_url"].startswith("data:image/jpeg;base64,")
    import base64

    data = base64.b64decode(image["image_url"].split(",", 1)[1])
    assert max(decode_rgb(data).shape[:2]) == 512
    json.dumps(content)  # serialisable request
