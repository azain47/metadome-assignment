"""Download the bundled HDRI set from Poly Haven (CC0) into assets/hdri/.

    cd agent && uv run python ../scripts/fetch_hdris.py [--res 2k]

Writes assets/hdri/<id>_<res>.hdr and assets/hdri/index.json. The agent uploads both to the
object store at startup (keys hdri/...), so the demo never depends on Poly Haven being reachable.
Edit HDRIS to add or remove one: id -> descriptive tags used when picking lighting for a shot.

Each index entry also carries the measured `sun` (brightest region of the equirect image) as a
compass direction in Blender's frame with no rotation: 0 = +Y, 90 = +X. With set_lighting's
mapping, the world sun azimuth = sun.azimuth_deg + hdri_rotation_deg, so rotate by
(wanted azimuth - sun.azimuth_deg). `contrast` (peak / median luminance) below ~50 means no
usable sun (overcast).
"""

import argparse
import json
import os
import urllib.request

import cv2
import numpy as np

# id -> descriptive tags (what the environment looks like, never a target location): the agent
# picks the HDRI whose tags best match the shot's scene and time of day. Picked from a search of
# the whole Poly Haven catalogue for mountain, arid, coastal, urban and road environments at
# midday and golden hour.
HDRIS: dict[str, list[str]] = {
    "alps_field": ["mountain", "alpine", "green", "meadow", "partly_cloudy", "midday"],
    "pizzo_pernice": ["mountain", "alpine", "dry_grass", "clear", "midday"],
    "camdeboo_road": ["mountain", "road", "scrub", "clear", "golden_hour", "low_sun"],
    "orbita": ["mountain", "road", "guardrail", "green", "soft_light"],
    "kiara_5_noon": ["mountain", "arid", "barren", "rocky", "clear", "midday", "harsh_sun"],
    "kiara_7_late-afternoon": ["mountain", "arid", "barren", "rocky", "clear", "low_sun"],
    "kiara_8_sunset": ["mountain", "arid", "barren", "rocky", "clear", "golden_hour", "low_sun"],
    "goegap_road": ["arid", "desert", "road", "clear", "low_sun"],
    "spiaggia_di_mondello": ["coast", "sea", "beach", "clear", "midday", "warm"],
    "simons_town_road": ["coast", "sea", "road", "hill", "clear", "midday"],
    "venice_sunset": ["coast", "sea", "town", "stone", "golden_hour", "low_sun"],
    "victoria_sunset": ["coast", "sea", "road", "golden_hour", "low_sun"],
    "noon_grass": ["open", "green", "clear", "midday"],
    "spruit_sunrise": ["open", "golden_hour", "low_sun", "warm"],
    "cannon": ["mountain", "overcast", "soft_light"],
    "wide_street_01": ["urban", "street", "daylight"],
    "dikhololo_night": ["open", "night", "stars"],
}

API = "https://api.polyhaven.com"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "hdri")


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "trueshot"}), timeout=60) as r:
        return json.load(r)


def measure_sun(path: str) -> dict:
    """Brightest region of an equirect HDR -> {azimuth_deg, elevation_deg, contrast}.
    Blender's equirect lookup puts compass 0 (+Y) at u = 0.25 and +X (90) at u = 0.5."""
    bgr = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR).astype(np.float32)
    lum = 0.0722 * bgr[..., 0] + 0.7152 * bgr[..., 1] + 0.2126 * bgr[..., 2]
    h, w = lum.shape
    smooth = cv2.GaussianBlur(lum, (0, 0), sigmaX=max(1.0, w / 512))
    y, x = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    u, v = (x + 0.5) / w, (y + 0.5) / h
    return {
        "azimuth_deg": round(((u - 0.25) * 360.0) % 360.0, 1),
        "elevation_deg": round(90.0 - v * 180.0, 1),
        "contrast": round(float(smooth[y, x] / max(float(np.median(lum)), 1e-6)), 1),
    }


# Fetched at 4k: the ones picked for hero shots, where a mirror-like clear coat shows a 2k map's
# softness. The rest stay 2k (lighting only, or fallbacks).
HIGH_RES = {"alps_field", "pizzo_pernice", "camdeboo_road", "kiara_5_noon", "kiara_7_late-afternoon",
            "kiara_8_sunset", "spiaggia_di_mondello", "simons_town_road", "venice_sunset"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", default="2k", choices=("1k", "2k", "4k"), help="resolution outside HIGH_RES")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    index = []
    for hdri_id, tags in HDRIS.items():
        info = _get_json(f"{API}/info/{hdri_id}")
        files = _get_json(f"{API}/files/{hdri_id}")
        res = "4k" if hdri_id in HIGH_RES else args.res
        entry = files["hdri"][res]["hdr"]
        name = f"{hdri_id}_{res}.hdr"
        path = os.path.join(OUT, name)
        if not (os.path.exists(path) and os.path.getsize(path) == entry["size"]):
            print(f"downloading {name} ({entry['size'] / 1e6:.1f} MB)")
            urllib.request.urlretrieve(entry["url"], path)
        index.append({
            "id": hdri_id, "key": f"hdri/{name}", "tags": tags,
            "title": info.get("name", hdri_id), "polyhaven_categories": info.get("categories", []),
            "sun": measure_sun(path),
        })
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    print(f"{len(index)} HDRIs in {OUT}")


if __name__ == "__main__":
    main()
