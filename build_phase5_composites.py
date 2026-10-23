"""Step 5: build the composite-cloud inputs (no model, no OSCD image or label is read).

For every validation and test city, level (c10/c30/c60) and replicate k = 0..REPLICATES-1, draws a catalogue scene from
the split's pool and a window (src/phase5/composite.py: draw_window, using masks only), then reads that subscene's
13 TOA bands from the catalogue's subscenes.zip and stores the 20 m crop. The zip is read by HTTP range from Zenodo
(src/phase3/remote_zip.py, as for the Phase 3 cloud spectrum; zipfile verifies each member's CRC-32), or from a local copy
at datasets/s2_cloud_mask_catalogue/subscenes.zip if one exists:
    experiments/robust/phase5/composites/{city}_{level}_k{k}.npz   cloud20 (h20, w20) bool, bands20 (h20, w20, 13) float32 in [0, 1]
and experiments/robust/phase5/composite_manifest.json (pools, exclusions and flags, every draw, per-file sha256). Refuses to
overwrite the manifest. Resume: an existing composite file (from an interrupted run) is kept, not rewritten, if its cloud20
equals the redrawn window of the catalogue mask (draws use masks only, so they are reproducible), its shapes and dtypes are
right and bands20 is in [0, 1]; otherwise the build stops. Resumed files are listed in the manifest.
Network reads use a 120 s socket timeout and up to 5 attempts (the zip is reopened after a failure); zipfile's CRC-32 check is
unchanged. at_upper_clip_fraction is the fraction of stored band values equal to 1 (the clip bound), computed from the file.
City shapes come from the Phase 3 mask manifest (the OSCD label grid).

Usage:
    .venv/Scripts/python.exe build_phase5_composites.py
"""

import hashlib
import io
import json
import socket
import zipfile

import numpy as np

from src.phase3.corruption import CATALOGUE, PHASE3, cloud_channel
from src.phase3.remote_zip import open_remote_zip
from src.phase5.composite import LEVELS, PHASE5, REPLICATES, draw_window, scene_pools, upsample

SUBSCENES_URL = "https://zenodo.org/api/records/4172871/files/subscenes.zip/content"

if __name__ == "__main__":
    out = PHASE5 / "composites"
    if (PHASE5 / "composite_manifest.json").exists():
        raise SystemExit("composite manifest exists; refusing to overwrite")
    socket.setdefaulttimeout(120)
    val_pool, test_pool, excluded = scene_pools()
    out.mkdir(parents=True, exist_ok=True)
    resumed = []
    shapes = {(e["split"], e["city"]): tuple(e["shape"]) for e in json.loads((PHASE3 / "mask_manifest.json").read_text(encoding="utf-8"))}
    local = CATALOGUE / "subscenes.zip"
    zf = zipfile.ZipFile(local) if local.exists() else open_remote_zip(SUBSCENES_URL)

    def read_member(info):
        global zf
        for attempt in range(1, 6):
            try:
                return zf.read(info)  # CRC-32 checked by zipfile
            except (OSError, TimeoutError) as err:
                print(f"read failed ({type(err).__name__}: {err}); attempt {attempt} of 5", flush=True)
                if attempt == 5:
                    raise
                zf = zipfile.ZipFile(local) if local.exists() else open_remote_zip(SUBSCENES_URL)

    members = {n.rsplit("/", 1)[-1][:-4]: n for n in zf.namelist() if n.endswith(".npy")}
    entries, crcs = [], {}
    for (split, city), shape in sorted(shapes.items()):
        pool = val_pool if split == "val" else test_pool
        for level in LEVELS:
            for k in range(REPLICATES):
                e = {"split": split, **draw_window(city, level, k, shape, pool)}
                y, x = e["window_yx_20m"]
                h20, w20 = e["window_hw_20m"]
                info = zf.getinfo(members[e["scene"]])
                crcs[e["scene"]] = f"{info.CRC:08x}"
                cloud20 = cloud_channel(CATALOGUE / "masks_raw" / "masks" / f"{e['scene']}.npy")[y:y + h20, x:x + w20].astype(bool)
                assert upsample(cloud20, shape).mean() == e["fraction"]
                f = out / f"{city}_{level}_k{k}.npz"
                if f.exists():
                    z = np.load(f)
                    assert np.array_equal(z["cloud20"], cloud20) and z["bands20"].shape == (h20, w20, 13), f.name
                    assert z["bands20"].dtype == np.float32 and 0 <= z["bands20"].min() and z["bands20"].max() <= 1, f.name
                    bands20 = z["bands20"]
                    resumed.append(f.name)
                else:
                    img = np.load(io.BytesIO(read_member(info)), allow_pickle=False)
                    assert img.shape == (1022, 1022, 13) and img.dtype == np.float32, (e["scene"], img.shape, img.dtype)
                    bands20 = np.clip(img[y:y + h20, x:x + w20], 0, 1)
                    np.savez_compressed(f, cloud20=cloud20, bands20=bands20)
                e.update({f"in_{k[5:]}": e["scene"] in set(v) for k, v in excluded.items() if k.startswith("flag_")})
                e.update(file=f.name, sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                         at_upper_clip_fraction=float((bands20 == 1).mean()), resumed=f.name in resumed)
                entries.append(e)
                print(split, city, level, k, e["scene"][:40], f"fraction {e['fraction']:.3f} attempts {e['attempts']}", flush=True)
    assert not ({e["scene"] for e in entries if e["split"] == "val"} & {e["scene"] for e in entries if e["split"] == "test"})
    doc = {"rule": __doc__, "levels": LEVELS, "replicates": REPLICATES, "val_pool": val_pool, "test_pool": test_pool,
           "exclusions_and_flags": excluded, "subscenes_source": str(local) if local.exists() else SUBSCENES_URL,
           "subscenes_zip_md5_zenodo": "0ad1de0ebeaff529782f456cad2e966f", "subscene_crc32": crcs,
           "resumed_files": resumed, "entries": entries}
    (PHASE5 / "composite_manifest.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"{len(entries)} composites; val pool {len(val_pool)} scenes, test pool {len(test_pool)} scenes")
