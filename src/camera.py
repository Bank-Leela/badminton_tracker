"""Phase 5, part 2: the full camera, for the shuttle's height.

The court homography fixes everything on the floor but, for a camera looking
straight down the court, not the focal length: the single-plane constraints
are degenerate (on World Champs 2025 they gave 461 px and 3043 px). The net
supplies the missing height: its white tape runs between posts on the
doubles sidelines at 1.55 m, sagging to 1.524 m at the centre. For each
candidate focal length the pose follows from the homography; the tape is
projected onto the empty-court background, and the focal length whose tape
lands on white wins. Court lines are erased first (the homography says
exactly where they are), and only poses a broadcast camera could have —
above the floor, behind the near baseline — are tried.

Camera frame: court x right, y away from the camera, z up (metres), as in
`court.py`. Principal point at the image centre, square pixels, no lens
distortion. The camera is static, so one focal length serves the match; a
span refitted after a bump gets its own pose from its own homography.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from config import Config, cache_dir
from court import line_mask, load_homographies, painted_segments, project

NET_HEIGHT_POST = 1.55
NET_HEIGHT_CENTRE = 1.524
POST_X = 3.05  # posts stand on the doubles sidelines


class CameraCheckError(AssertionError):
    """The camera could not be calibrated (raised explicitly, like `court.CourtCheckError`)."""


def net_tape(n: int = 200) -> np.ndarray:
    """3D points along the top of the net, `(n, 3)`."""
    x = np.linspace(-POST_X, POST_X, n)
    z = NET_HEIGHT_POST - (NET_HEIGHT_POST - NET_HEIGHT_CENTRE) * (1 - (x / POST_X) ** 2)
    return np.c_[x, np.zeros(n), z]


def pose_from_homography(court_to_image: np.ndarray, f: float, size: tuple[int, int]):
    """`K, R, t` with `x_img ~ K (R X + t)`, X in court metres (z up), for focal length `f`."""
    w, h = size
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ court_to_image
    lam = 2 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    if t[2] < 0:  # the court in front of the camera
        r1, r2, t = -r1, -r2, -t
    R = np.c_[r1, r2, np.cross(r1, r2)]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if (R @ np.array([0, 0, 1.0]) + t)[2] <= 0 or _up_is_down(K, R, t):
        R = R @ np.diag([1.0, 1.0, -1.0])  # z must point up, i.e. upwards in the image
    return K, R, t


def _up_is_down(K, R, t) -> bool:
    a, b = project_points(K, R, t, np.array([[0, 0, 0.0], [0, 0, 1.0]]))
    return b[1] > a[1]


def project_points(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Court-frame 3D points `(N, 3)` to image pixels `(N, 2)`."""
    p = (K @ (R @ np.asarray(X, float).T + t[:, None])).T
    return p[:, :2] / p[:, 2:3]


def camera_centre(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -R.T @ t


def net_evidence(bg: np.ndarray, image_to_court: np.ndarray, cfg_court: Config) -> np.ndarray:
    """White structure that is not a painted court line, blurred a little: where the tape can be."""
    mask = line_mask(bg, cfg_court).astype(np.uint8)
    lines = np.zeros_like(mask)
    C2I = np.linalg.inv(image_to_court)
    width = max(3, int(round(9 * bg.shape[0] / 1080)))
    for _, a, b in painted_segments():
        p, q = project(C2I, np.array([a, b]))
        cv2.line(lines, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), 1, width)
    mask[lines > 0] = 0
    return cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)


def standing_heights(K, R, t, feet: np.ndarray, heads: np.ndarray) -> np.ndarray:
    """Height above the floor of each image point `heads[i]`, on the vertical line over `feet[i]` (court m)."""
    P = K @ np.c_[R, t]
    a = np.c_[feet, np.zeros(len(feet)), np.ones(len(feet))] @ P.T  # the foot, homogeneous image point
    b = P[:, 2]  # image direction of +z
    u, v = heads[:, 0], heads[:, 1]
    A = np.c_[b[0] - u * b[2], b[1] - v * b[2]]
    B = -np.c_[a[:, 0] - u * a[:, 2], a[:, 1] - v * a[:, 2]]
    return (A * B).sum(1) / np.maximum((A * A).sum(1), 1e-12)


def fit_focal(bg: np.ndarray, image_to_court: np.ndarray, cfg_cam: Config, cfg_court: Config,
              players: dict | None = None) -> dict:
    """Focal length whose projected net tape lands on white. Raises `CameraCheckError`.

    The tape score can peak at more than one focal length (another white
    line the tape happens to fit). `players` — standing players' feet (court
    m), head points (image) and side — break the tie: both players spend
    time at both ends, so the near and the far end's median head height
    must agree; a wrong focal length scales them apart (Malaysia Masters
    2020: 1.97 vs 1.45 m at the wrong peak). Among peaks scoring at least
    `peak_ratio` of the best, the one where they agree best wins.
    """
    h, w = bg.shape[:2]
    evidence = net_evidence(bg, image_to_court, cfg_court)
    C2I = np.linalg.inv(image_to_court)
    lo, hi = cfg_cam.focal_px
    z_lo, z_hi = cfg_cam.height_m
    tape = net_tape()

    def score(f):
        K, R, t = pose_from_homography(C2I, f, (w, h))
        c = camera_centre(R, t)
        if not (z_lo <= c[2] <= z_hi and c[1] < -6.7):
            return 0.0
        uv = project_points(K, R, t, tape)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        if ok.mean() < 0.8:
            return 0.0
        return float(evidence[uv[ok, 1].astype(int), uv[ok, 0].astype(int)].mean())

    def refine(f0, fs):
        k = int(np.argmin(np.abs(fs - f0)))
        fine = np.linspace(fs[max(0, k - 2)], fs[min(len(fs) - 1, k + 2)], 81)
        sf = np.array([score(f) for f in fine])
        return float(fine[int(np.argmax(sf))]), float(sf.max())

    def mismatch(f):
        if players is None:
            return None, None
        K, R, t = pose_from_homography(C2I, f, (w, h))
        z = standing_heights(K, R, t, players["feet"], players["heads"])
        near, far = np.median(z[players["near"]]), np.median(z[~players["near"]])
        return float(near), float(far)

    scale = h / 1080
    fs = np.exp(np.linspace(np.log(lo * scale), np.log(hi * scale), 600))
    s = np.array([score(f) for f in fs])
    if s.max() < float(cfg_cam.min_tape_score):
        raise CameraCheckError(f"net tape not found (best score {s.max():.2f} < {cfg_cam.min_tape_score})")
    is_peak = np.r_[False, (s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:]), False]
    peaks = [i for i in np.flatnonzero(is_peak) if s[i] >= max(float(cfg_cam.min_tape_score),
                                                                 float(cfg_cam.peak_ratio) * s.max())]
    cands = []
    for i in peaks:
        f, sc = refine(fs[i], fs)
        near, far = mismatch(f)
        cands.append({"f": f, "tape_score": sc, "head_near": near, "head_far": far})
    if players is not None:
        best = min(cands, key=lambda c: abs(c["head_near"] - c["head_far"]))
    else:
        best = max(cands, key=lambda c: c["tape_score"])
    K, R, t = pose_from_homography(C2I, best["f"], (w, h))
    return {**best, "tape_score": round(best["tape_score"], 3), "K": K, "R": R, "t": t,
            "peaks": [{k: (round(v, 3) if v is not None else None) for k, v in c.items()} for c in cands]}


def draw_camera(bg: np.ndarray, K, R, t) -> np.ndarray:
    """Net tape (yellow), posts (red) and a 1.8 m figure at each T (magenta): the check image."""
    out = bg.copy()
    th = max(1, round(bg.shape[0] / 540))
    uv = project_points(K, R, t, net_tape()).round().astype(int)
    cv2.polylines(out, [uv.reshape(-1, 1, 2)], False, (0, 255, 255), th)
    for x in (-POST_X, POST_X):
        a, b = project_points(K, R, t, np.array([[x, 0, 0], [x, 0, NET_HEIGHT_POST]])).round().astype(int)
        cv2.line(out, tuple(a), tuple(b), (0, 0, 255), th)
    for y in (-3.5, 3.5):
        a, b = project_points(K, R, t, np.array([[0, y, 0], [0, y, 1.8]])).round().astype(int)
        cv2.line(out, tuple(a), tuple(b), (255, 0, 255), 2 * th)
    return out


def _standing_players(out_dir: Path, rows, base: np.ndarray, n: int = 3000) -> dict | None:
    """Feet (court m), nose (image) and side of players standing still in the spans filmed with `base`."""
    f = out_dir / "players.parquet"
    if not f.exists():
        return None
    from players import KEYPOINTS, keypoints_array, load_players

    nose = KEYPOINTS.index("nose")
    p = load_players(f)
    kp = keypoints_array(p)
    spans = {int(r.segment_id) for r in rows.itertuples() if np.allclose(r.H, base)}
    keep = (p["segment_id"].isin(spans) & (p["speed"] < 0.3) & (p["foot_src"] == "ankles")
            & (kp[:, nose, 2] > 0.8)).to_numpy()
    idx = np.flatnonzero(keep)
    if len(idx) < 200 or (p["side"].to_numpy()[idx] == "near").all() or (p["side"].to_numpy()[idx] == "far").all():
        return None
    idx = idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]
    return {"feet": p[["court_x", "court_y"]].to_numpy(np.float64)[idx], "heads": kp[idx, nose, :2].astype(np.float64),
            "near": p["side"].to_numpy()[idx] == "near"}


def solve_camera(cfg: Config, match_id: str, force: bool = False) -> dict:
    """Calibrate the match's camera; write `camera.json` and `camera_check.png`."""
    out_dir = cache_dir(cfg, match_id)
    hom_file, bg_file = out_dir / "homography.json", out_dir / "court_background.png"
    for f in (hom_file, bg_file):
        if not f.exists():
            raise SystemExit(f"no {f}; run `bda court` first")
    out_file = out_dir / "camera.json"
    cfg_cam = cfg.camera
    hom = json.loads(hom_file.read_text())
    inputs = {"camera": cfg_cam.to_dict(), "homography": hom["inputs"],
              "segments": [[s["segment_id"], s["image_to_court"]] for s in hom["segments"]]}
    if out_file.exists() and not force:
        doc = json.loads(out_file.read_text())
        if doc.get("inputs") == inputs:
            return doc
    rows = load_homographies(hom_file)
    rows = rows[rows.H.notna()]
    if rows.empty:
        raise SystemExit(f"{hom_file} has no homography")
    w, h = hom["image_size"]
    # The match background was fitted with the span homography most spans share.
    k_base = int(np.argmax([sum(np.allclose(H, G) for G in rows.H) for H in rows.H]))
    base = rows.H.iloc[k_base]
    bg = cv2.imread(str(bg_file))
    fit = fit_focal(bg, base, cfg_cam, cfg.court, _standing_players(out_dir, rows, base))
    if fit["head_near"] is not None and abs(fit["head_near"] - fit["head_far"]) > float(cfg_cam.max_head_mismatch_m):
        raise CameraCheckError(f"{match_id}: near and far players' head heights disagree "
                               f"({fit['head_near']:.2f} vs {fit['head_far']:.2f} m) at every net-tape fit: "
                               f"{fit['peaks']}")
    spans = []
    for r in rows.itertuples():
        K, R, t = pose_from_homography(np.linalg.inv(r.H), fit["f"], (w, h))
        spans.append({"segment_id": int(r.segment_id), "R": R.tolist(), "t": t.tolist(),
                      "centre": camera_centre(R, t).round(3).tolist()})
    c = camera_centre(fit["R"], fit["t"])
    doc = {"match_id": match_id, "image_size": [w, h], "f": round(fit["f"], 2), "tape_score": fit["tape_score"],
           "head_near_m": fit["head_near"], "head_far_m": fit["head_far"], "peaks": fit["peaks"],
           "centre": c.round(3).tolist(), "frame": "court metres: x right, y away from the camera, z up",
           "inputs": inputs, "segments": spans}
    out_file.write_text(json.dumps(doc, indent=1))
    cv2.imwrite(str(out_dir / "camera_check.png"), draw_camera(bg, fit["K"], fit["R"], fit["t"]))
    heads = "" if fit["head_near"] is None else f"; standing nose {fit['head_near']:.2f} near / {fit['head_far']:.2f} far"
    print(f"camera: wrote {out_file} — f {fit['f']:.0f} px, camera {c[1]:+.1f} m along, {c[2]:.1f} m up "
          f"(tape score {fit['tape_score']}{heads}; {len(fit['peaks'])} tape peak(s))")
    return doc


def load_camera(path: str | Path) -> dict:
    """`camera.json` with numpy `K` and per-segment `(R, t)` under `poses`."""
    doc = json.loads(Path(path).read_text())
    w, h = doc["image_size"]
    f = doc["f"]
    doc["K"] = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    doc["poses"] = {s["segment_id"]: (np.array(s["R"]), np.array(s["t"])) for s in doc["segments"]}
    return doc
