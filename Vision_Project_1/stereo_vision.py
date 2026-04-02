#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  STEREOVISION SYSTEM  —  Vision Artificielle Project
  USTHB · Master Informatique Visuelle · 2025/2026
  Prof. Slimane LARABI
═══════════════════════════════════════════════════════════════════════════════

PIPELINE
  Step 1 · Camera Calibration      → intrinsics matrix K + distortion
  Step 2 · Stereo Image Acquisition → synthetic scene: 3 boxes, 2 views
  Step 3 · SIFT Detection & Matching
  Step 4 · Fundamental & Essential Matrix  (RANSAC)
  Step 5 · 3D Triangulation
  Step 6 · 3D Point Cloud Visualization

Scene setup
  · 3 boxes with KNOWN dimensions on a flat surface
  · Single camera translated horizontally (pure X translation) → stereo pair
  · World frame: X = right, Y = down, Z = forward (standard camera frame)
  · Camera 1 at world origin;  Camera 2 at (baseline, 0, 0)
"""

import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from pathlib import Path

# ─── Output directory ─────────────────────────────────────────────────────────
OUT = Path("stereo_output")
OUT.mkdir(exist_ok=True)

BANNER = "═" * 65

print(BANNER)
print("  STEREOVISION PIPELINE  —  Vision Artificielle / USTHB")
print(BANNER)

###############################################################################
# ░░  STEP 1 — CAMERA CALIBRATION  ░░
###############################################################################
print("\n[STEP 1]  Camera Calibration")
print("─" * 45)

IMG_W, IMG_H = 800, 600          # image resolution
FOCAL        = 900.0             # focal length in pixels  (fx = fy)
CX, CY       = IMG_W / 2.0, IMG_H / 2.0

#  Intrinsic matrix  K
K = np.array([[FOCAL, 0.0,  CX ],
              [0.0,  FOCAL, CY ],
              [0.0,  0.0,   1.0]], dtype=np.float64)

DIST = np.zeros(5, dtype=np.float64)   # no distortion (synthetic camera)

BASELINE = 0.12    # 12 cm horizontal baseline (metres)

#  Camera extrinsics
#   Cam 1 : at world origin  →  R = I,  t = [0,0,0]
#   Cam 2 : at [BASELINE,0,0] in world  →  R = I,  t = [-BASELINE,0,0]
R1 = np.eye(3, dtype=np.float64);    t1 = np.zeros((3, 1), dtype=np.float64)
R2 = np.eye(3, dtype=np.float64);    t2 = np.array([[-BASELINE], [0.0], [0.0]])

#  Projection matrices  P = K [R | t]  (3×4)
P1 = K @ np.hstack([R1, t1])
P2 = K @ np.hstack([R2, t2])

print(f"  Resolution   : {IMG_W} × {IMG_H} px")
print(f"  Focal length : {FOCAL:.0f} px   ( fx = fy )")
print(f"  Principal pt : ({CX:.0f}, {CY:.0f})")
print(f"  Baseline     : {BASELINE*100:.0f} cm")
print(f"\n  Camera matrix K :\n{K}\n")

# ── Save calibration checkerboard image (for illustration) ────────────────────
def make_checkerboard(sq=60, rows=6, cols=9):
    h = rows * sq;  w = cols * sq
    board = np.zeros((h, w), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[r*sq:(r+1)*sq, c*sq:(c+1)*sq] = 255
    img = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    cv2.putText(img, f"Calibration board  {cols}x{rows}  sq={sq}px",
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,80,200), 2)
    return img

cv2.imwrite(str(OUT / "calibration_checkerboard.png"), make_checkerboard())
print("  ✓  Calibration board → stereo_output/calibration_checkerboard.png")

###############################################################################
# ░░  STEP 2 — SCENE & STEREO IMAGE GENERATION  ░░
###############################################################################
print("\n[STEP 2]  Scene Setup & Stereo Image Acquisition")
print("─" * 45)

FLOOR_Y = 0.22    # Y coord of floor plane  (Y↓ in our world frame)

# ─── Box factory ──────────────────────────────────────────────────────────────
def make_box(cx, cz, width, height, depth, bgr_color, label=""):
    """
    Box resting on the floor.
      cx, cz   : world X and Z of box centre
      width    : extent in X  (metres)
      height   : extent in Y  (metres)   ← the tall dimension
      depth    : extent in Z  (metres)
      bgr_color: base BGR colour
    """
    cy = FLOOR_Y - height / 2.0          # centre Y so that bottom touches floor
    return {
        "label"  : label,
        "center" : np.array([cx, cy, cz], dtype=np.float64),
        "dims"   : np.array([width, height, depth], dtype=np.float64),
        "color"  : bgr_color,
    }

BOXES = [
    make_box(cx= 0.05, cz=1.55, width=0.10, height=0.15, depth=0.10,
             bgr_color=(120, 90,  65),  label="Box A  10×15×10 cm"),
    make_box(cx=-0.23, cz=1.78, width=0.20, height=0.10, depth=0.15,
             bgr_color=( 55, 130, 80),  label="Box B  20×10×15 cm"),
    make_box(cx= 0.28, cz=1.65, width=0.12, height=0.12, depth=0.12,
             bgr_color=( 85,  60,140),  label="Box C  12×12×12 cm"),
]

print("  Boxes placed in the scene:")
for b in BOXES:
    d = b["dims"] * 100
    print(f"    {b['label']:22s}  centre = {b['center'].round(3)}")

# ─── Box geometry ─────────────────────────────────────────────────────────────
def box_corners(box):
    """Return 8 world-space corners of a box."""
    cx, cy, cz = box["center"]
    dx, dy, dz = box["dims"] / 2.0
    return np.array([
        [cx-dx, cy-dy, cz-dz],   # 0  L·Top·Near
        [cx+dx, cy-dy, cz-dz],   # 1  R·Top·Near
        [cx+dx, cy+dy, cz-dz],   # 2  R·Bot·Near
        [cx-dx, cy+dy, cz-dz],   # 3  L·Bot·Near
        [cx-dx, cy-dy, cz+dz],   # 4  L·Top·Far
        [cx+dx, cy-dy, cz+dz],   # 5  R·Top·Far
        [cx+dx, cy+dy, cz+dz],   # 6  R·Bot·Far
        [cx-dx, cy+dy, cz+dz],   # 7  L·Bot·Far
    ], dtype=np.float64)

#  (vertex_indices, outward_unit_normal_in_world)
FACE_DEFS = [
    ([3, 2, 1, 0], np.array([ 0,  0, -1], dtype=np.float64)),  # Near
    ([4, 5, 6, 7], np.array([ 0,  0,  1], dtype=np.float64)),  # Far
    ([0, 4, 7, 3], np.array([-1,  0,  0], dtype=np.float64)),  # Left
    ([1, 2, 6, 5], np.array([ 1,  0,  0], dtype=np.float64)),  # Right
    ([0, 1, 5, 4], np.array([ 0, -1,  0], dtype=np.float64)),  # Top
    ([3, 7, 6, 2], np.array([ 0,  1,  0], dtype=np.float64)),  # Bottom
]

# Directional light
LIGHT = np.array([-0.4, -1.0, 0.7], dtype=np.float64)
LIGHT /= np.linalg.norm(LIGHT)

def lambertian(normal):
    """Shading factor in [0.30, 1.00]."""
    return 0.30 + 0.70 * max(0.0, float(np.dot(normal, -LIGHT)))

def shade_color(bgr, factor):
    return tuple(int(np.clip(c * factor, 0, 255)) for c in bgr)

# ─── Pre-compute 3D texture points on every face (world-space) ───────────────
# Using the SAME 3D positions for both camera views guarantees that dot textures
# appear at geometrically corresponding pixels → SIFT can match them reliably.

def face_texture_pts_3d(face_verts, n=160, seed=0):
    """
    Place n random points inside a quad face using bilinear interpolation.
    Returns: pts  (n×3), sizes (n,), color_offsets (n×3)
    """
    rng    = np.random.default_rng(seed)
    v0, v1, v2, v3 = face_verts
    u = rng.random(n);  v = rng.random(n)
    pts = (np.outer((1-u)*(1-v), v0) + np.outer(u*(1-v), v1) +
           np.outer(u * v,       v2) + np.outer((1-u)*v, v3))
    sizes = rng.integers(2, 7, n)
    dcs   = rng.integers(-60, 60, (n, 3))
    return pts, sizes, dcs

FACE_TEXTURES = {}          # (box_idx, face_idx) → (pts3d, sizes, dcs)
for bi, box in enumerate(BOXES):
    corners = box_corners(box)
    for fi, (vidx, _) in enumerate(FACE_DEFS):
        fv = corners[vidx]
        FACE_TEXTURES[(bi, fi)] = face_texture_pts_3d(fv, n=160, seed=bi * 10 + fi)

# Floor scatter dots (also in world space for consistency)
_rfl = np.random.default_rng(13)
N_FLOOR = 500
FLOOR_DOTS = np.column_stack([
    _rfl.uniform(-0.65, 0.75, N_FLOOR),
    np.full(N_FLOOR, FLOOR_Y),
    _rfl.uniform(1.0,  2.85, N_FLOOR),
])
FLOOR_SIZES = _rfl.integers(2, 5,    N_FLOOR)
FLOOR_DCS   = _rfl.integers(-30, 30, (N_FLOOR, 3))

# ─── Projection utility ───────────────────────────────────────────────────────
def project(pts_w, R, t):
    """
    Project N world points  →  (uv: N×2,  depth: N)
    Uses camera intrinsics K from the outer scope.
    """
    pts_c = (R @ pts_w.T + t).T           # (N, 3) camera-frame coords
    z     = pts_c[:, 2].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(z > 0, FOCAL * pts_c[:, 0] / z + CX, -9999.0)
        v = np.where(z > 0, FOCAL * pts_c[:, 1] / z + CY, -9999.0)
    return np.column_stack([u, v]), z

def visible(uv, dep):
    """Boolean mask: in-image and in front of camera."""
    return ((dep > 0) &
            (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) &
            (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H))

# ─── Render function ──────────────────────────────────────────────────────────
def render_scene(R_cam, t_cam, label=""):
    """Render the full scene from camera pose (R_cam, t_cam)."""
    img = np.full((IMG_H, IMG_W, 3), 188, dtype=np.uint8)

    # ── Floor grid lines ──────────────────────────────────────────────────────
    GC = (148, 148, 148)
    for xi in np.arange(-0.65, 0.76, 0.10):
        pts_w = np.array([[xi, FLOOR_Y, z] for z in np.arange(1.0, 2.9, 0.03)])
        uv, dep = project(pts_w, R_cam, t_cam)
        ok  = visible(uv, dep)
        uvi = uv.astype(int)
        for k in range(len(ok) - 1):
            if ok[k] and ok[k + 1]:
                cv2.line(img, tuple(uvi[k]), tuple(uvi[k + 1]), GC, 1)

    for zi in np.arange(1.0, 2.91, 0.10):
        pts_w = np.array([[x, FLOOR_Y, zi] for x in np.arange(-0.65, 0.76, 0.03)])
        uv, dep = project(pts_w, R_cam, t_cam)
        ok  = visible(uv, dep)
        uvi = uv.astype(int)
        for k in range(len(ok) - 1):
            if ok[k] and ok[k + 1]:
                cv2.line(img, tuple(uvi[k]), tuple(uvi[k + 1]), GC, 1)

    # ── Floor texture dots ────────────────────────────────────────────────────
    uv_f, dep_f = project(FLOOR_DOTS, R_cam, t_cam)
    ok_f = visible(uv_f, dep_f)
    for i in np.where(ok_f)[0]:
        col = tuple(int(np.clip(148 + int(FLOOR_DCS[i, c]), 0, 255)) for c in range(3))
        cv2.circle(img, (int(uv_f[i, 0]), int(uv_f[i, 1])), int(FLOOR_SIZES[i]), col, -1)

    # ── Boxes — painter's algorithm (back-to-front) ───────────────────────────
    cam_origin_w = -(R_cam.T @ t_cam.squeeze())    # camera position in world

    face_draw_list = []    # (avg_depth, uv_face, shaded_col, bi, fi)

    for bi, box in enumerate(BOXES):
        corners = box_corners(box)
        for fi, (vidx, normal_w) in enumerate(FACE_DEFS):
            fv_w      = corners[vidx]                      # (4, 3)
            fc_w      = fv_w.mean(axis=0)
            to_cam    = cam_origin_w - fc_w
            if np.dot(normal_w, to_cam) < 0:              # back-face cull
                continue
            uv_face, dep_face = project(fv_w, R_cam, t_cam)
            if np.any(dep_face <= 0):
                continue
            s_col = shade_color(box["color"], lambertian(normal_w))
            face_draw_list.append((dep_face.mean(), uv_face, s_col, bi, fi))

    face_draw_list.sort(key=lambda x: x[0], reverse=True)   # farthest first

    for avg_dep, uv_face, s_col, bi, fi in face_draw_list:
        pts32 = uv_face.astype(np.int32)
        cv2.fillPoly(img, [pts32], s_col)

        # Texture dots (pre-computed in 3D)
        tex_pts, tex_sz, tex_dc = FACE_TEXTURES[(bi, fi)]
        uv_t, dep_t = project(tex_pts, R_cam, t_cam)
        ok_t = visible(uv_t, dep_t)
        for i in np.where(ok_t)[0]:
            dot_col = tuple(int(np.clip(s_col[c] + int(tex_dc[i, c]), 0, 255))
                            for c in range(3))
            cv2.circle(img, (int(uv_t[i, 0]), int(uv_t[i, 1])),
                       int(tex_sz[i]), dot_col, -1)

        cv2.polylines(img, [pts32], True, (22, 22, 22), 2)

    # ── Label overlay ─────────────────────────────────────────────────────────
    if label:
        cv2.putText(img, label, (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2)
        cv2.putText(img, label, (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 50), 1)
    return img

# ── Render the two stereo views ───────────────────────────────────────────────
print("\n  Rendering stereo images …")
img_L = render_scene(R1, t1, label="Camera 1  (Left)")
img_R = render_scene(R2, t2, label="Camera 2  (Right  +12 cm)")

cv2.imwrite(str(OUT / "img_left.png"),  img_L)
cv2.imwrite(str(OUT / "img_right.png"), img_R)

pair = np.hstack([img_L, img_R])
cv2.line(pair, (IMG_W, 0), (IMG_W, IMG_H), (30, 30, 220), 3)
cv2.imwrite(str(OUT / "stereo_pair.png"), pair)
print(f"  ✓  Stereo pair  → stereo_output/stereo_pair.png")

# Theoretical disparity at mean depth
d_mean = sum(b["center"][2] for b in BOXES) / 3
disp_px = FOCAL * BASELINE / d_mean
print(f"\n  Expected disparity : f·B/Z = {FOCAL:.0f}·{BASELINE}/"
      f"{d_mean:.2f} ≈ {disp_px:.1f} px")

###############################################################################
# ░░  STEP 3 — SIFT DETECTION & MATCHING  ░░
###############################################################################
print("\n[STEP 3]  SIFT Feature Detection & Matching")
print("─" * 45)

sift = cv2.SIFT_create(
    nfeatures        = 4000,
    contrastThreshold= 0.016,
    edgeThreshold    = 12,
    sigma            = 1.6,
)

gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

kp1, des1 = sift.detectAndCompute(gray_L, None)
kp2, des2 = sift.detectAndCompute(gray_R, None)
print(f"  SIFT keypoints  →  Left: {len(kp1)}   Right: {len(kp2)}")

if min(len(kp1), len(kp2)) < 8:
    print("  [ERROR] Too few keypoints — aborting."); sys.exit(1)

# ── FLANN kNN matching ────────────────────────────────────────────────────────
flann = cv2.FlannBasedMatcher(
    {"algorithm": 1, "trees": 5},     # KDTREE
    {"checks": 60},
)
raw = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good = [m for m, n in raw if m.distance < 0.74 * n.distance]
p1_all = np.float32([kp1[m.queryIdx].pt for m in good])
p2_all = np.float32([kp2[m.trainIdx].pt for m in good])

# Epipolar filter: pure horizontal translation → same row  (Δy ≈ 0)
epi_ok = np.abs(p1_all[:, 1] - p2_all[:, 1]) < 9.0
good   = [m for m, ok in zip(good, epi_ok) if ok]
p1_all = p1_all[epi_ok]
p2_all = p2_all[epi_ok]
print(f"  After ratio test + epipolar filter : {len(good)} matches")

# Fallback: brute-force if too few matches
if len(good) < 15:
    print("  → Fallback: Brute-Force matcher")
    bf   = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    bfm  = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:300]
    p1_all = np.float32([kp1[m.queryIdx].pt for m in bfm])
    p2_all = np.float32([kp2[m.trainIdx].pt for m in bfm])
    epi_ok = np.abs(p1_all[:, 1] - p2_all[:, 1]) < 12.0
    good   = [m for m, ok in zip(bfm, epi_ok) if ok]
    p1_all = p1_all[epi_ok]
    p2_all = p2_all[epi_ok]
    print(f"  BF matches : {len(good)}")

# ── Visualisations ────────────────────────────────────────────────────────────
kp_vis_L = cv2.drawKeypoints(img_L, kp1, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp_vis_R = cv2.drawKeypoints(img_R, kp2, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(str(OUT / "keypoints_left.png"),  kp_vis_L)
cv2.imwrite(str(OUT / "keypoints_right.png"), kp_vis_R)

match_vis = cv2.drawMatches(
    img_L, kp1, img_R, kp2, good[:60], None,
    matchColor=(0, 220, 0),
    singlePointColor=(0, 0, 200),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
cv2.imwrite(str(OUT / "sift_matches.png"), match_vis)
print(f"  ✓  Keypoint images  → stereo_output/keypoints_{{left,right}}.png")
print(f"  ✓  Match image      → stereo_output/sift_matches.png")

###############################################################################
# ░░  STEP 4 — FUNDAMENTAL & ESSENTIAL MATRIX  ░░
###############################################################################
print("\n[STEP 4]  Fundamental & Essential Matrix  (RANSAC)")
print("─" * 45)

F, maskF = cv2.findFundamentalMat(
    p1_all, p2_all,
    method    = cv2.FM_RANSAC,
    ransacReprojThreshold = 1.5,
    confidence= 0.999,
)
inl  = (maskF.ravel() == 1)
p1_i = p1_all[inl]
p2_i = p2_all[inl]

print(f"  RANSAC inliers : {inl.sum()} / {len(p1_all)}")
print(f"\n  Fundamental matrix F =\n{F.round(8)}")

# Essential matrix: E = Kᵀ F K
E = K.T @ F @ K
print(f"\n  Essential matrix E =\n{E.round(6)}")

# Decompose E → R, t
n_pts, R_est, t_est, mask_pose = cv2.recoverPose(E, p1_i, p2_i, K)
print(f"\n  Recovered rotation R ≈\n{R_est.round(5)}")
print(f"  Recovered translation t̂ ≈ {t_est.ravel().round(4)}")
print(f"  (t̂ is unit-norm; true direction: [{-1.0:.1f}, 0, 0])")

# ── Epipolar line visualisation ────────────────────────────────────────────────
def draw_epilines(img1, img2, pts1, pts2, F):
    """Draw 20 random epipolar line pairs."""
    vis1 = img1.copy();  vis2 = img2.copy()
    n    = min(20, len(pts1))
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(pts1), size=n, replace=False)
    lines2 = cv2.computeCorrespondEpilines(pts1[idx].reshape(-1,1,2), 1, F).reshape(-1,3)
    lines1 = cv2.computeCorrespondEpilines(pts2[idx].reshape(-1,1,2), 2, F).reshape(-1,3)
    for (a2,b2,c2), (a1,b1,c1), pt1, pt2 in zip(lines2, lines1, pts1[idx], pts2[idx]):
        col = tuple(int(x) for x in rng.integers(120, 255, 3))
        for a, b, c, vis in [(a2,b2,c2,vis2), (a1,b1,c1,vis1)]:
            if abs(b) > 1e-6:
                y0 = int(-c / b);  y1 = int((-c - a * IMG_W) / b)
            else:
                y0 = y1 = IMG_H // 2
            cv2.line(vis, (0, y0), (IMG_W, y1), col, 1)
        cv2.circle(vis1, tuple(pt1.astype(int)), 5, col, -1)
        cv2.circle(vis2, tuple(pt2.astype(int)), 5, col, -1)
    return np.hstack([vis1, vis2])

epi_vis = draw_epilines(img_L, img_R, p1_i, p2_i, F)
cv2.imwrite(str(OUT / "epipolar_lines.png"), epi_vis)
print(f"\n  ✓  Epipolar lines   → stereo_output/epipolar_lines.png")

###############################################################################
# ░░  STEP 5 — 3D TRIANGULATION  ░░
###############################################################################
print("\n[STEP 5]  3D Triangulation")
print("─" * 45)

pts4d = cv2.triangulatePoints(P1, P2, p1_i.T, p2_i.T)   # (4, N)
pts3d = (pts4d[:3] / pts4d[3:4]).T                       # (N, 3)

# ── Filtering ─────────────────────────────────────────────────────────────────
z = pts3d[:, 2]
keep = ((z > 0.4)              &
        (z < 6.0)              &
        (np.abs(pts3d[:, 0]) < 1.5) &
        (np.abs(pts3d[:, 1]) < 1.5))
pts3d_f = pts3d[keep]
p1_f    = p1_i[keep]

print(f"  Total triangulated : {len(pts3d)}")
print(f"  After filtering    : {len(pts3d_f)}")
if len(pts3d_f):
    print(f"  X ∈ [{pts3d_f[:,0].min():.3f},  {pts3d_f[:,0].max():.3f}] m")
    print(f"  Y ∈ [{pts3d_f[:,1].min():.3f},  {pts3d_f[:,1].max():.3f}] m")
    print(f"  Z ∈ [{pts3d_f[:,2].min():.3f},  {pts3d_f[:,2].max():.3f}] m")

# Reprojection error
pts4d_f    = cv2.triangulatePoints(P1, P2, p1_f.T, p2_i[keep].T)
pts3d_tmp  = (pts4d_f[:3] / pts4d_f[3:4]).T
proj_back1 = (P1 @ np.vstack([pts3d_tmp.T, np.ones((1, len(pts3d_tmp)))])).T
proj_back1 = proj_back1[:, :2] / proj_back1[:, 2:3]
reproj_err = np.mean(np.linalg.norm(proj_back1 - p1_f, axis=1))
print(f"  Reprojection error : {reproj_err:.3f} px")

# ── Colour each 3D point from the left image ──────────────────────────────────
colors_rgb = []
for (u, v) in p1_f:
    ui = int(np.clip(u, 0, IMG_W - 1))
    vi = int(np.clip(v, 0, IMG_H - 1))
    b, g, r = img_L[vi, ui]
    colors_rgb.append((r / 255.0, g / 255.0, b / 255.0))

###############################################################################
# ░░  STEP 6 — 3D POINT CLOUD VISUALIZATION  ░░
###############################################################################
print("\n[STEP 6]  3D Point Cloud Visualization")
print("─" * 45)

DARK_BG = "#12121e"

# ── Figure A : Full pipeline overview (4 panels) ──────────────────────────────
fig_a = plt.figure(figsize=(22, 5.5), facecolor=DARK_BG)
fig_a.suptitle("Stereovision Pipeline — USTHB · Vision Artificielle",
               color="white", fontsize=14, fontweight="bold", y=1.01)

ax1 = fig_a.add_subplot(141)
ax1.imshow(cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB))
ax1.set_title("Left Image  (Cam 1)", color="white", fontsize=10, fontweight="bold")
ax1.axis("off")

ax2 = fig_a.add_subplot(142)
ax2.imshow(cv2.cvtColor(img_R, cv2.COLOR_BGR2RGB))
ax2.set_title("Right Image  (Cam 2)", color="white", fontsize=10, fontweight="bold")
ax2.axis("off")

ax3 = fig_a.add_subplot(143)
ax3.imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB))
ax3.set_title(f"SIFT Matches  ({len(good)} shown)",
              color="white", fontsize=10, fontweight="bold")
ax3.axis("off")

ax4 = fig_a.add_subplot(144, projection="3d")
ax4.set_facecolor(DARK_BG)
if len(pts3d_f):
    ax4.scatter(pts3d_f[:,0], -pts3d_f[:,1], pts3d_f[:,2],
                c=colors_rgb, s=3, alpha=0.80, linewidths=0)
for i, b in enumerate(BOXES):
    cx, cy, cz = b["center"]
    col = tuple(c/255 for c in b["color"][::-1])
    ax4.scatter([cx], [-cy], [cz], s=160, marker="^",
                c=[col], zorder=6, label=f'{b["label"][:5]} ✓')
ax4.set_xlabel("X", color="white", fontsize=8)
ax4.set_ylabel("Y↑", color="white", fontsize=8)
ax4.set_zlabel("Z", color="white", fontsize=8)
ax4.set_title("3D Point Cloud", color="white", fontsize=10, fontweight="bold")
ax4.tick_params(colors="white")
ax4.legend(fontsize=7, facecolor="#2a2a3a", labelcolor="white", framealpha=0.8)
ax4.view_init(elev=-65, azim=-90)

plt.tight_layout()
fig_a.savefig(str(OUT / "stereo_reconstruction.png"),
              dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig_a)
print(f"  ✓  Overview figure  → stereo_output/stereo_reconstruction.png")

# ── Figure B : 3-view point cloud ─────────────────────────────────────────────
VIEWS = [
    ("Perspective View", -40, -60),
    ("Front View  (XZ)",  -89, -90),
    ("Side View  (YZ)",   -89,   0),
]

fig_b, axes = plt.subplots(1, 3, figsize=(19, 6),
                            subplot_kw={"projection": "3d"},
                            facecolor=DARK_BG)
fig_b.suptitle("3D Reconstruction — Point Cloud (Multiple Views)",
               color="white", fontsize=13, fontweight="bold")

for ax, (title, elev, azim) in zip(axes, VIEWS):
    ax.set_facecolor(DARK_BG)
    if len(pts3d_f):
        ax.scatter(pts3d_f[:,0], -pts3d_f[:,1], pts3d_f[:,2],
                   c=colors_rgb, s=4, alpha=0.85, linewidths=0)
    for i, b in enumerate(BOXES):
        cx, cy, cz = b["center"]
        col = tuple(c/255 for c in b["color"][::-1])
        ax.scatter([cx], [-cy], [cz], s=120, marker="^", c=[col], zorder=6)
        ax.text(cx, -cy, cz + 0.02, chr(65+i), color="white", fontsize=9)
    ax.set_xlabel("X (m)", color="white", fontsize=8)
    ax.set_ylabel("Y↑ (m)", color="white", fontsize=8)
    ax.set_zlabel("Z (m)", color="white", fontsize=8)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="white")
    ax.view_init(elev=elev, azim=azim)

plt.tight_layout()
fig_b.savefig(str(OUT / "pointcloud_3views.png"),
              dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig_b)
print(f"  ✓  3-view cloud     → stereo_output/pointcloud_3views.png")

# ── Figure C : Disparity map ───────────────────────────────────────────────────
# Compute a dense disparity map for visual analysis
stereo_bm = cv2.StereoSGBM_create(
    minDisparity   = 0,
    numDisparities = 96,
    blockSize      = 7,
    P1             = 8  * 3 * 7**2,
    P2             = 32 * 3 * 7**2,
    disp12MaxDiff  = 1,
    uniquenessRatio= 10,
    speckleWindowSize = 100,
    speckleRange   = 2,
)
disp = stereo_bm.compute(gray_L, gray_R).astype(np.float32) / 16.0
disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
cv2.imwrite(str(OUT / "disparity_map.png"), disp_color)
print(f"  ✓  Disparity map    → stereo_output/disparity_map.png")

# ── Figure D : Summary mosaic ─────────────────────────────────────────────────
mosaics = [
    (str(OUT / "stereo_pair.png"),          "Stereo Pair"),
    (str(OUT / "keypoints_left.png"),       "SIFT Keypoints"),
    (str(OUT / "sift_matches.png"),         "SIFT Matches"),
    (str(OUT / "epipolar_lines.png"),       "Epipolar Lines"),
    (str(OUT / "disparity_map.png"),        "Disparity Map"),
]

fig_c, axes_c = plt.subplots(1, len(mosaics), figsize=(24, 4.5), facecolor=DARK_BG)
fig_c.suptitle("Stereovision Pipeline — All Steps  (USTHB 2025/2026)",
               color="white", fontsize=13, fontweight="bold")

for ax, (path, title) in zip(axes_c, mosaics):
    img_m = cv2.imread(path)
    if img_m is not None:
        ax.imshow(cv2.cvtColor(img_m, cv2.COLOR_BGR2RGB))
    ax.set_title(title, color="white", fontsize=10, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
fig_c.savefig(str(OUT / "pipeline_summary.png"),
              dpi=130, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig_c)
print(f"  ✓  Pipeline summary → stereo_output/pipeline_summary.png")

###############################################################################
# ░░  FINAL SUMMARY  ░░
###############################################################################
print(f"\n{BANNER}")
print("  RESULTS SUMMARY")
print(BANNER)
print(f"  {'Camera K':<28}: f={FOCAL:.0f}px  cx={CX:.0f}  cy={CY:.0f}")
print(f"  {'Baseline':<28}: {BASELINE*100:.0f} cm")
print(f"  {'Expected disparity':<28}: ~{disp_px:.1f} px  at Z≈{d_mean:.2f} m")
print(f"  {'SIFT keypoints L/R':<28}: {len(kp1)} / {len(kp2)}")
print(f"  {'Good matches':<28}: {len(good)}")
print(f"  {'RANSAC inliers':<28}: {inl.sum()}")
print(f"  {'3D pts reconstructed':<28}: {len(pts3d_f)}")
print(f"  {'Mean reproj. error':<28}: {reproj_err:.3f} px")

print(f"\n  Output files  →  ./{OUT}/")
for f in sorted(OUT.iterdir()):
    print(f"    {f.name}")

print(f"\n  ✓  Pipeline complete!")
print(BANNER)
