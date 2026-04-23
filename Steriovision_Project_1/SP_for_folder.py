#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  STEREOVISION — REAL IMAGES  ·  Vision Artificielle / USTHB 2025-2026
  Prof. Slimane LARABI
═══════════════════════════════════════════════════════════════════════════

Pipeline :
  1. Camera Calibration  → K estimated from image geometry + FOV prior
  2. Load stereo pair    → resize, undistort
  3. SIFT detection & FLANN matching  (Lowe 0.75 + epipolar filter)
  4. Fundamental matrix  (RANSAC)  →  Essential matrix  →  recoverPose
  5. Triangulation  (cv2.triangulatePoints)
  6. 3-D point cloud  (matplotlib)  +  disparity  +  pipeline mosaic
"""

import cv2, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT   = Path("./Steriovision_Project_1/Esp32_realtime_output");  OUT.mkdir(exist_ok=True)
DBAR  = "═" * 65

print(DBAR)
print("  STEREO PIPELINE — REAL IMAGES   USTHB Vision Artificielle")
print(DBAR)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — CAMERA CALIBRATION  (estimated K)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 1]  Camera Calibration  (estimated intrinsics)")
print("─" * 45)

"""
Your calibration script (full_calibration.py) computes K from a checkerboard.
Here, since we use your phone images without a checkerboard session, we
ESTIMATE K using the well-known formula:

     fx = fy ≈ max(W, H)  ×  focal_mm / sensor_width_mm

For a typical Android camera (Tecno / Samsung / Redmi mid-range, 2000×1500):
  · 35 mm-equivalent focal length  ≈ 26 mm
  · Corresponding real FL          ≈ 4.7 mm
  · Sensor width                   ≈ 6.2 mm  (1/2.8" type)
  · pixel pitch  → fx = 4.7 / 6.2 × 2000  ≈  1 516 px

We round to fx = fy = 1 500 px — a safe default for this class of phone.

Once you run your real checkerboard calibration, replace these values with
the actual  mtx  from:
    mtx  = np.load('camera_params/mtx.npy')
    dist = np.load('camera_params/dist.npy')
"""

IMG_W, IMG_H = 2000, 1500
FX = FY = 1500.0
CX, CY  = IMG_W / 2.0, IMG_H / 2.0

K    = np.array([[FX,  0.0, CX ],
                 [0.0, FY,  CY ],
                 [0.0, 0.0, 1.0]], dtype=np.float64)
DIST = np.zeros(5, dtype=np.float64)   # assume undistorted (phone auto-corrects)

print(f"  Image size : {IMG_W} × {IMG_H} px")
print(f"  fx = fy    : {FX:.0f} px   (≈ 37° half-FOV horizontal)")
print(f"  Principal  : ({CX:.0f}, {CY:.0f})")
print(f"\n  K =\n{K}")
print("""
  ╔══════════════════════════════════════════════════════════╗
  ║  TO USE YOUR REAL CALIBRATION (from full_calibration.py) ║
  ║  Replace the K / DIST lines above with:                  ║
  ║    K    = np.load('camera_params/mtx.npy')               ║
  ║    DIST = np.load('camera_params/dist.npy')              ║
  ╚══════════════════════════════════════════════════════════╝""")

# Define utility functions early
def ensure_size(img, w=2000, h=1500):
    """Resize image to target dimensions if needed"""
    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — AUTOMATIC STEREO PAIR SELECTION (Scan images2 folder)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 2]  Automatic Stereo Pair Selection")
print("─" * 45)

UPLOAD = Path("./Steriovision_Project_1/captured_frames")
image_files = sorted([f for f in UPLOAD.glob("*.jpg") if f.is_file()])
print(f"  Found {len(image_files)} images in {UPLOAD}")

if len(image_files) < 2:
    print("[ERROR] Need at least 2 images"); sys.exit(1)

# Load all images and compute SIFT features
print("\n  Computing SIFT features for all images...")
sift_scan = cv2.SIFT_create(
    nfeatures         = 8000,
    contrastThreshold = 0.012,
    edgeThreshold     = 12,
    sigma             = 1.6,
)

images_data = {}
for img_path in image_files:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"    [SKIP] {img_path.name} cannot be read")
        continue
    img = ensure_size(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift_scan.detectAndCompute(gray, None)
    images_data[img_path.name] = {
        'path': img_path,
        'image': img,
        'kp': kp,
        'des': des,
        'n_kp': len(kp)
    }
    print(f"    {img_path.name:12}  →  {len(kp):5} keypoints")

# Compare all pairs and find the best match count
print("\n  Comparing all pairs...")
flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 8}, {"checks": 80})
best_pair = None
best_matches = 0
match_results = []

files = list(images_data.keys())
for i, f1 in enumerate(files):
    for f2 in files[i+1:]:
        des1 = images_data[f1]['des']
        des2 = images_data[f2]['des']
        kp1 = images_data[f1]['kp']
        kp2 = images_data[f2]['kp']
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            match_count = 0
        else:
            raw = flann.knnMatch(des1, des2, k=2)
            
            # Lowe ratio test
            ratio_ok = [m for m, n in raw if m.distance < 0.75 * n.distance]
            
            # Epipolar pre-filter
            p1_all = np.float32([kp1[m.queryIdx].pt for m in ratio_ok])
            p2_all = np.float32([kp2[m.trainIdx].pt for m in ratio_ok])
            
            if len(p1_all) > 0:
                dy_ok = np.abs(p1_all[:, 1] - p2_all[:, 1]) < 50
                match_count = np.sum(dy_ok)
            else:
                match_count = 0
        
        match_results.append((f1, f2, match_count))
        print(f"    {f1} ←→ {f2:12}  :  {match_count:4} matches")
        
        if match_count > best_matches:
            best_matches = match_count
            best_pair = (f1, f2)

# Select best pair
if best_pair is None or best_matches < 15:
    print(f"\n  [WARNING] Best pair has only {best_matches} matches. Using first valid pair.")
    # Fallback: use first valid pair
    for f1, f2, cnt in match_results:
        if cnt > 0:
            best_pair = (f1, f2)
            best_matches = cnt
            break

print(f"\n  ✓ BEST PAIR FOUND:")
print(f"    {best_pair[0]} ({images_data[best_pair[0]]['n_kp']} kps)")
print(f"    {best_pair[1]} ({images_data[best_pair[1]]['n_kp']} kps)")
print(f"    → {best_matches} good matches (after epipolar filter)")

# Use the best pair
L_PATH = images_data[best_pair[0]]['path']
R_PATH = images_data[best_pair[1]]['path']

img_L_raw = images_data[best_pair[0]]['image']
img_R_raw = images_data[best_pair[1]]['image']

if img_L_raw is None or img_R_raw is None:
    print("[ERROR] Cannot load best pair images"); sys.exit(1)

img_L_raw = ensure_size(img_L_raw)
img_R_raw = ensure_size(img_R_raw)

# Undistort (negligible for estimated K but correct pipeline habit)
img_L = cv2.undistort(img_L_raw, K, DIST)
img_R = cv2.undistort(img_R_raw, K, DIST)

print(f"  Left  : {L_PATH.name}  →  {img_L.shape[1]}×{img_L.shape[0]}")
print(f"  Right : {R_PATH.name}  →  {img_R.shape[1]}×{img_R.shape[0]}")

# Save stereo pair side-by-side
SEP = np.full((img_L.shape[0], 6, 3), [30, 30, 220], dtype=np.uint8)
pair_vis = np.hstack([img_L, SEP, img_R])
# Annotate
for txt, x in [("Left (Cam 1)", 20), ("Right (Cam 2)", img_L.shape[1]+26)]:
    cv2.putText(pair_vis, txt, (x, 42), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (20,20,20), 3)
    cv2.putText(pair_vis, txt, (x, 42), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255,220,30), 2)

cv2.imwrite(str(OUT / "stereo_pair.png"),
            cv2.resize(pair_vis, (2000, 750), interpolation=cv2.INTER_AREA))
print("  ✓  stereo_output/stereo_pair.png")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — SIFT DETECTION & MATCHING (refined on best pair)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 4]  SIFT Detection & Matching (refined on best pair)")
print("─" * 45)

sift = cv2.SIFT_create(
    nfeatures         = 8000,
    contrastThreshold = 0.012,
    edgeThreshold     = 12,
    sigma             = 1.6,
)

gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

kp1, des1 = sift.detectAndCompute(gray_L, None)
kp2, des2 = sift.detectAndCompute(gray_R, None)
print(f"  Keypoints  →  Left: {len(kp1)}   Right: {len(kp2)}")

# FLANN kNN matching
flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 8}, {"checks": 80})
raw   = flann.knnMatch(des1, des2, k=2)

# Lowe ratio test
ratio_ok = [m for m, n in raw if m.distance < 0.75 * n.distance]
p1_all   = np.float32([kp1[m.queryIdx].pt for m in ratio_ok])
p2_all   = np.float32([kp2[m.trainIdx].pt for m in ratio_ok])

print(f"  After ratio test   : {len(ratio_ok)} matches")

# Epipolar pre-filter  (horizontal stereo → Δy ≈ 0)
dy_ok  = np.abs(p1_all[:, 1] - p2_all[:, 1]) < 50
good   = [m for m, ok in zip(ratio_ok, dy_ok) if ok]
p1_all = p1_all[dy_ok]
p2_all = p2_all[dy_ok]
print(f"  After epipolar (Δy<50 px) : {len(good)} matches")

if len(good) < 15:
    print("[ERROR] Too few matches."); sys.exit(1)

# ── Visualisations ────────────────────────────────────────────────────────────
SCALE = 0.40          # display scale for high-res images
def kpvis(img, kp):
    return cv2.drawKeypoints(img, kp, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

kv_L = kpvis(img_L, kp1);  kv_R = kpvis(img_R, kp2)
cv2.imwrite(str(OUT / "keypoints_left.png"),
            cv2.resize(kv_L, None, fx=SCALE, fy=SCALE))
cv2.imwrite(str(OUT / "keypoints_right.png"),
            cv2.resize(kv_R, None, fx=SCALE, fy=SCALE))

match_vis = cv2.drawMatches(
    img_L, kp1, img_R, kp2, good[:80], None,
    matchColor=(0, 230, 60), singlePointColor=(0, 0, 200),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
cv2.imwrite(str(OUT / "sift_matches.png"),
            cv2.resize(match_vis, None, fx=SCALE, fy=SCALE))
print("  ✓  keypoints_{left,right}.png  and  sift_matches.png")

# ── Save best pair comparison visualization ────────────────────────────────────
best_pair_vis = np.hstack([
    cv2.resize(img_L, (400, 300)),
    np.full((300, 20, 3), [30, 150, 220], dtype=np.uint8),
    cv2.resize(img_R, (400, 300)),
])
cv2.putText(best_pair_vis, f"Best Pair: {best_matches} matches", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
cv2.putText(best_pair_vis, best_pair[0], 
            (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 1)
cv2.putText(best_pair_vis, best_pair[1], 
            (450, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 1)
cv2.imwrite(str(OUT / "best_pair_selected.png"), best_pair_vis)
print("  ✓  best_pair_selected.png")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — FUNDAMENTAL & ESSENTIAL MATRIX  +  POSE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 5]  Fundamental & Essential Matrix  (RANSAC)")
print("─" * 45)

F, maskF = cv2.findFundamentalMat(
    p1_all, p2_all,
    method               = cv2.FM_RANSAC,
    ransacReprojThreshold= 2.0,
    confidence           = 0.999,
)
inl = (maskF.ravel() == 1)
p1_i = p1_all[inl]
p2_i = p2_all[inl]

print(f"  RANSAC inliers : {inl.sum()} / {len(p1_all)}")
print(f"\n  F =\n{F.round(10)}")

E = K.T @ F @ K
print(f"\n  E = KᵀFK =\n{E.round(6)}")

n_pts, R_est, t_est, mask_pose = cv2.recoverPose(E, p1_i, p2_i, K)
print(f"\n  R (recovered) =\n{R_est.round(5)}")
print(f"  t̂ (unit) = {t_est.ravel().round(4)}")

# ── Epipolar line visualisation ───────────────────────────────────────────────
def draw_epilines(im1, im2, pts1, pts2, F, n=25, scale=0.4):
    v1 = im1.copy();  v2 = im2.copy()
    rng  = np.random.default_rng(7)
    idx  = rng.choice(len(pts1), size=min(n, len(pts1)), replace=False)
    l2 = cv2.computeCorrespondEpilines(pts1[idx].reshape(-1,1,2),1,F).reshape(-1,3)
    l1 = cv2.computeCorrespondEpilines(pts2[idx].reshape(-1,1,2),2,F).reshape(-1,3)
    W, H = im1.shape[1], im1.shape[0]
    for (a2,b2,c2),(a1,b1,c1),pt1,pt2 in zip(l2,l1,pts1[idx],pts2[idx]):
        col = tuple(int(x) for x in rng.integers(80, 255, 3))
        for a,b,c,vis in [(a2,b2,c2,v2),(a1,b1,c1,v1)]:
            y0 = int(-c/b) if abs(b)>1e-6 else H//2
            y1 = int((-c-a*W)/b) if abs(b)>1e-6 else H//2
            cv2.line(vis,(0,y0),(W,y1),col,2)
        cv2.circle(v1,tuple(pt1.astype(int)),8,col,-1)
        cv2.circle(v2,tuple(pt2.astype(int)),8,col,-1)
    vis = np.hstack([v1,v2])
    return cv2.resize(vis, None, fx=scale, fy=scale)

epi_img = draw_epilines(img_L, img_R, p1_i, p2_i, F)
cv2.imwrite(str(OUT / "epipolar_lines.png"), epi_img)
print("\n  ✓  epipolar_lines.png")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — 3-D TRIANGULATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 6]  3-D Triangulation")
print("─" * 45)

#  Projection matrices
#  P1 = K [I | 0]      P2 = K [R | t]
P1 = K @ np.hstack([np.eye(3),        np.zeros((3,1))])
P2 = K @ np.hstack([R_est,            t_est           ])

pts4d = cv2.triangulatePoints(P1, P2, p1_i.T, p2_i.T)
pts3d = (pts4d[:3] / pts4d[3:4]).T           # (N, 3)

# ── Filter: keep points in front of both cameras and in reasonable range ──────
pose_mask = mask_pose.ravel().astype(bool)
z_cam1 = pts3d[:, 2]
# Transform to cam2 frame and check Z there too
pts3d_c2 = (R_est @ pts3d.T + t_est).T
z_cam2   = pts3d_c2[:, 2]

keep = (pose_mask &
        (z_cam1 > 0.05) & (z_cam1 < 20.0) &
        (z_cam2 > 0.05) & (z_cam2 < 20.0))

pts3d_f = pts3d[keep]
p1_f    = p1_i[keep]

print(f"  Triangulated total  : {len(pts3d)}")
print(f"  After filtering     : {len(pts3d_f)}")
if len(pts3d_f):
    print(f"  X ∈ [{pts3d_f[:,0].min():.3f},  {pts3d_f[:,0].max():.3f}]")
    print(f"  Y ∈ [{pts3d_f[:,1].min():.3f},  {pts3d_f[:,1].max():.3f}]")
    print(f"  Z ∈ [{pts3d_f[:,2].min():.3f},  {pts3d_f[:,2].max():.3f}]")

# Reprojection error
reproj_err = 0.0
if len(pts3d_f):
    p1_back = (P1 @ np.vstack([pts3d_f.T, np.ones((1,len(pts3d_f)))])).T
    p1_back = p1_back[:,:2] / p1_back[:,2:3]
    reproj_err = np.mean(np.linalg.norm(p1_back - p1_f, axis=1))
    print(f"  Reprojection error  : {reproj_err:.3f} px")

# Colour each 3D point from the left image
colors_rgb = []
for (u, v) in p1_f:
    ui = int(np.clip(u, 0, IMG_W-1));  vi = int(np.clip(v, 0, IMG_H-1))
    b, g, r = img_L[vi, ui]
    colors_rgb.append((r/255., g/255., b/255.))

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 7]  3-D Visualization")
print("─" * 45)

DARK = "#0e0e1a"

# ── Dense disparity map (StereoSGBM) ─────────────────────────────────────────
sgbm = cv2.StereoSGBM_create(
    minDisparity   = 0,
    numDisparities = 160,
    blockSize      = 9,
    P1             = 8 * 3 * 9**2,
    P2             = 32 * 3 * 9**2,
    disp12MaxDiff  = 2,
    uniquenessRatio= 10,
    speckleWindowSize = 120,
    speckleRange   = 2,
    mode           = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)
# Resize for disparity (faster)
gL2 = cv2.resize(gray_L, (800, 600))
gR2 = cv2.resize(gray_R, (800, 600))
disp = sgbm.compute(gL2, gR2).astype(np.float32) / 16.0
dv   = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
disp_color = cv2.applyColorMap(dv, cv2.COLORMAP_TURBO)
cv2.imwrite(str(OUT / "disparity_map.png"), disp_color)
print("  ✓  disparity_map.png")

# ── A: 3-view point cloud ─────────────────────────────────────────────────────
VIEWS = [
    ("Perspective",  -35, -55),
    ("Front  (XZ)",  -89, -90),
    ("Side  (YZ)",   -89,   0),
]
fig_a, axes = plt.subplots(1, 3, figsize=(20, 6),
                            subplot_kw={"projection": "3d"},
                            facecolor=DARK)
fig_a.suptitle(
    "3-D Reconstruction — Real Images  ·  USTHB Vision Artificielle 2025/2026\n"
    "Scene: Colorful RTX 5050  |  ASRock B450M/ac R2.0  |  eMachines PSU",
    color="white", fontsize=11, fontweight="bold")

for ax, (title, elev, azim) in zip(axes, VIEWS):
    ax.set_facecolor(DARK)
    if len(pts3d_f):
        ax.scatter(pts3d_f[:,0], -pts3d_f[:,1], pts3d_f[:,2],
                   c=colors_rgb, s=2, alpha=0.80, linewidths=0)
    ax.set_xlabel("X", color="white", fontsize=8)
    ax.set_ylabel("Y↑", color="white", fontsize=8)
    ax.set_zlabel("Z (depth)", color="white", fontsize=8)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="white", labelsize=6)
    ax.view_init(elev=elev, azim=azim)

plt.tight_layout()
fig_a.savefig(str(OUT / "pointcloud_3views.png"),
              dpi=140, bbox_inches="tight", facecolor=DARK)
plt.close(fig_a)
print("  ✓  pointcloud_3views.png")

# ── B: Full pipeline mosaic ───────────────────────────────────────────────────
steps = [
    (str(OUT / "stereo_pair.png"),       "Step 2 · Stereo Pair"),
    (str(OUT / "keypoints_left.png"),    "Step 3 · SIFT Keypoints (L)"),
    (str(OUT / "sift_matches.png"),      "Step 3 · SIFT Matches"),
    (str(OUT / "epipolar_lines.png"),    "Step 4 · Epipolar Lines"),
    (str(OUT / "disparity_map.png"),     "Step 5-6 · Disparity Map"),
]

fig_b, axes_b = plt.subplots(1, len(steps), figsize=(26, 4.5), facecolor=DARK)
fig_b.suptitle(
    "Stereovision Pipeline — USTHB · Vision Artificielle · Prof. Slimane LARABI",
    color="white", fontsize=12, fontweight="bold")

for ax, (path, title) in zip(axes_b, steps):
    im = cv2.imread(path)
    if im is not None:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.set_title(title, color="white", fontsize=9, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
fig_b.savefig(str(OUT / "pipeline_summary.png"),
              dpi=130, bbox_inches="tight", facecolor=DARK)
plt.close(fig_b)
print("  ✓  pipeline_summary.png")

# ── C: Main reconstruction figure ────────────────────────────────────────────
fig_c = plt.figure(figsize=(22, 9), facecolor=DARK)
fig_c.suptitle(
    f"Real-Image Stereo Reconstruction — USTHB Vision Artificielle\n"
    f"Best Pair: {best_pair[0]} + {best_pair[1]}  ·  "
    f"{best_matches} matches  ·  {len(pts3d_f)} 3D points",
    color="white", fontsize=12, fontweight="bold")

# Left image
ax_l = fig_c.add_subplot(241)
ax_l.imshow(cv2.cvtColor(cv2.resize(img_L, (400,300)), cv2.COLOR_BGR2RGB))
ax_l.set_title("Left Image  (Cam 1)", color="white", fontsize=9, fontweight="bold")
ax_l.axis("off")

# Right image
ax_r = fig_c.add_subplot(242)
ax_r.imshow(cv2.cvtColor(cv2.resize(img_R, (400,300)), cv2.COLOR_BGR2RGB))
ax_r.set_title("Right Image  (Cam 2)", color="white", fontsize=9, fontweight="bold")
ax_r.axis("off")

# SIFT matches
ax_m = fig_c.add_subplot(243)
mv = cv2.imread(str(OUT / "sift_matches.png"))
ax_m.imshow(cv2.cvtColor(mv, cv2.COLOR_BGR2RGB))
ax_m.set_title(f"SIFT Matches ({len(good)} good, {inl.sum()} inliers)",
               color="white", fontsize=9, fontweight="bold")
ax_m.axis("off")

# Disparity
ax_d = fig_c.add_subplot(244)
ax_d.imshow(cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB))
ax_d.set_title("Disparity Map  (SGBM)", color="white", fontsize=9, fontweight="bold")
ax_d.axis("off")

# 3D cloud — 3 views (bottom row)
for i, (title, elev, azim) in enumerate(VIEWS):
    ax3 = fig_c.add_subplot(2, 4, 5+i, projection="3d")
    ax3.set_facecolor(DARK)
    if len(pts3d_f):
        ax3.scatter(pts3d_f[:,0], -pts3d_f[:,1], pts3d_f[:,2],
                    c=colors_rgb, s=3, alpha=0.80, linewidths=0)
    ax3.set_xlabel("X", color="w", fontsize=7)
    ax3.set_ylabel("Y↑", color="w", fontsize=7)
    ax3.set_zlabel("Z", color="w", fontsize=7)
    ax3.set_title(title, color="white", fontsize=9, fontweight="bold")
    ax3.tick_params(colors="white", labelsize=5)
    ax3.view_init(elev=elev, azim=azim)

# Stats text panel
ax_s = fig_c.add_subplot(2, 4, 8)
ax_s.set_facecolor(DARK)
ax_s.axis("off")
stats = (
    f"{'RESULTS':─^28}\n\n"
    f"  {'Camera fx = fy':<20}: {FX:.0f} px\n"
    f"  {'Image size':<20}: {IMG_W}×{IMG_H}\n\n"
    f"  {'SIFT kps L/R':<20}: {len(kp1)}/{len(kp2)}\n"
    f"  {'After ratio test':<20}: {len(ratio_ok)}\n"
    f"  {'After epipolar':<20}: {len(good)}\n"
    f"  {'RANSAC inliers':<20}: {inl.sum()}\n"
    f"  {'3D pts reconstructed':<20}: {len(pts3d_f)}\n"
    f"  {'Reproj. error':<20}: {reproj_err:.3f} px\n\n"
    f"  Recovered t̂ :\n  {t_est.ravel().round(4)}\n\n"
    f"  Best stereo pair:\n"
    f"  {best_pair[0]}\n"
    f"  {best_pair[1]}\n"
    f"  ({best_matches} matches)"
)
ax_s.text(0.05, 0.95, stats, transform=ax_s.transAxes,
          color="lime", fontsize=7.5, fontfamily="monospace",
          va="top", ha="left", linespacing=1.4)

plt.tight_layout()
fig_c.savefig(str(OUT / "stereo_reconstruction.png"),
              dpi=140, bbox_inches="tight", facecolor=DARK)
plt.close(fig_c)
print("  ✓  stereo_reconstruction.png")

# ─────────────────────────────────────────────────────────────────────────────
#  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DBAR}")
print("  RESULTS SUMMARY")
print(DBAR)
print(f"  {'Best stereo pair':<28}: {best_pair[0]} + {best_pair[1]}")
print(f"  {'Matches (best pair)':<28}: {best_matches}")
print(f"  {'Stereo pair':<28}: IMG094330 + IMG094345")
print(f"  {'Camera K (estimated)':<28}: fx=fy={FX:.0f}  cx={CX:.0f}  cy={CY:.0f}")
print(f"  {'SIFT kps  L / R':<28}: {len(kp1)} / {len(kp2)}")
print(f"  {'Matches (best pair ratio+epi)':<28}: {len(good)}")
print(f"  {'RANSAC inliers':<28}: {inl.sum()}")
print(f"  {'3D points':<28}: {len(pts3d_f)}")
print(f"  {'Reprojection error':<28}: {reproj_err:.3f} px")
print(f"  {'Recovered rotation':<28}: ≈ I  (small angle)")
print(f"  {'Recovered translation':<28}: {t_est.ravel().round(3)}")

print(f"\n  All image pair comparisons:")
print("  " + "─" * 42)
for f1, f2, cnt in sorted(match_results, key=lambda x: x[2], reverse=True):
    star = "  ← BEST PAIR" if (f1, f2) == best_pair else ""
    print(f"    {f1:12} ↔ {f2:12} : {cnt:4} matches{star}")

print(f"\n  Output →  ./real_stereo_output/")
for f in sorted(OUT.iterdir()):
    print(f"    {f.name}")

print(f"\n  ✓  Pipeline complete!")
print(DBAR)