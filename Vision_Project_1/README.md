# Stereovision System — Vision Artificielle
**USTHB · Master Informatique Visuelle · 2025/2026 · Prof. Slimane LARABI**

---

## Pipeline Steps

| Step | Description | Output |
|------|-------------|--------|
| 1 | **Camera Calibration** — intrinsic matrix K, distortion | `calibration_checkerboard.png` |
| 2 | **Stereo Image Acquisition** — synthetic scene (3 boxes, 2 views) | `img_left.png`, `img_right.png`, `stereo_pair.png` |
| 3 | **SIFT Detection & Matching** — Lowe ratio + epipolar filter | `keypoints_*.png`, `sift_matches.png` |
| 4 | **F & E Matrix** — RANSAC estimation, recoverPose | `epipolar_lines.png` |
| 5 | **3D Triangulation** — `cv2.triangulatePoints` with P1, P2 | 364 3D points |
| 6 | **Point Cloud Visualization** — 3 views + disparity map | `pointcloud_3views.png`, `disparity_map.png` |

---

## Scene Setup

```
World frame: X = right,  Y = down,  Z = forward

Box A : 10 × 15 × 10 cm   at Z = 1.55 m
Box B : 20 × 10 × 15 cm   at Z = 1.78 m
Box C : 12 × 12 × 12 cm   at Z = 1.65 m

Camera 1 : origin  [0, 0, 0]
Camera 2 : baseline [0.12, 0, 0]   (12 cm translation along X)
```

---

## Camera Model

```
K = [[900,   0, 400],
     [  0, 900, 300],
     [  0,   0,   1]]

P1 = K [I | 0]           (reference camera)
P2 = K [I | -B  0  0]    (B = 0.12 m)
```

---

## Key Results

- **SIFT keypoints** : ~1 200 per image
- **Good matches** : 404  (Lowe ratio 0.74 + epipolar Δy < 9 px)
- **RANSAC inliers** : 402 / 404
- **3D points** : 364
- **Reprojection error** : 0.052 px  ✓
- **Recovered t̂** : `[-1, ~0, ~0]`  (matches true translation direction)

---

## Installation & Run

```bash
pip install -r requirements.txt
python stereo_vision.py
```

All output images are saved in `./stereo_output/`.

---

## Theory

### Disparity → Depth
```
Z = f · B / d       (d = disparity in pixels)
```

### Triangulation
Given matched pixel pairs (p1, p2) and projection matrices (P1, P2):
```
[x P1[2,:] - P1[0,:]]      [0]
[y P1[2,:] - P1[1,:]]  X = [0]
[x'P2[2,:] - P2[0,:]]      [0]
[y'P2[2,:] - P2[1,:]]
```
Solved with SVD → homogeneous 3D point.

### Essential Matrix
```
E = Kᵀ F K
E = [t]× R         (t = cross-product matrix of translation)
```
Decomposed via `cv2.recoverPose` to recover R and t.
