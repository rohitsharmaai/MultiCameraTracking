# Dataset Instructions

## Source

**EPFL CVLAB — Multi-camera pedestrians video (POM dataset)**

- Official page: [https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/)
- Copyright: CVLab – EPFL (research use only)
- Please cite when publishing (see References below)

All videos are synchronized multi-camera streams of the same area from different angles. Cameras are ~2 m from the ground. Frame rate: **25 fps**.

---

## Where to place downloaded files

Put all videos, calibration files, and ground truth into the `data/` folder.
---

## 1. Laboratory sequences

Shot indoors with **4 cameras**. Four / six people enter and walk for ~2.5 minutes. Codec: MPEG-4.

### 4-people

| Camera | Download |
|--------|----------|
| Camera 0 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c0.avi |
| Camera 1 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c1.avi |
| Camera 2 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c2.avi |
| Camera 3 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c3.avi |

Save to: `data/`

### 6-people

| Camera | Download |
|--------|----------|
| Camera 0 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/6p-c0.avi |
| Camera 1 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/6p-c1.avi |
| Camera 2 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/6p-c2.avi |
| Camera 3 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/6p-c3.avi |

Save to: `data/`

Calibration (homographies): download from the official page section *Laboratory sequences* → place in `data/`

---

## 2. Campus sequences

Outdoor campus, **3 DV cameras**, up to 4 people. Codec: Indeo 5.

### Seq. 1 (`campus4`)

| Camera | Download |
|--------|----------|
| Cam 0 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/campus4-c0.avi |
| Cam 1 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/campus4-c1.avi |
| Cam 2 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/campus4-c2.avi |

Save to: `data/`

### Seq. 2 (`campus7`)

| Camera | Download |
|--------|----------|
| Cam 0 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/campus7-c0.avi |
| Cam 1 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/campus7-c1.avi |
| Cam 2 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/campus7-c2.avi |

Save to: `data/`

Calibration: from the official page *Campus sequences* → `data/`

Tsai calibration (optional): linked on the official page under *Camera calibration*.

---

## 3. Terrace sequences

Outdoor terrace, **4 DV cameras**, up to 7 people, ~3.5 minutes. Codec: Indeo 5.

### Seq. 1

| Camera | Download |
|--------|----------|
| Cam 0 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c0.avi |
| Cam 1 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c1.avi |
| Cam 2 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c2.avi |
| Cam 3 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c3.avi |

Save to: `data/`

### Seq. 2

| Camera | Download |
|--------|----------|
| Cam 0 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace2-c0.avi |
| Cam 1 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace2-c1.avi |
| Cam 2 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace2-c2.avi |
| Cam 3 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace2-c3.avi |

Save to: `data/`

Calibration: from the official page *Terrace sequences* → `data/`

Tsai calibration (optional): linked on the official page (required for terrace ground truth bounding-box alignment).

---

## 4. Passageway sequence

Underground passageway, **4 DV cameras**, poor lighting. Codec: Indeo 5.

| Camera | Download |
|--------|----------|
| Cam 0 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/passageway1-c0.avi |
| Cam 1 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/passageway1-c1.avi |
| Cam 2 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/passageway1-c2.avi |
| Cam 3 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/passageway1-c3.avi |

Save to: `data/`

---

## 5. Basketball sequence

Basketball training, **4 DV cameras**. Codec: Indeo 5.

| Camera | Download |
|--------|----------|
| Cam 0 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/match5-c0.avi |
| Cam 1 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/match5-c1.avi |
| Cam 2 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/match5-c2.avi |
| Cam 3 | https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/match5-c3.avi |

Save to: `data/`

---

## Ground truth (when available)

| Sequence | Ground truth | Grid | Top-view origin | Top-view size | Matching calibration |
|----------|--------------|------|-----------------|---------------|----------------------|
| Laboratory 6p | [gt_lab_6p.txt](https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/gt_lab_6p.txt) | 56×56 | (0, 0) | 358×360 | [calibration-6p.txt](https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/calibration-6p.txt) |
| Terrace seq. 1 | [gt_terrace1.txt](https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/gt_terrace1.txt) | 30×44 | (−500, −1500) | 7500×11000 | [terrace-tsai.zip](https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/terrace-tsai.zip) (Tsai) |
| Passageway seq. 1 | [gt_passageway1.txt](https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/gt_passageway1.txt) | 40×99 | (0, 38.48) | 155×381 | [calibration-passageway.txt](https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/calibration-passageway.txt) |

Save GT and calibration files to `data/`.

Positions in GT are discrete grid indices. Convert to top-view coordinates before projecting with homographies / Tsai:

```c
void grid_to_tv(int pos, int grid_width, int grid_height,
                float tv_origin_x, float tv_origin_y, float tv_width,
                float tv_height, float &tv_x, float &tv_y) {
    tv_x = ((pos % grid_width) + 0.5) * (tv_width / grid_width) + tv_origin_x;
    tv_y = ((pos / grid_width) + 0.5) * (tv_height / grid_height) + tv_origin_y;
}
```

Special position codes: `-1` = undefined label, `-2` = person outside the grid.

---

## Calibration notes

POM-style calibration uses **two homographies per camera**:

1. Ground plane → top view  
2. Head plane (parallel to ground, ~1.75 m) → top view  

Convention: `H * X_image = X_topview`.

If the camera lies in the head plane, a horizontal line height (% of image height from the top) is given instead of a head-plane homography.

Tsai calibrations for campus / terrace are also available on the official page; image size used for calibration may differ from the video resolution — normalize coordinates accordingly.

---

## Third-party annotations

Additional multi-view annotations:  
https://bitbucket.org/merayxu/multiview-object-tracking-dataset

---

## References

F. Fleuret, J. Berclaz, R. Lengagne, P. Fua.  
**Multi-Camera People Tracking with a Probabilistic Occupancy Map.**  
IEEE TPAMI, 2008. Vol. 30, no. 2, pp. 267–282.  
DOI: [10.1109/TPAMI.2007.1174](https://doi.org/10.1109/TPAMI.2007.1174)
