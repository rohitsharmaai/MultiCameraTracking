# Multi-Camera Person Tracking & Re-Identification
## Development Agent Implementation Specification

**Status:** Implementation specification
**Audience:** AI coding/development agent
**Primary goal:** Build a production-oriented multi-camera person tracking, global identity, and re-identification system.

---

# 1. Non-Negotiable Rules

The development agent MUST follow these rules.

## 1.1 Commercial model licensing is a hard gate

Do not treat a GitHub repository being public as proof that a model is commercially usable.

Models named in this specification are **APPROVED** by default.

After download, verify the exact artifact. If verification fails, mark that artifact **REJECTED**. Never use or ship a REJECTED file. Do not silently substitute another checkpoint.

```text
APPROVED (default in this spec)
   ↓
DOWNLOAD
   ↓
VERIFY exact artifact
   ├── PASS → stay APPROVED (record SHA-256, filename, version, license)
   └── FAIL → REJECTED (never use / never ship)
```

Post-download VERIFY must confirm:

1. Exact filename / artifact name.
2. Exact version.
3. SHA-256 of the downloaded file.
4. Weight-file license (not only training-code license).
5. Commercial-use rights.
6. Redistribution rights.

If the exact weights have a permissive commercial license (MIT/Apache-2.0/etc.) and there is no explicit restriction on commercial use, VERIFY **PASSES** and the model stays **APPROVED**.

Unknown training-data provenance alone does not automatically fail VERIFY.

However, if the weight license itself is unclear, contradictory, non-commercial, research-only, or requires additional permission, VERIFY **FAILS** → **REJECTED**.

The license must be on the **exact weight file**, not only on training code.

Production must not load a **REJECTED** model. Do not silently substitute another checkpoint because it has the same model name.

## 1.2 No hidden model substitutions

The implementation must use the exact model selected in this document unless the model-change procedure in Section 33 is followed.

Do not replace a model with:

- an Ultralytics model/package merely for convenience;
- a random Hugging Face checkpoint;
- a checkpoint whose **weight license** is unclear, contradictory, non-commercial, research-only, or requires extra permission;
- a model with an incompatible commercial license.

Unknown training-data provenance alone does not reject a checkpoint whose exact weights are MIT or Apache-2.0 (or another permissive commercial license) with no explicit commercial-use restriction.

## 1.3 Separate local tracking from global identity

A tracker ID is NOT a global person identity.

Example:

- Camera A / Track A_17 -> Global P1042
- Camera B / Track B_31 -> Global P1042
- Camera C / Track C_08 -> Global P1042

A local track may disappear, fragment, or restart. A new local track must be matched against the existing Global Identity Gallery before creating a new Global ID.

This applies even when the new track is on the same camera.

## 1.4 Do not force weak matches

Never assign the nearest identity simply because it is the nearest candidate.

A new Global ID may be created only when no existing identity satisfies the configured evidence thresholds.

The matcher must support:

- absolute similarity thresholds;
- score margins between best and second-best candidates;
- quality weighting;
- temporal plausibility;
- camera-topology constraints;
- identity hysteresis;
- an UNCERTAIN/UNRESOLVED state.

---

# 2. System Objective

Build a system that receives multiple camera streams and produces a persistent Global Person ID for each physical person during a session.

The system must answer:

- Who is this person?
- Which camera currently sees them?
- Which local track represents them?
- Which cameras did they visit?
- When did they transition between cameras?
- Can a fragmented local track be reconnected to the existing identity?
- Can a person be re-identified after disappearing and returning?

The system is not limited to simultaneous cross-camera matching. It must support temporal handoffs and same-camera re-entry.

---

# 3. Required End-to-End Architecture

```text
RTSP / video cameras
        |
        v
+-------------------+
| RF-DETR-L         |
| Person Detection  |
+-------------------+
        |
        v
+-------------------+
| BoT-SORT          |
| Local Tracking    |
+-------------------+
        |
        v
+-----------------------------+
| Track Observation Manager   |
| - stabilization             |
| - crop extraction           |
| - quality scoring           |
| - observation buffering     |
+-----------------------------+
       |             |
       |  crop the target person (bbox)
       |             +--------------------+
       |                                  |
       v                                  v
+-------------------+              +----------------+
| PPHumanSeg        |              | YuNet          |
| REQUIRED          |              | Face detection |
| on person crop    |              +----------------+
| not full frame    |                     |
+-------------------+                     v
       |                           Face alignment
       |  mask / quality check            |
       v                                  v
+---------------+                  AuraFace-v1
| RTMPose-Body2d|                  (glintr100.onnx)
| Body quality  |
+---------------+
       |
       |  only selected high-quality crops
       v
+----------------+
| OSNet x1.0    |
| Body ReID      |
+----------------+
       |
       +--------------------+
                            |
                            v
                  +-----------------------+
                  | Candidate Generation |
                  +-----------------------+
                            |
                            v
                  +-----------------------+
                  | Global Identity      |
                  | Matcher               |
                  | - body ReID           |
                  | - face ReID           |
                  | - quality             |
                  | - time                |
                  | - camera topology     |
                  | - motion/history      |
                  +-----------------------+
                            |
                            v
                  +-----------------------+
                  | Global Identity       |
                  | Manager               |
                  +-----------------------+
                            |
                 +----------+----------+
                 |                     |
                 v                     v
          Existing Global ID       New Global ID
                 |                     |
                 +----------+----------+
                            |
                            v
                  PostgreSQL + pgvector
                            |
                            v
                    Analytics / API
```

Use only the exact artifacts named in this document. They are APPROVED by default. After download, VERIFY the exact file. If VERIFY fails, mark REJECTED and never use or ship it. Do not silently substitute.

---

# 4. Model Registry

The project must maintain a machine-readable model registry at:

`configs/models.yaml`

This is the file the application reads at startup. `models/MANIFEST.yaml` is the on-disk license and SHA-256 audit record for every artifact in `models/`. The two files must agree on name, artifact, version, SHA-256, and status. If they disagree, production must fail fast.

Each entry must contain:

```yaml
name:
stage:
source_url:
artifact:
version:
commit:
license:
commercial_use_verified: false
redistribution_verified: false
training_provenance_verified: false
sha256:
verification_date:
verification_source:
status: APPROVED | REJECTED
```

Models in this specification start as `APPROVED`. After download, VERIFY the exact artifact (filename/version, SHA-256, license, commercial use, redistribution). If VERIFY fails, set `REJECTED` and never load that file. Do not silently substitute.

Do not rely on comments in source code as the licensing record.

---

# 5. Fixed Model Assignment

## 5.1 Person detection — RF-DETR-L

**Model:** RF-DETR-L / `RFDETRLarge`.

**Official source:** [https://github.com/roboflow/rf-detr](https://github.com/roboflow/rf-detr) via `pip install rfdetr`.

**License:** Apache-2.0 on the `rfdetr` package and Nano–Large weights. This project treats that as commercially usable.

**Selected artifact:** `RFDETRLarge` / `rfdetr-large`

Record the installed `rfdetr` package version and the SHA-256 of the downloaded Large checkpoint. Do **not** use RF-DETR-XL or RF-DETR-2XL (`rfdetr_plus`, PML-1.0).

**Purpose:** Detect people in every camera frame.

Requirements:

- Run only the person class for this application.
- Preserve detector confidence.
- Preserve bounding-box coordinates.
- Support batch inference where useful.
- Prefer GPU inference.
- Provide configurable input resolution.
- Record detector version and configuration in runtime metadata.

The detector is responsible for finding people, not determining their global identity.

### Detection output

```python
Detection(
    bbox=(x1, y1, x2, y2),
    confidence=float,
    class_id=person_class_id,
    frame_id=int,
    timestamp=float,
    camera_id=str,
)
```

Do not pass detections directly to the global identity matcher. They must first enter the local tracker.

---

# 6. Local Tracking — BoT-SORT

**Algorithm:** BoT-SORT.

**Official source:** [https://github.com/NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT)

**License:** MIT on the tracker code. This project treats that as commercially usable.

Pin that repository (record commit SHA). Use only the BoT-SORT **tracker** (association, Kalman filter, track lifecycle).

Do **not** use the YOLOX / YOLOv7 / Ultralytics detector that ships with that repo.

Feed BoT-SORT detections from the models already selected in this document:

- **Person tracks:** RF-DETR-L person boxes (`bbox`, `confidence`, `camera_id`, `frame_id`, `timestamp`).
- **Faces:** YuNet on the person crop, as in Sections 11–12. YuNet is not a second people-tracker and must not replace RF-DETR-L.

Do **not** enable FastReID / MOT appearance weights (`--with-reid` off). Local association uses motion and IoU. Global identity uses OSNet, not the tracker appearance model.

Purpose:

- associate consecutive detections within one camera;
- maintain camera-local Track IDs;
- tolerate short detector failures;
- provide track lifecycle events.

The local tracker MUST NOT own Global Person IDs.

Each camera has its own tracker namespace:

```text
camera_A: A_1, A_2, A_3...
camera_B: B_1, B_2, B_3...
```

Never assume `A_17 == B_17`.

### Required track lifecycle

```text
NEW
  -> TENTATIVE
  -> CONFIRMED
  -> LOST
  -> TERMINATED
```

A terminated track can later produce another local track that may map to the same Global Person ID.

---

# 7. Observation Quality Layer

Do not generate embeddings from every frame.

For every confirmed local track, collect candidate observations and score them.

Each observation should contain:

```text
frame_id
camera_id
timestamp
bbox
crop
sharpness
bbox_area
bbox_aspect_ratio
occlusion_score
pose_score
body_visibility_score
detector_confidence
face_quality
viewpoint
```

Use separate body and face quality decisions.

The best body observation and best face observation do not have to be the same frame.

---

# 8. Body Pose — RTMPose-Body2d

**Model:** RTMPose-Body2d (Qualcomm AI Hub export of RTMPose).

**Official source:** [https://aihub.qualcomm.com/compute/models/rtmpose_body2d](https://aihub.qualcomm.com/compute/models/rtmpose_body2d)

**Weight repo:** [https://huggingface.co/qualcomm/RTMPose-Body2d](https://huggingface.co/qualcomm/RTMPose-Body2d)

**License:** Apache-2.0 on the published model. Section 1.1 applies.

**Selected artifact:** ONNX float export `rtmpose_body2d-onnx-float` (input 256×192). Do not use the w8a16 quantized file unless a later benchmark requires it. Prefer the universal ONNX float so the pose model can run on the project GPU, not only Snapdragon.

Record SHA-256 after download. The model returns 133 joints with confidences. Use body keypoints for quality/viewpoint; extra joints may be used if useful. Do not silently substitute a different RTMPose size or a third-party re-export.

Purpose:

- estimate body keypoints;
- determine whether the body crop is sufficiently complete;
- estimate pose/viewpoint;
- help reject unusable observations;
- support diversity selection.

RTMPose-Body2d is NOT the identity model.

Use it as a quality/geometry signal.

### Body quality features

At minimum calculate:

- keypoint confidence;
- number of visible keypoints;
- torso visibility;
- left/right body visibility;
- pose stability;
- approximate viewpoint;
- crop completeness.

Do not require a perfectly frontal pose. Side/back views are valuable for body ReID.

---

# 9. Body Segmentation — PPHumanSeg (required)

**Model:** PPHumanSeg.

**Official weight source:** [https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg](https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg)

**License:** Apache-2.0 on the published ONNX files. This project treats that as commercially usable.

**Selected artifact:** `human_segmentation_pphumanseg_2023mar.onnx`

Record SHA-256 after download. Do not use the int8 quantized file unless a later benchmark shows it is required.

PPHumanSeg is **required** in the body ReID pipeline. It is a quality model, not an identity model.

Required order:

```text
RF-DETR-L person bbox
    -> crop that target person
    -> PPHumanSeg on that crop only
    -> mask / quality check
    -> RTMPose-Body2d / other body quality
    -> OSNet x1.0 embedding
```

Do **not** run segmentation on the full frame and then try to associate masks with detection boxes.

Do not generate a body embedding from a crop that fails the mask or pose quality gate.

---

# 10. Body Re-Identification — OSNet x1.0

**Model:** OSNet x1.0.

**Official weight source:** [https://huggingface.co/kaiyangzhou/osnet](https://huggingface.co/kaiyangzhou/osnet)

**License:** MIT on the architecture and the published weight files. Section 1.1 applies: APPROVED by default; after download VERIFY the exact artifact. Unknown training-data provenance alone does not fail VERIFY.

**Selected artifact:** `osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth`

Record SHA-256 after download. Do not silently substitute a different OSNet size or a third-party re-upload.

Purpose:

Generate an embedding representing a person's visual appearance.

The embedding must NOT be treated as a perfect identifier.

For every selected body observation:

```text
RF-DETR bbox
    -> person crop
    -> PPHumanSeg (crop only)
    -> mask / quality check
    -> RTMPose-Body2d / quality
    -> OSNet x1.0 preprocessing
    -> normalized embedding
```

Store normalized embeddings.

Use cosine similarity for the first implementation.

---

# 11. Face Detection — YuNet

**Model:** OpenCV Zoo YuNet.

**Official weight source:** [https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)

**License:** MIT on the architecture and the published ONNX files. This project treats that as commercially usable.

**Selected artifact:** `face_detection_yunet_2023mar.onnx`

Record SHA-256 after download. `face_detection_yunet_2026may.onnx` is a re-export of the same 2023mar weights with dynamic input shape for OpenCV 5; do not treat it as a different model.

Use YuNet for face detection and five-point facial landmarks.

Pipeline:

```text
person crop
   -> YuNet
   -> face bbox + landmarks
   -> face quality scoring
   -> alignment
```

Do not run face recognition on a low-quality face simply because a face was detected.

### Face quality

Calculate at least:

- face bounding-box size;
- detector confidence;
- blur/sharpness;
- yaw;
- pitch;
- roll;
- landmark geometry;
- occlusion;
- exposure/brightness;
- face crop completeness.

Maintain a separate face observation gallery.

---

# 12. Face Embedding — AuraFace-v1

**Model:** AuraFace-v1 (`glintr100.onnx`).

**Official weight source:** [https://huggingface.co/fal/AuraFace-v1](https://huggingface.co/fal/AuraFace-v1)

**Download:** [https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx?download=true](https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx?download=true)

**License:** Apache-2.0 on the published weights ([model card](https://huggingface.co/fal/AuraFace-v1)). Section 1.1 applies.

**Selected artifact:** `glintr100.onnx`

Record SHA-256 after download. Input is a 112×112 aligned face crop from YuNet landmarks. Output is a face embedding. Do not download the other ONNX files in that repo (`scrfd_*`, `genderage`, etc.) for this pipeline. Face detection stays YuNet.

Do not replace this artifact with InsightFace buffalo/antelope packs or a random Hugging Face face model.

---

# 13. Feature Extraction Strategy

For each local track:

1. Wait until the track is stable.
2. Collect observations (RF-DETR bbox → person crop).
3. Run PPHumanSeg on that crop and apply the mask/quality check.
4. Score body quality (RTMPose-Body2d and other signals).
5. Score face quality independently.
6. Reject low-quality observations.
7. Select a small Top-K body set.
8. Select a small Top-K face set.
9. Enforce viewpoint diversity.
10. Generate embeddings only for selected observations.
11. Store embeddings with quality metadata.

Example:

```text
300 frames
   -> 47 acceptable observations
   -> 8 diverse body observations
   -> 3 high-quality face observations
```

Do not compute 300 expensive embeddings unnecessarily.

---

# 14. Global Identity Gallery

Each Global Person ID must maintain a gallery rather than a single embedding.

Example:

```text
Global P1042
 |
 +-- body embedding: front, camera A
 +-- body embedding: side, camera A
 +-- body embedding: back, camera B
 +-- body embedding: side, camera C
 |
 +-- face embedding: high quality, camera A
 +-- face embedding: high quality, camera C
```

Each gallery item must retain:

- embedding;
- timestamp;
- camera;
- track ID;
- quality score;
- viewpoint;
- source frame reference.

Do not blindly average every embedding. Low-quality embeddings must not poison the identity representation.

### Gallery eviction

`gallery_size` is a hard cap per identity, applied separately to body and face galleries.

When the cap is exceeded:

- never overwrite an existing embedding;
- never drop the last embedding of a viewpoint if another viewpoint still has extras;
- evict the lowest-quality item first;
- if quality is tied, evict the oldest item.

Eviction means delete the lowest-value item, then insert the new one. Previous evidence must remain unless this policy removes it.

---

# 14A. Video-Based Persistent Identity Dataset (Development/Test Mode)

Because the initial development environment does not have live camera/RTSP feeds, the system MUST support **video-file ingestion as the primary test harness**. Video files simulate individual camera sessions. The system must persist identities across separate executions so that a new video can discover identities learned from previous videos.

## Required behavior

When processing a video file:

1. Read the video name/path as the `source_video`.
2. Run detection, local tracking, quality filtering, body/face feature extraction, and global identity matching.
3. For every confidently identified person, persist a Global Person ID and all selected high-quality embeddings.
4. Persist the source video name for every observation/embedding and every track.
5. Render the assigned Global Person ID on the output video.
6. When the same person appears in a later execution/video, search the persistent identity gallery first.
7. If a reliable match is found, reuse the existing Global Person ID.
8. If no reliable match is found, create the next Global Person ID and persist it.
9. Never reset the identity database between normal test runs unless an explicit `--reset-gallery`/equivalent command is supplied.
10. Every run must add new observations to the persistent gallery; it must not overwrite previous evidence.

### Example workflow

```text
Run 1:
video_01.mp4
    Person -> P0001
    Person -> P0002
    Person -> P0003

Persistent gallery:
P0001 <- embeddings <- video_01.mp4
P0002 <- embeddings <- video_01.mp4
P0003 <- embeddings <- video_01.mp4

Run 2:
video_02.mp4
    Person A -> matches P0002 -> P0002
    Person B -> no reliable match -> P0004

Persistent gallery now:
P0001 <- video_01.mp4
P0002 <- video_01.mp4 + video_02.mp4
P0003 <- video_01.mp4
P0004 <- video_02.mp4
```

The source video is evidence/provenance, **not part of the identity itself**. The same Global ID may legitimately contain observations from many videos.

## Video run manifest

Each execution must create a run record containing at minimum:

```text
run_id
source_video_name
source_video_path_or_uri
video_sha256 (when practical)
started_at
completed_at
frames_processed
tracks_detected
global_ids_seen
new_global_ids_created
existing_global_ids_reused
```

## Persistent identity provenance

Every stored body/face embedding MUST retain:

```text
global_person_id
source_video_id
source_video_name
source_frame_number
source_timestamp_ms
camera_id (synthetic/declared for video testing)
local_track_id
quality_score
viewpoint
model_name
model_version
embedding
```

This allows the developer to answer: **“Why was P0007 assigned in this video?”** and trace the identity back to the exact source video, frame, track, and embedding.

## Output video requirements

For every input video, generate an annotated output video by default. At minimum draw:

```text
Global ID: P0007
Local Track: T_42
Source: video_02.mp4
Match: existing | new | unresolved
Confidence: 0.93
```

The UI overlay must make it immediately visible when the same Global ID is reused across different input videos.

## CLI requirements

The implementation should support commands equivalent to:

```bash
python -m app.process_video --input videos/video_01.mp4
python -m app.process_video --input videos/video_02.mp4
python -m app.process_video --input videos/video_03.mp4

# Synchronized multi-camera sequence (one run, shared clock):
python -m app.process_video \
  --run-name lab-4p \
  --synchronized \
  --fps 25 \
  --camera A=data/4p-c0.avi \
  --camera B=data/4p-c1.avi \
  --camera C=data/4p-c2.avi \
  --camera D=data/4p-c3.avi

# Explicitly destructive development operation:
python -m app.gallery --reset

# Inspect what has been learned:
python -m app.gallery --list
python -m app.gallery --show P0002
```

The exact command names may differ, but the capabilities are mandatory.

## Synchronized multi-camera video runs

Single-file runs (one video = one synthetic camera session) remain valid for persistence tests (Scenario J).

When several files are views of the **same scene at the same time**, they must be processed as **one run**, not as four unrelated videos.

Rules:

- one `run_id` and one `run-name` for the whole sequence;
- each file maps to a declared `camera_id` (A/B/C/D or 0/1/2/3);
- shared clock: frame `n` in every file is the same timestamp (`n / fps`);
- one topology, one gallery update, one set of Global IDs;
- overlapping same-timestamp views use the overlap rules in Section 17;
- do not treat “seen in A and B at the same time” as a camera transition;
- write one annotated output video per input file, all showing the same Global IDs.

The `video_runs` record must list every camera file in the run, not only a single path.

## Test footage

Development and evaluation videos are specified in `DATA.md` at the project root. Do not invent a second dataset layout.

- Download and place files under `data/` as `DATA.md` describes.
- Prefer the EPFL POM laboratory 4-camera sequences first (`4p-c0.avi` … `4p-c3.avi`) with the synchronized multi-camera CLI.
- EPFL POM is for **local testing only**. It is not a commercial license for shipping those videos in a product.
- Calibration and ground-truth files in `DATA.md` are for evaluation (Section 28). They are not required for the first process-video run.

## Critical persistence rule

The application MUST NOT keep the identity gallery only in Python memory. A process restart must not erase learned identities. PostgreSQL/pgvector is the source of truth. A local cache may be used for performance, but it must be reconstructable from the persistent database.

# 15. Candidate Generation

Never compare every new track against every identity if the identity population becomes large.

Candidate generation should use:

1. Camera topology.
2. Time window.
3. Recently active identities.
4. Recently missing identities.
5. Same-camera historical identities.
6. Approximate embedding nearest-neighbor search.

Then perform a more expensive identity score on the candidate set.

Use PostgreSQL + pgvector initially unless benchmarking demonstrates a need for a dedicated vector database.

---

# 16. Global Identity Matching

The matcher must combine multiple evidence sources.

Conceptual score:

```text
S =
    w_body * body_similarity
  + w_face * face_similarity
  + w_time * temporal_plausibility
  + w_camera * camera_transition_plausibility
  + w_motion * movement_plausibility
```

Weights must be configuration, not hard-coded constants.

Quality is **not** an additive term in `S`.

### Scoring rules

- Scale every term to `[0, 1]` before applying weights. Do not add raw scores of different ranges.
- Compute `S` only on the candidate set from Section 15.
- Accept a match only if `S >=` the configured threshold **and** `(best_S - second_best_S) >=` the configured margin.
- Quality is a gate: it decides whether body and/or face evidence is used, and may scale `w_body` / `w_face` down to zero. It must not add a bonus that can create a match when visual similarity is weak.
- If both body and face fail their quality gates, set identity to `UNRESOLVED` and ignore `S`.

### Important behavior

If face quality is poor:

```text
face evidence -> low weight / ignore
body evidence -> primary
```

If body is heavily occluded but face is excellent:

```text
face evidence -> primary
body evidence -> secondary
```

If both are poor:

```text
identity -> UNRESOLVED
```

Do not force a match.

---

# 17. Camera Topology

Represent cameras as a directed graph.

Example:

```text
A -> B
A -> C
B -> D
C -> D
```

Each edge may contain:

```yaml
from_camera: A
to_camera: B
min_transition_seconds: 2
max_transition_seconds: 90
expected_direction: east
confidence: 0.9
```

Camera topology is a major false-match prevention mechanism.

If a person disappears from Camera A, do not consider every camera equally likely.

Use physically plausible transitions first.

### Simultaneous multi-camera overlap

This path applies only when **two or more cameras run at the same time** and observations share the **same timestamp** (or the same synchronized frame time). Sequential single-camera / single-video runs do not use this path.

If the same physical person is visible in overlapping cameras at that timestamp:

- the same Global ID may appear on both cameras;
- that is overlapping visibility, **not** a camera transition.

If two different Global IDs are proposed for what appears to be the same person across those simultaneous same-timestamp views:

- if the match-score difference is **above** the configured overlap margin, keep the higher-confidence assignment;
- if the difference is **not** above that margin, mark `CONFLICT` / `UNRESOLVED` and do not silently merge or split.

Do not force a merge on nearest score when the two assignments are close.

---

# 18. Same-Camera Re-Identification

This is mandatory.

A local tracker may lose a person because of:

- occlusion;
- detector failure;
- crowding;
- temporary disappearance;
- camera boundary;
- tracker fragmentation.

When a new local track starts, search the Global Identity Gallery, including identities historically seen by the same camera.

Example:

```text
Camera A
Track A_17 -> P1042

A_17 lost

later:

Track A_42
    -> global gallery search
    -> strong body match with P1042
    -> A_42 -> P1042
```

Never automatically create P2041 merely because A_42 is a new tracker ID.

---

# 19. Global Identity State Machine

Recommended states:

These states belong to a **local track's identity assignment**, not to the person record.

```text
UNRESOLVED
  -> MATCHED     reliable match (threshold + margin + hysteresis)
  -> NEW         no reliable match after the stabilization window
  -> UNCERTAIN   best and second-best are too close, or quality is too low

MATCHED
  -> UNCERTAIN   later evidence conflicts
  -> stays MATCHED unless another ID has a large score advantage

UNCERTAIN
  -> MATCHED or NEW after more observations
  -> otherwise stay UNCERTAIN (do not force)

NEW
  -> MATCHED to the ID just created for that track
```

Person records have a separate status (`active` / `inactive`). Do not mix person lifecycle with track assignment states.

Identity assignment should use hysteresis.

Once a track is confidently mapped to P1042, one weak observation must not immediately switch it to P2041.

Require sustained evidence or a large score advantage to change an established assignment.

---

# 20. Data Model

Use PostgreSQL.

Use pgvector for embeddings initially.

## cameras

```sql
id
camera_id
name
source_type
source_uri
enabled
declared_for_video_testing
created_at
```

`source_type` is `rtsp` or `video_file`. Topology edges and tracks must reference `camera_id`.

## persons

```sql
id
created_at
last_seen_at
status
first_camera_id
last_camera_id
```

## camera_tracks

```sql
id
track_id
camera_id
global_person_id
start_time
end_time
first_bbox
last_bbox
confidence
status
source_video_id
source_video_name
```

## person_embeddings

```sql
id
global_person_id
track_id
camera_id
timestamp
embedding
quality_score
viewpoint
source_frame_id
source_video_id
source_video_name
source_frame_number
source_timestamp_ms
model_name
model_version
```

## face_embeddings

```sql
id
global_person_id
track_id
camera_id
timestamp
embedding
quality_score
viewpoint
source_frame_id
source_video_id
source_video_name
source_frame_number
source_timestamp_ms
model_name
model_version
```

## video_runs

```sql
id
run_id
source_video_name
source_video_path_or_uri
video_sha256
started_at
completed_at
frames_processed
tracks_detected
global_ids_seen
new_global_ids_created
existing_global_ids_reused
```

## source_videos

```sql
id
source_video_name
source_video_path_or_uri
video_sha256
first_processed_at
last_processed_at
```

## transitions

```sql
id
global_person_id
from_camera_id
to_camera_id
from_time
to_time
match_score
match_method
confidence
```

## model_registry

```sql
id
model_name
artifact
version
sha256
license
commercial_use_verified
verification_date
status
```

---

# 21. Camera Processing Architecture

Each camera should have an isolated processing pipeline.

```text
CameraWorker(camera_id)
    |
    +-- frame ingestion
    +-- detector
    +-- local tracker
    +-- observation manager
    +-- quality models
    +-- embedding extraction
    +-- global identity service
```

Do not allow one camera worker to mutate another camera's local tracker state.

Global identity state must be centralized and concurrency-safe.

---

# 22. Frame Processing Requirements

The system must support:

- RTSP;
- reconnect after camera failure;
- frame dropping under overload;
- configurable FPS sampling;
- GPU batching where useful;
- asynchronous embedding extraction;
- bounded queues;
- backpressure;
- graceful shutdown.

Do not let an overloaded camera block every other camera.

---

# 23. Performance Architecture

Separate real-time tracking from expensive feature extraction.

```text
FAST PATH
frame -> RF-DETR -> BoT-SORT -> track update

ASYNC PATH
selected person crops -> PPHumanSeg -> RTMPose -> OSNet / YuNet -> face embedding

GLOBAL PATH
embeddings -> candidate search -> identity matcher -> DB
```

The tracker must not wait unnecessarily for a face embedding model.

---

# 24. API Requirements

Provide at least:

```text
GET  /health
GET  /cameras
POST /cameras/{camera_id}/start
POST /cameras/{camera_id}/stop
GET  /cameras/{camera_id}/tracks
GET  /persons
GET  /persons/{global_id}
GET  /persons/{global_id}/timeline
GET  /persons/{global_id}/cameras
GET  /transitions
GET  /metrics
```

Every endpoint except `GET /health` requires authentication (API key or equivalent). Unauthorized requests must be rejected.

Video-file processing and gallery list/show/reset are **CLI only** (`python -m app.process_video`, `python -m app.gallery`). Do not add HTTP endpoints for those unless this document is updated first.

The API must never expose raw biometric embeddings unnecessarily.

---

# 25. Event Model

Publish internal events such as:

```text
DetectionCreated
TrackStarted
TrackConfirmed
TrackLost
TrackTerminated
ObservationAccepted
BodyEmbeddingCreated
FaceEmbeddingCreated
IdentityMatched
IdentityCreated
IdentityUnresolved
CameraTransitionDetected
```

Every event must include timestamp and camera/track context where applicable.

---

# 26. Observability

Record metrics for:

### Detection

- detections/frame;
- detector confidence;
- missed detection rate;
- processing latency.

### Tracking

- active tracks;
- track duration;
- ID switches;
- track fragmentation;
- lost/recovered tracks.

### ReID

- candidate count;
- best similarity;
- second-best similarity;
- score margin;
- match/reject rate;
- uncertain rate;
- same-camera recovery rate;
- cross-camera recovery rate.

### System

- GPU utilization;
- CPU utilization;
- queue depth;
- frame latency;
- DB latency;
- embedding-search latency.

---

# 27. Testing Requirements

The agent must create automated tests for:

## Unit tests

- bbox conversion;
- crop extraction;
- quality scoring;
- embedding normalization;
- cosine similarity;
- candidate filtering;
- camera topology;
- transition windows;
- identity state machine.

## Integration tests

- detector -> tracker;
- tracker -> observation manager;
- observation -> ReID;
- ReID -> candidate search;
- candidate search -> identity assignment;
- DB persistence;
- camera reconnect.

## Scenario tests

### Scenario A — single camera

One person remains in one camera.

Expected: one Global ID.

### Scenario B — tracker fragmentation

One person receives Track A_1 and later A_2.

Expected: both map to the same Global ID.

### Scenario C — cross-camera transition

Person moves A -> B.

Expected: same Global ID.

### Scenario D — multiple similar people

Two people wear similar clothes.

Expected: identity assignment uses multiple observations and does not blindly merge them.

### Scenario E — face unavailable

Person is visible from behind.

Expected: body ReID remains usable.

### Scenario F — body poor, face good

Expected: face evidence can dominate when quality is high.

### Scenario G — both poor

Expected: UNRESOLVED / no forced identity assignment.

### Scenario H — same-camera return

Person leaves and later re-enters.

Expected: existing Global ID is recovered when evidence is strong.

### Scenario I — camera transition impossible by topology

Strong visual similarity but impossible camera transition.

Expected: topology reduces or rejects the candidate.

### Scenario J — persistent video-to-video identity

Process `video_01.mp4`, stop the application, then process `video_02.mp4`. The same person appears in both videos.

Expected: the person receives the same Global ID in both runs; the gallery contains embeddings from both source videos.

### Scenario K — new identity added to existing gallery

Process a video containing known and previously unseen people.

Expected: known people reuse their existing Global IDs; unseen people receive new IDs without changing existing IDs.

### Scenario L — source provenance

Inspect the database and output video after multiple runs.

Expected: every embedding/track can be traced to its source video, frame/time, local track ID, model version, and quality score.

### Scenario M — process restart persistence

Run video 1, terminate the process, restart the application, then run video 2.

Expected: identities learned from video 1 remain available and are used for video 2.

---

# 28. Evaluation Dataset

Start with the sequences and ground truth listed in `DATA.md`. Then, before production, also build a project-specific evaluation set containing:

- different cameras;
- different lighting;
- front/side/back views;
- crowded scenes;
- occlusion;
- similar clothing;
- different clothing where possible;
- short and long transitions;
- same-camera tracker fragmentation;
- re-entry.

Ground truth must contain:

```text
frame
camera
person_ground_truth_id
track_id
```

Evaluate separately:

1. person detection;
2. local tracking;
3. body ReID;
4. face ReID;
5. global identity assignment;
6. camera transition accuracy.

Do not judge the complete system using detector mAP alone.

---

# 29. Identity Evaluation Metrics

Track at minimum:

- IDF1;
- HOTA;
- MOTA where appropriate;
- ID switches;
- track fragmentation;
- cross-camera identity accuracy;
- same-camera recovery accuracy;
- false merge rate;
- false split rate;
- unresolved rate;
- identity-switch rate.

For this project, **false merges are particularly important**. A system that merges two different people into one Global ID can corrupt the entire timeline.

---

# 30. Security and Privacy

The system processes biometric-like information.

Implement:

- authentication for APIs;
- authorization by role;
- encryption in transit;
- encryption at rest where appropriate;
- access logging;
- configurable retention;
- deletion workflows;
- minimum necessary exposure of embeddings;
- no embeddings in ordinary application logs;
- no raw camera frames in logs.

Before deployment, the project owner must perform the applicable legal/privacy review for the deployment jurisdiction and use case.

Do not assume that an open-source model license resolves biometric/privacy obligations.

---

# 31. Repository Structure

Recommended structure:

```text
project/
├── app/
│   ├── process_video.py
│   └── gallery.py
├── configs/
│   ├── cameras.yaml
│   ├── models.yaml
│   ├── thresholds.yaml
│   └── topology.yaml
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── pose/
│   ├── segmentation/
│   ├── face/
│   ├── reid/
│   ├── identity/
│   ├── camera/
│   ├── database/
│   ├── api/
│   └── metrics/
├── models/
│   └── MANIFEST.yaml
├── tests/
├── scripts/
├── docker/
├── docs/
└── README.md
```

CLI commands such as `python -m app.process_video` and `python -m app.gallery` must live in the `app/` package. Library code stays under `src/`.

---

# 32. Model Manifest

Every model placed in `models/` must have an entry in `models/MANIFEST.yaml`. This file is the audit record for files on disk. Runtime loading still uses `configs/models.yaml`. The two must match; production must fail if they do not.

Example:

```yaml
- name: RF-DETR-L
  stage: person_detection
  source_url: https://github.com/roboflow/rf-detr
  artifact: RFDETRLarge / rfdetr-large
  sha256: <sha256>
  license: Apache-2.0
  commercial_use_verified: true
  redistribution_verified: true
  status: APPROVED
  notes: Do not use XL or 2XL (PML-1.0).

- name: BoT-SORT
  stage: local_tracking
  source_url: https://github.com/NirAharon/BoT-SORT
  artifact: NirAharon/BoT-SORT (pin commit SHA)
  sha256: <commit-sha>
  license: MIT
  commercial_use_verified: true
  redistribution_verified: true
  status: APPROVED
  notes: Appearance / FastReID weights off. Use RF-DETR-L person boxes, not YOLOX. Faces use YuNet.

- name: OSNet-x1.0
  stage: body_reid
  source_url: https://huggingface.co/kaiyangzhou/osnet
  artifact: osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth
  sha256: <sha256>
  license: MIT
  commercial_use_verified: true
  redistribution_verified: true
  status: APPROVED

- name: RTMPose-Body2d
  stage: body_pose
  source_url: https://aihub.qualcomm.com/compute/models/rtmpose_body2d
  artifact: rtmpose_body2d-onnx-float
  sha256: <sha256>
  license: Apache-2.0
  commercial_use_verified: true
  redistribution_verified: true
  status: APPROVED
  notes: Use ONNX float from https://huggingface.co/qualcomm/RTMPose-Body2d. Not Snapdragon-only.

- name: PPHumanSeg
  stage: body_segmentation
  source_url: https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg
  artifact: human_segmentation_pphumanseg_2023mar.onnx
  sha256: <sha256>
  license: Apache-2.0
  commercial_use_verified: true
  redistribution_verified: true
  status: APPROVED

- name: YuNet
  stage: face_detection
  source_url: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
  artifact: face_detection_yunet_2023mar.onnx
  sha256: <sha256>
  license: MIT
  commercial_use_verified: true
  redistribution_verified: true
  status: APPROVED

- name: AuraFace-v1
  stage: face_embedding
  source_url: https://huggingface.co/fal/AuraFace-v1
  artifact: glintr100.onnx
  sha256: <sha256>
  license: Apache-2.0
  commercial_use_verified: true
  redistribution_verified: true
  status: APPROVED
  notes: Download only glintr100.onnx. Face detection stays YuNet.
```

Never keep `status: APPROVED` after a failed post-download VERIFY. Download the exact file, check filename/version, SHA-256, license, commercial use, and redistribution. Fail → REJECTED. Do not silently substitute.

---

# 33. Model Change Procedure

If a developer wants to replace a model:

1. Identify why the change is needed.
2. Define required functionality.
3. Benchmark candidate models.
4. Verify exact artifact license.
5. Verify exact weights.
6. Verify commercial use.
7. Verify redistribution.
8. Apply the Section 1.1 weight-license rule. Unknown training-data provenance alone does not reject the model.
9. Record SHA-256.
10. Run regression tests.
11. Compare latency and memory.
12. Update `models.yaml` and `MANIFEST.yaml`.
13. Update this document.
14. Record the decision in `docs/model-decisions/`.

No model substitution may happen silently.

---

# 34. Configuration Requirements

Do not hard-code thresholds.

At minimum configure:

```yaml
tracking:
  min_confirmed_frames: 3
  max_lost_frames: 30

reid:
  body_similarity_threshold: <calibrated>
  face_similarity_threshold: <calibrated>
  minimum_score_margin: <calibrated>
  gallery_size: 20  # per identity, per modality; see Section 14 eviction policy
  top_k_observations: 8

identity:
  create_new_id_only_after_rejection: true
  hysteresis_enabled: true

camera_transition:
  topology_enabled: true
  temporal_constraints_enabled: true

overlap:
  enabled_only_when_cameras_are_simultaneous: true
  same_timestamp_required: true
  score_margin: <calibrated>  # above this: keep max confidence; else CONFLICT
```

Values must be calibrated against project data. Do not invent production thresholds without evaluation.

---

# 35. Important Matching Rule

The global identity manager is the authority.

The sequence is:

```text
New local track
       |
       v
Track stabilization
       |
       v
Quality observations
       |
       v
Body + face features
       |
       v
Candidate generation
       |
       v
Existing Global Identity Gallery
       |
       +---------------------+
       |                     |
 reliable match          no reliable match
       |                     |
       v                     v
reuse Global ID        create Global ID
```

Do not reverse this order.

Do not create a new Global ID immediately when the local tracker starts.

---

# 36. Failure Handling

The system must degrade gracefully.

### Detector failure

Continue tracker prediction briefly, then mark track uncertain/lost.

### Face detector failure

Continue using body ReID.

### Face embedding unavailable

Do not block body ReID.

### Body ReID unavailable

Face evidence may still be used if sufficiently strong and policy allows it.

### Database unavailable

Do not silently create inconsistent identities. Use a bounded local buffer and enter degraded mode.

### GPU failure

Expose a health failure and configurable CPU fallback where practical.

---

# 37. Performance Targets

Do not assume benchmark numbers from model repositories equal production performance.

Measure on the target hardware and target video streams.

Measure:

- FPS per camera;
- total cameras per GPU;
- detector latency;
- tracker latency;
- pose latency;
- body ReID latency;
- face latency;
- global matching latency;
- end-to-end latency;
- GPU VRAM;
- CPU/RAM;
- queue depth.

Optimize only after collecting measurements.

---

# 38. Development Phases

## Phase 1 — Infrastructure

Implement:

- project structure;
- configuration;
- model registry;
- camera abstraction;
- PostgreSQL schema;
- pgvector;
- logging;
- metrics;
- Docker environment.

## Phase 2 — Detection + Local Tracking

Implement:

- RF-DETR-L;
- BoT-SORT;
- local track lifecycle;
- visualization;
- performance benchmark.

## Phase 3 — Quality Pipeline

Implement:

- observation buffering;
- RTMPose-Body2d;
- blur/quality metrics;
- PPHumanSeg on each person crop (required; not full-frame);
- body observation selection;
- face observation selection.

## Phase 4 — ReID

Implement:

- OSNet x1.0 from https://huggingface.co/kaiyangzhou/osnet (MIT weights);
- YuNet;
- AuraFace-v1 (`glintr100.onnx` from https://huggingface.co/fal/AuraFace-v1, Apache-2.0);
- embedding normalization;
- gallery storage.

## Phase 5 — Global Identity

Implement first. The video harness in Phase 6 depends on this.

- candidate generation;
- pgvector search;
- scoring;
- topology;
- temporal transitions;
- hysteresis;
- same-camera re-identification;
- new Global ID creation.

## Phase 6 — Persistent Video Test Harness

Required after Phase 5. This phase is not optional.

- implement video-file ingestion (single file and synchronized multi-camera);
- implement persistent `video_runs` and `source_videos`;
- persist Global IDs and embeddings across executions;
- persist source video provenance on tracks and embeddings;
- generate annotated output videos;
- implement gallery inspection/reset commands;
- verify process-restart persistence before live camera integration.

## Phase 7 — Evaluation

Run the complete evaluation suite and calibrate thresholds.

## Phase 8 — Production Hardening

Implement:

- recovery;
- monitoring;
- security;
- retention;
- deployment;
- load testing.

---

## Video-based development acceptance criteria

The implementation is not considered complete until the following sequence works end-to-end:

```text
1. process video_A.mp4
2. assign and persist P0001...Pn
3. persist body/face embeddings + source video metadata
4. stop application
5. start application again
6. process video_B.mp4
7. recover matching Global IDs from the persistent gallery
8. create new IDs only for genuinely unmatched people
9. render Global IDs on video_B output
10. inspect the gallery and see evidence from both videos
```

# 39. Definition of Done

The implementation is not complete until all of the following are true:

- [ ] All required models are registered.
- [ ] Every exact artifact has a recorded SHA-256.
- [ ] Every required model is APPROVED after post-download VERIFY (or REJECTED and unused).
- [ ] RF-DETR-L detects people reliably on project footage.
- [ ] BoT-SORT maintains camera-local tracks.
- [ ] Local Track IDs are separate from Global Person IDs.
- [ ] Body observation quality selection works.
- [ ] Face observation quality selection works.
- [ ] Same-camera re-identification works.
- [ ] Cross-camera re-identification works.
- [ ] Camera topology is enforced.
- [ ] Weak matches can remain unresolved.
- [ ] Identity hysteresis prevents rapid identity switching.
- [ ] Global galleries retain multiple high-quality observations.
- [ ] Database persistence works.
- [ ] Video-file ingestion works (single file and synchronized multi-camera).
- [ ] Identities survive process restart.
- [ ] The same person across video_A and video_B reuses the same Global ID.
- [ ] Source provenance is stored on every embedding and track.
- [ ] Annotated output videos are rendered.
- [ ] Gallery list / show / reset commands exist.
- [ ] API works.
- [ ] Metrics work.
- [ ] Automated tests pass.
- [ ] Load testing has been performed.
- [ ] Model licensing manifest is complete.
- [ ] No unapproved pretrained checkpoint is included.
- [ ] No hidden model/package substitution exists.

---

# 40. Final Model List

| Component | Selected model | Purpose | Production gate |
|---|---|---|---|
| Person detector | **RF-DETR-L** (`RFDETRLarge` / `rfdetr-large`) | Person bounding boxes | Apache-2.0 published weights. Do not use XL/2XL |
| Local tracker | **BoT-SORT** ([NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT)) | Camera-local track IDs | MIT tracker. Appearance/FastReID off |
| Body ReID | **OSNet x1.0** from [kaiyangzhou/osnet](https://huggingface.co/kaiyangzhou/osnet) | Person appearance embeddings | MIT architecture + MIT published weights; use the listed `osnet_x1_0` artifact |
| Pose | **RTMPose-Body2d** (`rtmpose_body2d-onnx-float`) | Body keypoints, pose and quality | Apache-2.0 from [Qualcomm AI Hub](https://aihub.qualcomm.com/compute/models/rtmpose_body2d) / [HF](https://huggingface.co/qualcomm/RTMPose-Body2d) |
| Body segmentation | **PPHumanSeg** (`human_segmentation_pphumanseg_2023mar.onnx`) | Required crop-level mask/quality before pose and OSNet | Apache-2.0. Run on person crop only, never full-frame |
| Face detector | **YuNet** (`face_detection_yunet_2023mar.onnx`) | Face bbox + landmarks | MIT published weights from OpenCV Zoo |
| Face embedding | **AuraFace-v1** (`glintr100.onnx`) | Face embeddings | Apache-2.0 from [fal/AuraFace-v1](https://huggingface.co/fal/AuraFace-v1). Face detection stays YuNet |

### Critical implementation instruction

The agent must treat the licensing gate as a build requirement, not documentation.

If post-download VERIFY fails, the correct behavior is:

```text
DO NOT LOAD IN PRODUCTION
DO NOT SHIP
DO NOT SILENTLY SUBSTITUTE
MARK REJECTED
```

The engineering objective is not merely to make the demo work. It is to produce a technically robust, auditable, commercially deployable multi-camera identity system.
