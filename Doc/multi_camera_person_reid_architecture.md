# Multi-Camera Person Tracking & Re-Identification --- Architecture Specification

## 0. Strict Commercial Model Licensing Policy

This project has a hard requirement:

> **Only models and exact pretrained weights with explicit commercial-use rights may be used.**

"Open source", "free", "MIT code", or "available on GitHub" is NOT sufficient.

For every model we ship, verify separately:

1. Source-code license.
2. Exact pretrained-weight license.
3. Commercial-use permission.
4. Redistribution/deployment permission.
5. Training-data/model provenance when relevant.
6. Exact version/checkpoint SHA-256.
7. Required attribution/notices.

If any item is unclear, the model is **not approved**.

We prefer permissive licenses such as MIT and Apache-2.0, but the exact model weights still need to be checked.

The approved whitelist is intentionally small. We will not add a popular model merely because it is technically strong if its commercial licensing is ambiguous.


# 0A. Fixed Model Assignment for This Version

Every processing stage in this architecture has an explicitly assigned model.

## Approved model stack

| Stage | Model | Exact artifact / implementation | Purpose | License status |
|---|---|---|---|---|
| Person detection | **RF-DETR-L** | `RFDETRLarge` pretrained checkpoint (RF-DETR-L) | Detect person bounding boxes | Apache-2.0 model directory |
| Local tracking | **BoT-SORT** | Pinned MIT implementation | Maintain camera-local track IDs | MIT |
| Body ReID | **OSNet x1.0** | OSNet x1.0 pretrained checkpoint | Produce person appearance embeddings | Apache-2.0 model directory |
| Body/pose quality | **RTMPose-M** | `rtmpose-m_8xb256-420e_coco-256x192` (MMPose) | Estimate body keypoints and visibility | OpenCV Zoo model directory / Apache-2.0 |
| Occlusion/body-mask quality | **PPHumanSeg** | `human_segmentation_pphumanseg_2023mar.onnx` | Estimate visible human pixels and improve body-quality scoring | Apache-2.0 model directory |
| Face detection | **YuNet** | `face_detection_yunet_2023mar.onnx` | Detect face and facial landmarks | MIT model directory |
| Face embedding | **Custom face embedding model trained from scratch** | Project-owned checkpoint | Produce face embeddings without importing research-only pretrained weights | Project-owned; training data must be commercially licensed/customer-owned |

### Important

This is a **strict whitelist**.

Do not substitute another checkpoint merely because it has the same architecture name.

For example:

```text
OpenCV Zoo OSNet x1.0 checkpoint
        !=
arbitrary Youtu/OSNet/ReID checkpoint downloaded elsewhere
```

The exact file, version, SHA-256, and license must be recorded.

The OpenCV Zoo repository states that individual model licenses must be checked separately. citeturn1search0 The OSNet implementation is open source, but the exact pretrained weight used here must be separately pinned and its weight/training-data terms verified before commercial deployment. citeturn1search2turn1search4 The RF-DETR-L model directory is Apache-2.0. citeturn3search0turn3search5 YuNet's model directory is MIT. citeturn0search6 MMPose itself is Apache-2.0 and provides the RTMPose-M human-pose configuration; the exact pretrained checkpoint terms must still be recorded. citeturn1search5turn1search6


# 1. Purpose

This document defines the architecture for a multi-camera video
analytics system that:

-   Detects people in multiple camera feeds.
-   Maintains a stable local track for each person within each camera.
-   Re-identifies the same physical person when they move between
    cameras.
-   Assigns one persistent Global Person ID while the person remains in
    the monitored environment.
-   Uses person/body appearance as the primary cross-camera signal.
-   Uses face recognition as an additional identity signal when a
    sufficiently good face is available.
-   Uses camera topology, time, movement direction, and track history to
    reduce false cross-camera matches.
-   Stores track history, embeddings, identity associations, and
    transition events for later analytics.

The system must be designed so that detection, tracking, person ReID,
face recognition, and global association are replaceable components.

------------------------------------------------------------------------

# 2. Core Design Principle

A camera-local tracker and a global identity system are different
things.

Example:

``` text
Camera A
    Track A_17
        |
        v
    Global Person P1042
        |
        v
Camera B
    Track B_31
        |
        v
    Global Person P1042
        |
        v
Camera D
    Track D_08
        |
        v
    Global Person P1042
```

`A_17`, `B_31`, and `D_08` are temporary camera-local track IDs.

`P1042` is the persistent global identity.

The tracker does NOT perform cross-camera identification.

The global identity engine performs cross-camera association.

------------------------------------------------------------------------

# 3. High-Level Architecture

``` text
                       CAMERA / RTSP STREAMS
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
          Camera A         Camera B         Camera C
             |                |                |
             v                v                v
       Person Detector   Person Detector   Person Detector
             |                |                |
             v                v                v
        Local Tracker    Local Tracker    Local Tracker
        A_17, A_18       B_04, B_31       C_08, C_12
             |                |                |
             +----------------+----------------+
                              |
                              v
                     Track Observation Layer
                              |
                +-------------+-------------+
                |                           |
                v                           v
         Person ReID                   Face Pipeline
         Body embedding               Face embedding
                |                           |
                +-------------+-------------+
                              |
                              v
                   Candidate Generation
                              |
                              v
                  Global Identity Matcher
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
        ReID similarity   Face similarity   Temporal/
                                           spatial logic
             |                |                |
             +----------------+----------------+
                              |
                              v
                    Global Identity Manager
                              |
                +-------------+-------------+
                |                           |
                v                           v
          Global Person ID             Uncertain / New
             P1042                         ID
                |
                v
          Identity Gallery
                |
                v
       PostgreSQL + pgvector
```

------------------------------------------------------------------------

# 4. Component Responsibilities

## 4.1 Person Detector

**Model: RF-DETR-L — `RFDETRLarge` pretrained checkpoint (RF-DETR-L)**

The detector answers:

> Where are the people in this frame?

Output:

``` text
class = person
bounding_box = [x1, y1, x2, y2]
confidence = 0.93
```

The detector does NOT determine identity.

The detector does NOT perform ReID.

The detector only supplies person detections to the tracker.

### Detector requirements

The production detector should:

-   Have a commercial-use-compatible license for the exact code and
    weights being shipped.
-   Support the required camera resolution.
-   Maintain adequate recall under occlusion.
-   Have sufficient FPS for the deployment GPU.
-   Support ONNX/TensorRT or another production inference path if
    required.

### Licensing warning

Do not assume that a popular detector is commercially free because its
source code is public.

For example, Ultralytics licensing has historically included AGPL and a
separate commercial license option. Therefore, if a strict
free-commercial requirement applies, verify the exact detector
repository, version, model weights, and deployment terms before adopting
it.

------------------------------------------------------------------------

# 5. Local Multi-Object Tracking

Recommended starting choices:

-   BoT-SORT
-   BoT-SORT

The tracker receives detections:

``` text
Frame 100:
person [100, 200, 150, 400]
person [500, 100, 570, 390]
```

and maintains:

``` text
Track A_17
Track A_18
```

Across frames:

``` text
Frame 100 -> A_17
Frame 101 -> A_17
Frame 102 -> A_17
...
Frame 200 -> A_17
```

The tracker is responsible for temporal continuity inside ONE camera.

It is not responsible for:

``` text
Camera A -> Camera B
```

That is the job of the global association engine.

------------------------------------------------------------------------

# 6. Person Re-Identification

**Model: OSNet x1.0 — OSNet x1.0 pretrained checkpoint**

This is the approved body/person appearance feature extractor for this project.

## 6.1 Purpose

Person ReID answers:

> Does this person visually resemble a previously observed person?

A person crop is extracted from a track:

``` text
Tracker
   |
   v
Person crop
   |
   v
Person ReID model
   |
   v
Embedding
```

Example:

``` text
[0.12, -0.04, 0.81, ..., 0.21]
```

The embedding is normally L2-normalized.

Similarity can initially be calculated using cosine similarity:

``` text
similarity(A, B) = normalized_embedding_A dot normalized_embedding_B
```

The threshold must be calibrated on the actual camera environment. Do
not hard-code a generic threshold from an internet example.

------------------------------------------------------------------------

# 7. Approved Person ReID Model

## 7.1 OSNet x1.0

For this project, the approved pretrained Person ReID model is:

**OSNet x1.0 (OSNet x1.0 pretrained checkpoint)**

The model was added to OpenCV Zoo with the exact model weight identified as **Apache-2.0 licensed**. The OpenCV Zoo contribution record identifies the source model as Tencent Youtu Lab's Youtu ReID baseline and specifies Apache-2.0 for the weight. citeturn4search2

OpenCV Zoo also provides the model as:

```text
person_reid_youtu_2021nov.onnx
person_reid_youtu_2021nov_int8.onnx
```

and publishes benchmarks for it. citeturn4search3

### Why this is the approved choice

We are now using OSNet x1.0 as the requested body ReID model. Its implementation is MIT, but the exact pretrained checkpoint and training-data provenance must be separately verified before commercial deployment.

The OSNet implementation repository is MIT, but the pretrained checkpoints are trained on external ReID datasets. For a strict "no licensing ambiguity anywhere" commercial requirement, we do not want to rely on a checkpoint unless its exact weights and training-data rights have been verified separately.

Therefore:

```text
APPROVED:
OSNet x1.0 exact checkpoint — license/provenance verification required

NOT APPROVED FOR THIS PROJECT:
OSNet x1.0 pretrained checkpoint
unless a separate legal/compliance review verifies
the exact checkpoint and its training-data rights.
```

### OSNet x1.0 output

The model receives a person crop:

```text
person bounding box
        |
        v
OSNet x1.0
        |
        v
appearance embedding
```

The embedding is normalized and compared using cosine similarity or another calibrated metric.

The exact matching threshold must be established from the project's validation data.

---

# 7.2 Person ReID Training Policy

If OSNet x1.0 is insufficient for the deployment environment, do NOT simply download another popular ReID checkpoint.

The next compliant option is:

```text
commercially permitted ReID architecture
+
customer-owned / properly licensed training data
+
our own trained weights
```

The training dataset must itself have explicit commercial-use rights.

This prevents the common mistake of having:

```text
MIT code
+
research-only pretrained weights
```

and assuming the resulting commercial deployment is license-safe.

# 8. Important ReID Principle: Do Not Store Only One Embedding

A Global Person ID should have an appearance gallery.

Example:

``` text
P1042
 |
 +-- Camera A
 |    +-- embedding_001
 |    +-- embedding_002
 |    +-- embedding_003
 |
 +-- Camera B
 |    +-- embedding_004
 |    +-- embedding_005
 |
 +-- Camera C
      +-- embedding_006
      +-- embedding_007
```

This allows the identity representation to adapt to:

-   different viewpoints
-   different cameras
-   lighting changes
-   pose changes
-   partial occlusion
-   clothing appearance differences

The system should not blindly append every frame embedding.

Only high-quality observations should become gallery exemplars.

------------------------------------------------------------------------

# 9. Embedding Quality Filtering

Before adding an embedding to the gallery, evaluate:

-   Person bounding-box size
-   Detection confidence
-   Occlusion
-   Blur
-   Crop completeness
-   Pose/viewpoint
-   Duplicate similarity to existing gallery entries

Example policy:

``` text
if detection_confidence < threshold:
    reject

if crop_too_small:
    reject

if heavily_occluded:
    reject

if image_quality < threshold:
    reject

otherwise:
    candidate_gallery_embedding
```

This prevents a poor observation from contaminating the person's
identity gallery.

------------------------------------------------------------------------

# 10. Face Recognition Pipeline

Face recognition is a separate subsystem.

``` text
Person track
     |
     v
Face detector
     |
     v
Face crop
     |
     v
Face alignment
     |
     v
Face recognition model
     |
     v
Face embedding
```

Face recognition answers:

> Are these two detected faces likely to belong to the same person?

It should NOT be the only mechanism for cross-camera tracking.

------------------------------------------------------------------------

# 11. Face Model Policy — Strict Commercial License Requirement

### Face detection model: YuNet

Use:

```text
YuNet
face_detection_yunet_2023mar.onnx
```

Purpose:

- face bounding box
- five-point facial landmarks
- face quality measurements
- alignment input

The exact OpenCV Zoo YuNet model directory is MIT licensed. citeturn0search6

### Face embedding model: project-owned custom model

For face identity, this architecture intentionally does **not** import a research-only pretrained ArcFace/InsightFace checkpoint.

Use a project-owned face embedding model trained from scratch:

```text
ResNet-18 architecture
+
random initialization
+
commercially licensed/customer-owned face training data
+
project training
=
project-owned face embedding checkpoint
```

The model's purpose is:

```text
aligned face
    |
    v
custom face embedding network
    |
    v
normalized face embedding
```

The architecture can be replaced later, but the production checkpoint must remain project-owned or explicitly commercially licensed.

Face recognition is subject to the same strict rule as Person ReID:

> **No pretrained face-recognition weight is allowed into the production architecture unless the exact weight has an explicit permissive commercial-use license.**

## 11.1 Face Detection

The approved face detector is:

**OpenCV Zoo YuNet**

The current OpenCV Zoo YuNet model directory states that its files are licensed under the MIT License. citeturn1search3

Pipeline:

```text
person track
    |
    v
YuNet face detector
    |
    v
face bounding box + landmarks
```

YuNet is therefore approved for the face-detection stage, subject to retaining the applicable license/notice files.

## 11.2 Face Recognition / Face Embedding

We are deliberately **not approving InsightFace/ArcFace pretrained model packages** for this project.

The current official InsightFace documentation states that its code is MIT, while its distributed pretrained models are for non-commercial research and require separate commercial licensing for production use. citeturn0search5turn0search2

Therefore the following are explicitly excluded from this project unless a separate paid commercial license is obtained:

```text
InsightFace buffalo_l
InsightFace buffalo_m
InsightFace buffalo_s
InsightFace antelopev2
InsightFace ArcFace pretrained packages
InsightFace SCRFD pretrained packages
```

The requested face embedding model is OpenCV Zoo SFace: `face_recognition_sface_2021dec.onnx`. The OpenCV Zoo directory states Apache-2.0 for all files, but a July 2026 public issue asks for clarification of the exact weight's commercial-use coverage and training-data provenance. Therefore SFace is included in the requested architecture, but it is **NOT considered legally cleared under our strict zero-ambiguity policy until that provenance question is resolved**. citeturn0search3turn0search9

### Approved approach for face recognition

For the first production-compliant version:

```text
YuNet
  |
  v
face detection + landmarks
  |
  v
NO unverified pretrained face-recognition checkpoint
```

If face identity is required, use one of these compliant paths:

### Path A — Customer-owned/licensed face model

Use a face embedding model whose exact weights are explicitly licensed for commercial use.

### Path B — Train our own face embedding model

Use:

```text
commercially permitted architecture/code
+
commercially licensed or customer-owned face dataset
+
our own training
```

The resulting checkpoint is owned/controlled by the project and avoids importing research-only pretrained weights.

Until Path A or Path B is completed, **face recognition remains an optional subsystem and must not be represented as commercially cleared.**

# 12. Face Embedding Gallery

A Global Person ID can also maintain face observations:

``` text
P1042
 |
 +-- Face embedding F1
 +-- Face embedding F2
 +-- Face embedding F3
```

Only high-quality face observations should be stored.

A face observation should ideally pass:

-   minimum face size
-   sharpness/blur threshold
-   pose threshold
-   detection confidence
-   alignment quality
-   optional liveness/security policy if the application requires it

------------------------------------------------------------------------

# 13. Cross-Camera Re-Identification

This is the central algorithm.

Suppose:

``` text
Camera A
Track A_17
```

is assigned:

``` text
Global ID P1042
```

Later:

``` text
Camera B
Track B_31
```

appears.

The system creates:

``` text
Body embedding B_new
Face embedding F_new (if available)
Camera = B
Timestamp = T
```

Then candidate generation identifies existing global identities that
could plausibly be the same person.

------------------------------------------------------------------------

# 14. Candidate Generation

Do NOT compare every new track against every historical identity when
the system becomes large.

First filter candidates using:

## 14.1 Camera topology

Example:

``` text
A -> B
A -> C
B -> D
C -> D
```

If a person disappeared from A, only B/C may be plausible next cameras.

## 14.2 Time

If:

``` text
A last seen = 12:05:10
B first seen = 12:05:14
```

the transition may be plausible.

If the cameras are far apart and the elapsed time is physically
impossible, reject the candidate.

## 14.3 Direction

If the person is moving east and the next camera is west, reduce the
candidate probability.

## 14.4 Zone/entrance information

If the system knows:

``` text
Camera A exit zone = north corridor
Camera B entry zone = north corridor
```

that is additional evidence.

------------------------------------------------------------------------

# 15. Candidate Matching

For each candidate:

``` text
Body ReID similarity
Face similarity
Temporal consistency
Camera transition consistency
Motion/direction consistency
Track quality
```

A conceptual score:

``` text
score =
    w_body * body_similarity
  + w_face * face_similarity
  + w_time * temporal_score
  + w_camera * transition_score
  + w_motion * motion_score
```

The weights should be learned or tuned from validation data.

Do not assume:

``` text
body = 60%
face = 30%
time = 10%
```

without testing.

------------------------------------------------------------------------

# 16. Handling Missing Face

Face is optional.

### Scenario A

``` text
Body = good
Face = good
```

Use both.

### Scenario B

``` text
Body = good
Face = unavailable
```

Use:

``` text
Body + camera topology + time + motion
```

### Scenario C

``` text
Body = poor
Face = good
```

Face can dominate.

### Scenario D

``` text
Body = poor
Face = poor
```

Do not force a match.

Create an uncertain association or new identity.

------------------------------------------------------------------------

# 17. Confidence States

Every global association should have a state.

Example:

``` text
CONFIRMED
LIKELY
UNCERTAIN
NEW
REJECTED
```

The application should not treat every ReID match as equally reliable.

Example:

``` text
B_31 -> P1042

body similarity = 0.91
face similarity = 0.94
transition = valid
time = valid

=> CONFIRMED
```

Versus:

``` text
B_31 -> P1042

body similarity = 0.69
face = unavailable
transition = possible

=> UNCERTAIN
```

------------------------------------------------------------------------

# 18. Global Identity State Machine

A Global Person ID can have:

``` text
NEW
  |
  v
ACTIVE
  |
  v
TEMPORARILY_MISSING
  |
  +----> REIDENTIFIED
  |          |
  |          v
  |        ACTIVE
  |
  +----> EXITED
```

Example:

``` text
P1042
ACTIVE
last_seen = Camera A @ 12:05:10

Camera A lost track

TEMPORARILY_MISSING

Camera B sees B_31

ReID match

ACTIVE
P1042 -> B_31
```

------------------------------------------------------------------------

# 19. Identity Gallery

The identity gallery is the central store of observations.

Example:

``` text
Global ID P1042

Body:
  E1
  E2
  E3
  E4

Face:
  F1
  F2

Last camera:
  B

Last timestamp:
  12:07:41
```

Gallery entries should contain metadata:

``` text
embedding
camera_id
timestamp
track_id
quality_score
viewpoint
bounding_box
```

------------------------------------------------------------------------

# 20. Database Architecture

Recommended starting stack:

``` text
PostgreSQL
+
pgvector
```

Suggested logical tables:

## persons

``` text
global_id
created_at
last_seen_at
status
first_camera
last_camera
```

## camera_tracks

``` text
track_id
camera_id
global_id
start_time
end_time
first_bbox
last_bbox
confidence
```

## person_embeddings

``` text
id
global_id
track_id
camera_id
timestamp
embedding
quality_score
viewpoint
```

## face_embeddings

``` text
id
global_id
track_id
camera_id
timestamp
embedding
quality_score
```

## transitions

``` text
id
global_id
from_camera
to_camera
from_time
to_time
match_score
match_method
confidence
```

------------------------------------------------------------------------

# 21. Vector Search

For a new body embedding:

``` text
new embedding
      |
      v
pgvector nearest-neighbor search
      |
      v
candidate identities
```

However, vector similarity should be combined with application
constraints.

Do not blindly accept the nearest vector.

Example:

``` text
Nearest ReID match:
P1042 = 0.91

But:

A -> B transition impossible

Therefore:
reject P1042
```

------------------------------------------------------------------------

# 22. Camera Topology

Represent the physical camera layout as a directed graph.

Example:

``` text
        B
       / \
      /   \
     A     D
      \   /
       \ /
        C
```

Data:

``` text
camera A -> camera B
camera A -> camera C
camera B -> camera D
camera C -> camera D
```

Each edge can contain:

``` text
minimum_transition_time
maximum_transition_time
expected_transition_time
allowed_direction
zone
```

This is extremely useful for preventing false ReID matches.

------------------------------------------------------------------------

# 23. Why Pure ReID Similarity Is Not Enough

Two people may wear:

``` text
black shirt
blue jeans
white shoes
```

Their body embeddings may be close.

If the system simply does:

``` text
argmax(cosine_similarity)
```

it can produce:

``` text
Person A -> Person B
```

This creates a global ID switch.

Therefore the system must combine:

``` text
appearance
+
time
+
camera topology
+
movement
+
track continuity
+
face when available
```

------------------------------------------------------------------------

# 24. Camera Calibration

For production deployment, document:

-   Camera position
-   Camera orientation
-   Field of view
-   Resolution
-   FPS
-   Entry/exit zones
-   Overlap with neighboring cameras
-   Expected walking direction
-   Approximate transition time

Define zones:

``` text
Camera A

+-------------------------+
|                         |
|       tracking          |
|                         |
|                         |
| EXIT -----------------> |
+-------------------------+
```

Then:

``` text
Camera B

<---------------- ENTRY
+-------------------------+
|                         |
|                         |
+-------------------------+
```

This allows better handoff matching.

------------------------------------------------------------------------

# 25. Tracking Frequency vs ReID Frequency

Do not run ReID on every video frame.

Example:

``` text
Camera = 30 FPS

Tracking:
30 FPS

ReID:
2-5 observations/sec per active track
```

Depending on the GPU and scene, even lower rates may be sufficient.

Face extraction can be less frequent because high-quality face
observations are more valuable than thousands of nearly identical
frames.

------------------------------------------------------------------------

# 26. Track-Level Embedding Aggregation

Instead of:

``` text
Track
 -> one embedding
```

use:

``` text
Track
 |
 +-- embedding 1
 +-- embedding 2
 +-- embedding 3
 +-- embedding 4
 |
 v
Track representation
```

Possible aggregation methods:

-   quality-weighted average
-   top-K highest-quality embeddings
-   medoid embedding
-   multiple-view gallery

For the first version, top-K quality-filtered embeddings are simple and
robust.

------------------------------------------------------------------------

# 27. Example End-to-End Scenario

## Camera A

Person enters.

``` text
Track A_17
```

System collects:

``` text
E1
E2
E3
```

and assigns:

``` text
P1042
```

## Camera A exit

``` text
P1042
last_seen = 12:05:10
exit_zone = north
```

## Camera B

Four seconds later:

``` text
Track B_31
```

Candidate generation:

``` text
P1038
P1042
P1047
```

ReID:

``` text
P1038 = 0.62
P1042 = 0.89
P1047 = 0.67
```

Face unavailable.

Camera transition:

``` text
A -> B = valid
```

Time:

``` text
4 seconds = valid
```

Result:

``` text
B_31 -> P1042
```

## Camera C

Later:

``` text
Track C_08
```

Face visible.

``` text
Body similarity to P1042 = 0.83
Face similarity to P1042 = 0.96
C is valid next camera
```

Result:

``` text
C_08 -> P1042
```

The final journey:

``` text
P1042

Camera A
  A_17
    |
    v
Camera B
  B_31
    |
    v
Camera C
  C_08
```

------------------------------------------------------------------------

# 28. Handling Simultaneous Camera Visibility

Sometimes the same person can appear in Camera A and Camera B at the
same time.

This is not necessarily a new person.

If cameras overlap:

``` text
Camera A  <----> Camera B
```

then global association should permit:

``` text
P1042
  |
  +-- A_17
  |
  +-- B_31
```

simultaneously.

This is another reason why a global identity engine is preferable to a
simple sequential handoff algorithm.

------------------------------------------------------------------------

# 29. ID Switch Prevention

Important safeguards:

1.  Do not change global identity from one weak observation.
2.  Require multiple observations when possible.
3.  Use hysteresis.
4.  Keep a short-term track memory.
5.  Use camera transition constraints.
6.  Use face evidence when available.
7.  Reject impossible transitions.
8.  Maintain uncertainty rather than forcing a match.

Example:

``` text
Current:
B_31 -> P1042

New observation:
similarity P1043 > P1042

Do NOT immediately switch.

Require sustained evidence:

P1043 wins for N observations
AND
transition/time are plausible
```

------------------------------------------------------------------------

# 30. Metrics

The project should not be evaluated only on detector accuracy.

Track:

## Detection

-   Precision
-   Recall
-   mAP

## Single-camera tracking

-   IDF1
-   HOTA
-   ID switches
-   MOTA/MOTP where appropriate

## ReID

-   Rank-1 accuracy
-   mAP
-   CMC
-   cross-camera retrieval accuracy

## Global system

Most important:

-   Global IDF1
-   Cross-camera ID switch rate
-   False merge rate
-   False split rate
-   Track fragmentation
-   Correct camera transition rate
-   Unknown/uncertain rate
-   End-to-end latency
-   GPU utilization
-   FPS per camera

------------------------------------------------------------------------

# 31. Critical Business Metric

For this application, define:

``` text
Global ID Accuracy
```

Example:

Ground truth:

``` text
Person 17:
A -> B -> D -> E
```

System:

``` text
P1042:
A -> B -> D -> E
```

Correct.

But:

``` text
P1042:
A -> B

P1091:
D -> E
```

is a global identity fragmentation problem even if each individual
camera tracker is perfect.

------------------------------------------------------------------------

# 32. Recommended MVP

Do NOT start with 20 cameras.

Start with:

``` text
2 cameras
10-30 people
```

Pipeline:

``` text
Camera A
  |
  +-- detector
  +-- BoT-SORT/BoT-SORT
  +-- OSNet x1.0
  |
  v
Global Identity Manager
  ^
  |
  +-- OSNet x1.0
  +-- tracker
  |
Camera B
```

Ignore face initially.

First prove:

``` text
A_17 -> P1042
B_31 -> P1042
```

reliably.

Then add face.

------------------------------------------------------------------------

# 33. Development Phases

## Phase 1 --- Single camera

Implement:

``` text
video
 -> detector
 -> tracker
 -> visualization
```

Output:

``` text
Track A_1
Track A_2
Track A_3
```

## Phase 2 --- Person ReID

Add:

``` text
track
 -> person crop
 -> OSNet x1.0
 -> embedding
```

Store embeddings.

## Phase 3 --- Two-camera ReID

Implement:

``` text
Camera A
 -> gallery

Camera B
 -> query gallery
```

Measure cross-camera matching.

## Phase 4 --- Camera topology

Add:

``` text
A -> B
A -> C
B -> D
C -> D
```

and transition timing.

## Phase 5 --- Identity state machine

Implement:

``` text
NEW
ACTIVE
TEMPORARILY_MISSING
REIDENTIFIED
EXITED
```

## Phase 6 --- Face recognition

Add:

``` text
face detector
 -> alignment
 -> face embedding
 -> face gallery
```

## Phase 7 --- Score fusion

Combine:

``` text
body
+
face
+
time
+
camera topology
+
motion
```

## Phase 8 --- Database

Deploy:

``` text
PostgreSQL
pgvector
```

## Phase 9 --- Real-time optimization

Optimize:

``` text
ONNX
TensorRT
batching
GPU streams
frame skipping
embedding frequency
```

## Phase 10 --- Production hardening

Add:

-   monitoring
-   logging
-   health checks
-   camera reconnect
-   model versioning
-   identity audit trail
-   privacy controls
-   retention policy
-   access control

------------------------------------------------------------------------

# 34. Recommended Initial Technology Stack

The initial implementation should use this exact model stack:

```text
1. Person Detection
   RF-DETR-L
   object_detection_yolox_2022nov.onnx

2. Local Multi-Object Tracking
   BoT-SORT
   pinned MIT implementation

3. Person ReID
   OSNet x1.0
   person_reid_youtu_2021nov.onnx

4. Body Quality / Pose
   OpenCV Zoo RTMPose-M
   pose_estimation_mediapipe_2023mar.onnx

5. Body Visibility / Occlusion
   PPHumanSeg
   human_segmentation_pphumanseg_2023mar.onnx

6. Face Detection
   YuNet
   face_detection_yunet_2023mar.onnx

7. Face Recognition
   project-owned custom face embedding model
   trained from scratch using commercially licensed/customer-owned data
```

### Runtime

```text
Python
OpenCV
ONNX Runtime / OpenCV DNN
PostgreSQL
pgvector
FastAPI
```

Do not use an Ultralytics model/tracker in the approved stack unless its separate licensing requirements are deliberately accepted. The Ultralytics-integrated tracking configuration is under AGPL-3.0, which is why the architecture uses a standalone MIT BoT-SORT implementation instead. citeturn0search15turn0search0

# 35. Model Licensing Policy

Every model must have a recorded license entry:

``` text
model_name
repository
version/commit
code_license
weights_license
training_data_license
commercial_use_allowed
redistribution_allowed
attribution_required
license_url
verification_date
```

Example:

``` text
OSNet x1.0
code: MIT
weights: verify exact checkpoint
commercial use: verify exact checkpoint
```

Do not rely only on:

``` text
"GitHub says MIT"
```

because the repository, pretrained weights, and datasets can have
different terms.

------------------------------------------------------------------------

# 36. Privacy and Security

Because this system tracks people across physical spaces, production
design must include:

-   lawful deployment basis appropriate to the jurisdiction/use case
-   clear data retention policy
-   access control
-   encryption at rest/in transit
-   audit logs
-   restricted access to face embeddings
-   deletion mechanisms
-   minimal retention
-   separation of operational identity from unnecessary personal
    information

Do not store raw face images indefinitely unless there is a justified
requirement.

Prefer storing only the minimum information needed for the application.

------------------------------------------------------------------------

# 37. Recommended Final Architecture

``` text
                           MULTI-CAMERA SYSTEM
                                  |
        +-------------------------+-------------------------+
        |                         |                         |
        v                         v                         v
     Camera A                  Camera B                  Camera C
        |                         |                         |
        v                         v                         v
    Detector                  Detector                  Detector
        |                         |                         |
        v                         v                         v
    Tracker                   Tracker                   Tracker
        |                         |                         |
        +-------------------------+-------------------------+
                                  |
                                  v
                         TRACK OBSERVATIONS
                                  |
              +-------------------+-------------------+
              |                                       |
              v                                       v
        PERSON PIPELINE                         FACE PIPELINE
              |                                       |
          Person crop                              Face crop
              |                                       |
           OSNet x1.0/ReID                            Face model
              |                                       |
        Body embedding                         Face embedding
              |                                       |
              +-------------------+-------------------+
                                  |
                                  v
                       CANDIDATE GENERATOR
                                  |
                   +--------------+--------------+
                   |              |              |
                   v              v              v
              Camera graph      Time         Motion/zone
                   |              |              |
                   +--------------+--------------+
                                  |
                                  v
                         GLOBAL MATCHER
                                  |
              +-------------------+-------------------+
              |                   |                   |
              v                   v                   v
         Body similarity     Face similarity    Context score
              |                   |                   |
              +-------------------+-------------------+
                                  |
                                  v
                     GLOBAL IDENTITY MANAGER
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
             Existing identity             New identity
                P1042                         P1087
                    |
                    v
             IDENTITY GALLERY
                    |
                    v
             PostgreSQL + pgvector
                    |
                    v
             Analytics / Dashboard
```

------------------------------------------------------------------------

# 38. Final Recommendation

The current implementation should use the following fixed model stack.

## Model 1 — Person Detection

**RF-DETR-L**

```text
RFDETRLarge pretrained checkpoint
```

Purpose:

```text
frame -> person bounding boxes
```

License:

```text
Apache-2.0 model directory
```

RF-DETR-L is one of the core RF-DETR models and is explicitly listed as Apache-2.0, with 56.5 COCO AP and 6.8 ms T4 TensorRT FP16 latency in the current benchmark. citeturn1search0turn1search1

---

## Model 2 — Local Tracking

**BoT-SORT**

Purpose:

```text
person detections
      |
      v
A_17, A_18, A_19...
```

It maintains local camera track IDs.

License:

```text
MIT
```

Use a pinned standalone MIT implementation, not a framework distribution that changes the licensing stack. citeturn0search0turn0search2

---

## Model 3 — Person ReID

**OSNet x1.0**

```text
Exact pinned OSNet x1.0 pretrained checkpoint
```

Purpose:

```text
person crop
    |
    v
body appearance embedding
```

This is the primary cross-camera and same-camera re-identification feature extractor.

License:

```text
Implementation: MIT (Torchreid)
Exact pretrained weight: VERIFY before commercial deployment
```

Torchreid exposes the OSNet x1.0 pretrained checkpoint, but the repository does not by itself establish commercial rights for the distributed pretrained weight/training data. citeturn1search2turn1search4

---

## Model 4 — Body Pose / Visibility

**RTMPose-M**

```text
rtmpose-m_8xb256-420e_coco-256x192
```

Purpose:

```text
person crop
    |
    v
body keypoints
    |
    v
visibility / pose / body-completeness score
```

It is not an identity model.

License:

```text
Apache-2.0 model directory
```

citeturn2search0

---

## Model 5 — Human Segmentation

**PPHumanSeg**

```text
human_segmentation_pphumanseg_2023mar.onnx
```

Purpose:

```text
person crop
    |
    v
human mask
    |
    v
visible body area / occlusion quality
```

It is optional for the first lightweight prototype but approved for the architecture when stronger body-quality estimation is needed.

License:

```text
Apache-2.0 model directory
```

citeturn0search7

---

## Model 6 — Face Detection

**YuNet**

```text
face_detection_yunet_2023mar.onnx
```

Purpose:

```text
person crop
    |
    v
face box + landmarks
```

Used for:

- face localization
- face quality
- alignment
- pose estimation
- selecting the best face observations

License:

```text
MIT model directory
```

citeturn0search6

---

## Model 7 — Face Re-Identification

**Project-owned custom face embedding model**

We deliberately do not use a research-only pretrained face checkpoint.

Initial design:

```text
ResNet-18 architecture
       |
random initialization
       |
commercially licensed/customer-owned training data
       |
our own training
       |
project-owned checkpoint
```

Purpose:

```text
aligned face
    |
    v
face embedding
    |
    v
face gallery
```

This avoids importing an externally distributed research-only face checkpoint.

The exact training dataset and resulting checkpoint must be documented before this component is considered production-ready.

---

# 39. Complete Approved Model List

| # | Model | Exact artifact | Purpose | License |
|---|---|---|---|---|
| 1 | **RF-DETR-L** | `RFDETRLarge` pretrained checkpoint | Person detection | Apache-2.0 (core RF-DETR-L) |
| 2 | **BoT-SORT** | Pinned standalone implementation | Local tracking | MIT |
| 3 | **OSNet x1.0** | Exact pinned OSNet x1.0 pretrained checkpoint | Body/person ReID | Check exact checkpoint + training-data terms |
| 4 | **RTMPose-M** | `rtmpose-m_8xb256-420e_coco-256x192` | Body keypoints / quality | MMPose code Apache-2.0; verify exact checkpoint terms |
| 5 | **PPHumanSeg** | `human_segmentation_pphumanseg_2023mar.onnx` | Body visibility / occlusion | Apache-2.0 |
| 6 | **YuNet** | `face_detection_yunet_2023mar.onnx` | Face detection + landmarks | MIT |
| 7 | **SFace** | `face_recognition_sface_2021dec.onnx` | Face embeddings | OpenCV Zoo directory Apache-2.0; exact-weight provenance still requires clearance |

The OpenCV Zoo repository explicitly notes that individual model licenses must be checked separately rather than assuming the whole zoo has one universal license. citeturn1search0

---

# 40. Models Explicitly Excluded

The following are not part of the approved production stack:

```text
Ultralytics YOLO/track distribution
    -> AGPL/commercial licensing considerations

InsightFace pretrained recognition packages
    -> separate commercial licensing required for distributed pretrained models

Research-only ArcFace checkpoints
    -> not approved

None — OSNet x1.0 is the requested ReID model, but its exact pretrained checkpoint remains subject to license/training-data verification.

SFace
    -> requested face embedding model; OpenCV Zoo directory is Apache-2.0, but exact-weight commercial/training-data provenance remains an open verification item

Any random Hugging Face/GitHub checkpoint
    -> not approved without exact license verification
```

The project must not substitute any of these silently.

---

# 41. Model Change Procedure

If we later want to replace a model, use this process:

```text
Candidate model
      |
      v
Technical benchmark
      |
      v
Exact checkpoint identified
      |
      v
Code license checked
      |
      v
Weight license checked
      |
      v
Training-data/provenance checked
      |
      v
Commercial use confirmed
      |
      v
SHA-256 recorded
      |
      v
Only then add to approved model list
```

This keeps model selection separate from licensing approval.

A technically better model is not automatically a better production choice if its licensing is unclear.

---

# 42. Current End-to-End Model Pipeline

```text
RTSP Camera
    |
    v
YOLOX
Person Detection
    |
    v
BoT-SORT
Local Track ID
    |
    v
Observation Quality Layer
    |
    +-----------------------+
    |                       |
    v                       v
RTMPose-M                  PPHumanSeg
body keypoints          visible-body mask
    |                       |
    +-----------+-----------+
                |
                v
        Best Body Observations
                |
                v
           OSNet x1.0
                |
                v
        Body Embeddings
                |
                +-------------------+
                                    |
                                    v
                              Global Matcher
                                    ^
                                    |
                  +-----------------+
                  |
                  |
        Face Quality Pipeline
                  |
                  v
                YuNet
                  |
                  v
           Face + landmarks
                  |
                  v
           Best Face Frames
                  |
                  v
      Project-owned Face Embedding
                  |
                  v
           Face Embeddings
```

The Global Matcher then combines:

```text
body ReID
+
face ReID when available
+
camera topology
+
time
+
movement
+
track history
+
same-camera history
```

and follows the core rule:

```text
NEW LOCAL TRACK
       |
       v
TRY EXISTING GLOBAL IDs FIRST
       |
       +---- reliable match ----> REUSE EXISTING ID
       |
       +---- no reliable match --> CREATE NEW ID
```

This is the approved baseline architecture. Model changes should be discussed and benchmarked individually rather than silently replacing components.


# Video-File Development/Test Mode

Live camera feeds are not required for the initial implementation. Every video file is treated as a simulated camera/session source. The identity database is persistent across executions.

For each input video, the system must:

1. use the video filename as the source/provenance identifier;
2. detect and locally track people;
3. select high-quality observations;
4. extract body/face embeddings;
5. search the persistent Global Identity Gallery before creating a new ID;
6. reuse an existing Global ID when the match passes the calibrated decision rules;
7. create a new Global ID only when no reliable identity exists;
8. save every selected embedding with Global ID, source video name, frame number, timestamp, local track ID, quality score, viewpoint, and model version;
9. retain all historical observations instead of overwriting them;
10. annotate the output video with Global ID, local track ID, source video, match/new status, and confidence.

Example:

```text
video_01.mp4 -> P0001, P0002, P0003
video_02.mp4 -> P0002, P0003, P0004
video_03.mp4 -> P0001, P0004, P0005
```

The gallery therefore grows over time and provides the persistent identity source for every subsequent video run. PostgreSQL/pgvector is the source of truth; process restart must not erase learned identities.
