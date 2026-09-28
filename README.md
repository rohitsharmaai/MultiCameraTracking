# Multi-Camera Person Tracking & Re-Identification

Assigns a persistent Global Person ID (`P0001`, `P0002`, …) to each person across cameras and across separate video runs. A local tracker ID such as `A_17` is **not** an identity.

The pipeline answers: who is this person, which camera sees them, which local track they are, which cameras they visited, and whether a fragmented or returning track can be rejoined to an existing identity. Weak matches stay `UNRESOLVED` / `UNCERTAIN` instead of forcing the nearest ID.

This repository implements `Doc/multi_camera_person_reid_development_instructions.md`. Test videos are listed in `DATA.md`.

## Features

- Multi-camera and multi-run identity: the same person keeps one Global ID after a camera change, a track break, or a later video
- Strict evidence matching: similarity thresholds, score margin, quality gates, temporal plausibility, camera topology, and identity hysteresis
- Commercially licensed model whitelist with download + SHA-256 verify (no silent checkpoint swaps)
- Persistent gallery in SQLite (local) or PostgreSQL + pgvector (production)
- CLI video processing, gallery inspection, annotated outputs, and an optional read-only HTTP API

---

# How to run

Do these steps in order, from the **project root** (the folder that contains `app/`, `src/`, `configs/`, and `DATA.md`).

## 1. Create a Python environment

Python 3.10+ is required. GPU is optional (CPU works, slower). Use **venv** or **conda**.

### venv

```bash
cd /path/to/MultiCameraTracking

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e ".[dev]"
```

### conda

```bash
cd /path/to/MultiCameraTracking

conda env create -f environment.yml
conda activate multicam-reid

pip install -U pip
pip install -e ".[dev]"
```

Recreate after `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
conda activate multicam-reid
```

If `rfdetr` fails to install, install it explicitly (required for person detection):

```bash
pip install rfdetr
```

## 2. Create `.env`

```bash
cp .env.example .env
```

For a first run on this machine, edit `.env` to use SQLite (no Docker required):

```bash
DATABASE_URL=sqlite:///./data/gallery.db
ENVIRONMENT=development
API_KEY=change-me
ALLOW_CPU_FALLBACK=true
MODELS_DIR=models
```

`data/gallery.db` is created automatically. Identities survive process restart.

For PostgreSQL + pgvector instead (production store):

```bash
docker compose -f docker/docker-compose.yml up -d postgres
```

Then in `.env`:

```bash
DATABASE_URL=postgresql+psycopg2://mct:mct@localhost:5432/mct
ENVIRONMENT=development
API_KEY=change-me
```

Create tables:

```bash
python scripts/init_db.py
```

Production (`ENVIRONMENT=production`) refuses SQLite.

## 3. Download and verify models

This downloads the exact approved weight files into `models/` and records SHA-256.

```bash
python scripts/download_models.py
python scripts/verify_models.py
```

Expected files after a successful download:

```text
models/osnet/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth
models/pphumanseg/human_segmentation_pphumanseg_2023mar.onnx
models/yunet/face_detection_yunet_2023mar.onnx
models/auraface/glintr100.onnx
models/rtmpose/rtmpose_body2d.onnx
```

RF-DETR-L weights are fetched by the `rfdetr` package the first time detection runs (`RFDETRLarge` only — do not use XL/2XL).

If VERIFY fails, that artifact is marked **REJECTED**. Do not load, ship, or substitute another checkpoint.

## 4. Download test videos

Place EPFL POM files in `data/` as described in `DATA.md`. Start with the laboratory 4-people sequence:

```bash
mkdir -p data
cd data
curl -L -o 4p-c0.avi https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c0.avi
curl -L -o 4p-c1.avi https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c1.avi
curl -L -o 4p-c2.avi https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c2.avi
curl -L -o 4p-c3.avi https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c3.avi
cd ..
```

These videos are for **local testing only** (EPFL research use). Do not ship them in a product.

## 5. Process video (this is the main command)

Activate the environment first (`source .venv/bin/activate` or `conda activate multicam-reid`) if it is not already active.

### One camera / one file

```bash
python -m app.process_video --video data/Lab/4p-c0.avi --camera lab-c0
# same as: --input data/Lab/4p-c0.avi --camera-id A
```

### Persistence check (stop, start, second video)

The gallery is on disk. Close the process, then:

```bash
python -m app.process_video --video data/Lab/4p-c1.avi --camera lab-c1
```

The same person should reuse `P0001`… if the match is strong. New people get the next ID.

### Four synchronized cameras (one run, shared clock, 25 fps)

```bash
python -m app.process_video \
  --run-name lab-4p \
  --synchronized \
  --fps 25 \
  --camera A=data/Lab/4p-c0.avi \
  --camera B=data/Lab/4p-c1.avi \
  --camera C=data/Lab/4p-c2.avi \
  --camera D=data/Lab/4p-c3.avi
```

### Optional flags

| Flag | Meaning |
|---|---|
| `--input PATH` / `--video PATH` | Single video file |
| `--camera-id A` | Camera name for `--input`/`--video` (default `A`) |
| `--camera lab-c0` | Same as `--camera-id` when used with `--video` |
| `--camera ID=PATH` | Repeatable. Example: `--camera A=data/Lab/4p-c0.avi` |
| `--synchronized` | Shared timestamps across cameras |
| `--fps 25` | Override file FPS |
| `--run-name lab-4p` | Name written into `video_runs` and output files |
| `--max-frames 200` | Process only the first N frames (useful for a smoke test) |

Smoke test (first 100 frames):

```bash
python -m app.process_video --video data/Lab/4p-c0.avi --camera lab-c0 --max-frames 100
```

### What you get

- Console: `run_id`, frame count, track count, Global IDs created/reused
- Annotated videos in `outputs/` (Global ID, local track, source file, match kind, confidence)
- Identities and embeddings in the database (`data/gallery.db` or PostgreSQL)

## 6. Inspect the gallery

```bash
python -m app.gallery --list
python -m app.gallery --show P0001
```

`--list` shows every Global ID, embedding counts, and which source videos contributed.

Reset (deletes all identities — needs confirmation):

```bash
python -m app.gallery --reset --yes
```

## 7. Operator web GUI (recommended)

This is the easiest way to see **new IDs vs re-identified IDs**.

```bash
conda activate multicam-reid
python -m app.web
```

Open http://127.0.0.1:8000

- **Delete all gallery data** clears Global IDs, embeddings, and saved crops.
- Choose a local video and click **Start processing**. The file name is the camera source (`4p-c0.avi`).
- **Play / Pause / Stop** control the live job. Play also resumes a paused video or starts the selected file.
- Check **Body embedding required** and/or **Face embedding required** to block Global ID assignment until that modality passes the quality gate. Unchecked keeps the default rule (body or face can be enough). The checkboxes apply immediately, including mid-run.
- After a Global ID is assigned, the same BoT-SORT local track keeps being monitored. A later full-body pose or a real face replaces the first (often partial) entrance crop as the stored base for that ID. Different viewpoints are kept; weaker same-pose crops are updated.
- After processing, download the annotated MP4 from the GUI (also written to `outputs/`). Drag the frame slider or click a log line to scroll back and see when a Global ID was assigned or changed. Colored ticks on the slider mark those events. **Follow live** returns to the current frame.
- Live view: bounding box, Global ID, local track, source file, match kind.
- Right panel: each Global ID with a **Person** crop and a **Face** crop (`no image` if that crop was never saved), source videos, and embedding counts.
- Play a second file (`4p-c1.avi`): if the same person is reused, the card says **Re-ID from another video**.
- In the same file, a broken local track that rematches shows **Re-ID in this video**.

First start loads the models (can take a minute). A full Lab video takes several minutes.

## 8. Run the HTTP API (optional)

The JSON API is for inspection. Video upload and reset also live under `/gui/*`.

```bash
export API_KEY=change-me
uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

```bash
curl http://127.0.0.1:8000/health
curl -H "X-API-Key: change-me" http://127.0.0.1:8000/persons
curl -H "X-API-Key: change-me" http://127.0.0.1:8000/persons/P0001
curl -H "X-API-Key: change-me" http://127.0.0.1:8000/transitions
```

Inspection routes except `GET /health` require header `X-API-Key`. The operator GUI at `/` and `/gui/*` is local and does not send embeddings.

| Method | Path |
|---|---|
| GET | `/health` |
| GET | `/cameras` |
| POST | `/cameras/{id}/start` |
| POST | `/cameras/{id}/stop` |
| GET | `/cameras/{id}/tracks` |
| GET | `/persons` |
| GET | `/persons/{id}` |
| GET | `/persons/{id}/timeline` |
| GET | `/persons/{id}/cameras` |
| GET | `/transitions` |
| GET | `/metrics` |

## 9. Run tests

No GPU and no weight files required:

```bash
source .venv/bin/activate   # or: conda activate multicam-reid
pytest
```

---

# If something fails

| Symptom | What to check |
|---|---|
| `Video not found` | Files are under `data/` and the path matches `--input` / `--camera` |
| `RF-DETR-L failed to load` | `pip install rfdetr` inside the active venv or conda env |
| `OSNet x1.0 checkpoint missing` (or YuNet / PPHumanSeg / AuraFace / RTMPose) | Re-run `python scripts/download_models.py` then `python scripts/verify_models.py` |
| `REJECTED` in `configs/models.yaml` | VERIFY failed. Do not substitute another file. Re-download the exact artifact |
| Empty gallery after a run | Check that detections exist (confidence in `configs/thresholds.yaml`) and that the process exited cleanly |
| SQLite vs Postgres confusion | `.env` `DATABASE_URL` must be the same for `process_video` and `gallery` |
| Production error about SQLite | Set `ENVIRONMENT=development` for local runs, or use PostgreSQL |

---

# Approved model stack

| Stage | Model | Artifact | License |
|---|---|---|---|
| Person detection | RF-DETR-L | `RFDETRLarge` / `rfdetr-large` | Apache-2.0 |
| Local tracking | BoT-SORT (tracker only) | motion/IoU, FastReID **off** | MIT |
| Body ReID | OSNet x1.0 | `osnet_x1_0_msmt17_...pth` from [kaiyangzhou/osnet](https://huggingface.co/kaiyangzhou/osnet) | MIT |
| Pose / quality | RTMPose-Body2d | `rtmpose_body2d-onnx-float` | Apache-2.0 |
| Body mask | PPHumanSeg | `human_segmentation_pphumanseg_2023mar.onnx` | Apache-2.0 |
| Face detection | YuNet | `face_detection_yunet_2023mar.onnx` | MIT |
| Face embedding | AuraFace-v1 | `glintr100.onnx` only | Apache-2.0 |

Do **not** use RF-DETR-XL/2XL, Ultralytics detectors, FastReID, or InsightFace buffalo/antelope packs.

# Configuration

| File | Role |
|---|---|
| `environment.yml` | Conda environment (`multicam-reid`, Python 3.10) |
| `.env` | Database URL, environment, API key |
| `configs/thresholds.yaml` | Detection, tracking, ReID, identity thresholds (uncalibrated defaults) |
| `configs/topology.yaml` | Directed camera graph |
| `configs/cameras.yaml` | Declared camera sources |
| `configs/models.yaml` | Runtime model registry |
| `models/MANIFEST.yaml` | On-disk license / SHA-256 audit |

`configs/models.yaml` and `models/MANIFEST.yaml` must agree on name, artifact, version, SHA-256, and status, or the process fails fast.

# Architecture (short)

```
RF-DETR-L → BoT-SORT (local IDs)
    → person crop → PPHumanSeg (crop only) → RTMPose-Body2d
    → OSNet x1.0 / YuNet + AuraFace
    → candidate generation → global matcher (body, face, time, topology, motion)
    → Global Identity Manager → SQLite (dev) or PostgreSQL + pgvector
```

Quality is a gate, not an additive match bonus. Weak matches stay `UNRESOLVED` / `UNCERTAIN`. Identity hysteresis blocks rapid ID switches.


## 0. One-command start (recommended)

Everything needed to run the app is wrapped in `start.sh`. From the project root:

```bash
./start.sh
```

That single command:

1. Checks that the **Data** drive is mounted (the project lives on it).
2. Starts PostgreSQL in Docker (`docker compose up -d postgres`) and waits until it is **healthy**.
3. Starts the operator web app on **http://127.0.0.1:8000** (uses the `multicam-reid` conda env).

Press **Ctrl+C** to stop the app. Other controls:

| Command | What it does |
|---|---|
| `./start.sh` | mount check -> PostgreSQL -> web app (all-in-one) |
| `./start.sh status` | show containers + whether the app answers `/health` |
| `./start.sh down` | stop the app and PostgreSQL |
| `./start.sh mount` | mount the NTFS **Data** drive (asks for sudo) |
| `./start.sh db` | start only PostgreSQL |
| `./start.sh app` | start only the web app |
| `./start.sh logs` | follow PostgreSQL logs |
| `./start.sh install` | refresh python deps (`pip install -e ".[dev]"`) + download models |

Overridable via `.env` or the environment: `CONDA_ENV`, `API_PORT`, `DATA_MOUNT`, `DATA_DEV`.

### Prerequisites (one time)

- Docker running, with the repo's compose project (`mct-postgres`).
- Conda env `multicam-reid` (or set `CONDA_ENV`).
- The project must sit on the **Data** NTFS partition, which must be mounted.

### Mount the Data drive automatically on boot

If the drive is not auto-mounted, the project folder appears empty after a restart.
Mount it once:

```bash
./start.sh mount
# or
sudo mount -t ntfs-3g -o uid=1000,gid=1000 /dev/nvme0n1p5 /media/parmpal/Data
```

To make it permanent, add this line to `/etc/fstab` (use your partition's UUID from `sudo blkid`):

```
UUID=90C2D29FC2D288BC  /media/parmpal/Data  ntfs-3g  defaults,nofail,uid=1000,gid=1000,windows_names  0  0
```

> Dual-boot tip: disable **Windows Fast Startup** and fully shut Windows down once,
> otherwise NTFS may be left "dirty" and Linux will refuse to mount it.

### Data & backups

- Identities (persons/embeddings) live in **PostgreSQL** (Docker volume `multicameratracking_pgdata`).
- Crop images live in **`~/.multicam_reid/gui_snapshots`** (`SNAPSHOT_DIR`).
- Nothing is deleted unless you delete it (see the delete controls in the GUI).

Back up:
```bash
docker exec mct-postgres pg_dump -U mct -d mct > ~/mct_backup.sql
cp -r ~/.multicam_reid/gui_snapshots ~/mct_snapshots_backup
```
