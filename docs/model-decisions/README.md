# Model decisions

No model may be substituted silently.

To replace a model, follow Section 33 of `Doc/multi_camera_person_reid_development_instructions.md`:

1. Record why the change is needed.
2. Benchmark candidates.
3. VERIFY the exact artifact license, weights, commercial use, and redistribution.
4. Record SHA-256.
5. Update `configs/models.yaml` and `models/MANIFEST.yaml`.
6. Update the development specification.
7. Drop a decision note in this directory.
