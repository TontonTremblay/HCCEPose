# Frozen-PBR BOP training adaptation

This folder ports the HCCEPose(BF) adaptation used to train object-specific pose
heads on frozen BOP-derived HOPEv2, HANDAL and HOT3D populations. It adds strict
input/geometry handling, front/back label generation, checkpoint recovery and a
fixed-budget training driver. **The upstream network, loss, PnP and renderer
source files are not edited.** Runtime adapters and their corrections are
explicitly described below.

Base revision: `f75cc4d5db2fc298fc7b853f8c75273757872382`.
`SOURCE_PROVENANCE.json` records hashes of the original adaptation utilities.
This port changes installation-path discovery and adds portable CLI entrypoints;
its hashes differ from the original experiment. Generate new smoke/contract
receipts for a new run. Do not point these utilities at an existing active run or
rewrite old checkpoint identities to bypass compatibility checks.

## Scope and scientific qualification

This is the released **ResNet34 training-example** configuration, **not** an exact
reproduction of the published EfficientNet-B4 BOP comparison or its reported
0.100s/image T-LESS result. The training recipe is:

- One independent BF network per object; ImageNet-pretrained ResNet34 encoder.
- No task-weight initialization from another model, including smoke checkpoints.
- Batch24, Adam LR0.0002 (constant), default betas0.9/0.999, no weight decay.
- CUDA AMP/dynamic loss scaling; **50,000 attempted iterations**. Successful
  optimizer updates and AMP skips are counted separately.
- Input crop256, output maps128, padding1.5, upstream DZI/photometric augmentation
  and BGR ImageNet normalization; seed42 and8 data-loader workers by default.
- Released adaptive coordinate losses with
  `3 * front_loss + 3 * back_loss + mask_loss`.
- Deterministic held-out synthetic checks every500 iterations to obtain the
  stored PnP-combination prior. Use the fixed final checkpoint, not test-best or
  real-validation-best weights. No real target-dataset supervision is added.

The current upstream BOP example selects `ransac+vvs+comb` with crop-scaled
reprojection tolerance, whereas the regular/PBR test example selects
`ransac+comb`. These are both released options, not interchangeable benchmark
protocols. This training port does not change either inference entrypoint or
claim the exact T-LESS submission settings. It does not include the separate
provided-detection test adapter or an asserted end-to-end runtime measurement.

## What is included

- `hccepose_data.py`: frozen COCO→BOP-frame adapter, strict image/map IO,
  symmetry normalization and original-pose front/back rendering.
- `hccepose_prepare_visibility.py`: labelled keypoints remain subject to strict
  cheirality and reprojection checks; only unlabelled COCO `v=0` points may be
  behind the camera. No sample is silently removed.
- `hccepose_label_clipping.py`: only nonpositive near planes are corrected to a
  small positive value. Positive-near renders use the upstream implementation.
  The original CAD origin and R/t/K remain unchanged.
- `hccepose_train.py`: label hash receipts, full checkpoint state (including
  optimizer, scaler, adaptive-loss histories and RNG), explicit resume and fixed
  final-endpoint training. NumPy loss-history dtypes are preserved.
- `hccepose_smoke.py`: bounded geometry/finite-update pilot with overlay previews.
- `hccepose_contract.py`: explicit operator review acknowledgment and synthetic
  GPU PnP/loss-serialization/checkpoint-IO contract. Full training never loads
  smoke weights.
- `hccepose_render_contract.py`: checks declared renderer-extension attestation
  before invoking training.
- `hccepose_visibility_gate.py`: original hash-bound extension-review validator,
  retained for workflows that independently review changes to that extension.

The wrapper deliberately rejects unreadable coordinate maps rather than allowing
upstream missing-file fallbacks to synthesize zero supervision. Valid empty masks
remain valid in full training. The geometry smoke requires visible foreground to
assess alignment. For offset symmetries, both depth surfaces are rendered at the
original pose; canonicalization changes the coordinate labels, not the depth
render pose. These are real adapter corrections—not a claim of untouched
end-to-end upstream behavior.

No datasets, generated maps, checkpoints, credentials, private registry URLs,
cluster schedulers or campaign-specific repair receipts are distributed here.
The code is provided under the repository's existing LGPL-2.1 license.

## Input format: an existing frozen export, not a new raw-BOP sampler

The bridge preserves the per-object image/instance populations of an existing
COCO export. It does **not** automatically resample raw BOP scene directories.
The original upstream scripts remain available for native BOP ingestion.

Expected source layout:

```text
frozen/obj_000001/
  geometry/
    obj_000001.ply            # original-frame triangular mesh, millimetres
    models_info.json         # BOP object metadata / symmetries, millimetres
    keypoints_3d.json         # {"keypoints_3d": [[x,y,z], ...]}, metres
  dataset/
    train/
      annotations.json
      images/<image.file_name>
      masks/<annotation mask file>
    val/
      annotations.json
      images/<image.file_name>
      masks/<annotation mask file>
```

Each split contains exactly one retained annotation per image. Standard COCO image
entries need `id`, `file_name`, `width`, `height`. Each annotation additionally
requires:

- `mask_file`: path relative to `dataset/<split>`.
- `bbox`: frozen top-left `[x,y,width,height]` crop source.
- `camera_intrinsics`: `fx`, `fy`, `cx`, `cy`, in native pixels.
- `camera_extrinsics`:4×4 camera-to-world transform in the original OpenGL/world
  convention, with translations in metres.
- `object_pose`:4×4 object-to-world transform in that same world frame, metres.
- `keypoints`: ordered flattened COCO `[u,v,visibility,...]`, matching the geometry
  keypoint order. Labelled keypoints must reproject within0.1px.
- Optional `bop: {"scene": "000001", "frame": 0}` identifiers. Distinct instances
  must have unique mask filename stems within a scene/split.

The conversion is `T_cv = diag(1,-1,-1,1) @ inv(T_camera_world) @ T_object_world`,
then translation metres→millimetres exactly once. **Do not insert a native BOP
object-to-camera pose into `object_pose`: that is a different schema.** No mesh
recentering or camera guessing occurs. Sources are read-only; the destination is
new. Training and validation RGB populations must be disjoint.

## Environment

Follow the upstream repository's environment instructions, including its pinned
PyTorch/OpenCV/Kornia/imgaug dependencies, `trimesh`, and extraction of the bundled
`bop_toolkit.zip`. Rendering requires a functioning NVIDIA EGL/OpenGL runtime.
A dataset checksum/CPU unit test is not a substitute for the GPU geometry smoke.
For minimal Linux containers, also install font support required by the renderer:
`apt-get install fontconfig libfontconfig1 libfreetype6 fonts-dejavu-core`.

```bash
unzip bop_toolkit.zip
python -m pip install -r bop_adaptation/requirements-dev.txt
export HCCEPOSE_ROOT="$PWD"       # optional; defaults to this checkout
export PYOPENGL_PLATFORM=egl
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
python -m pytest bop_adaptation/tests -q
```

The legacy `--allow-nonstandard-backbone` flag explicitly acknowledges this
ResNet34 adaptation recipe; it is retained for compatibility with its provenance.
It does not change the upstream repository's general backbone support.

## Example: new experiment, one object

Use absolute paths. Keep long label generation and training in a named tmux
session or your GPU scheduler; do not use a bare background process.

```bash
tmux new -s hcce-bop
# Run the following inside that session, from the repository root.
TOOLS="$PWD/bop_adaptation/scripts"
EXP="$PWD/work_dirs/bop_example"
SOURCE="/absolute/path/to/frozen/obj_000001"
mkdir -p "$EXP"

python "$TOOLS/prepare_frozen.py" inventory \
  --source-root "$SOURCE" --dataset hopev2 --object 1 --out "$EXP/source.json"
python "$TOOLS/prepare_frozen.py" prepare --source "$EXP/source.json" \
  --out "$EXP/smoke-v3/hopev2/obj_000001" --smoke
python "$TOOLS/hccepose_smoke.py" --root "$EXP/smoke-v3/hopev2/obj_000001" \
  --iterations 20 --allow-nonstandard-backbone
```

The inventory pins the files you supplied; it does not prove they match a prior
experiment. Check your chosen source/population before pinning it. The smoke uses
up to64 training and16 held-out samples. If AMP skips prevent20 actual updates,
inspect the pilot and use a **new** smoke workspace with a larger bounded budget
(at most100 attempts); never initialize full training from a pilot.

**Actually inspect all generated `review/*.jpg` overlays** for RGB/mask/coordinate
alignment and symmetric-branch correctness. Numeric reprojection consistency is
not proof of correct pose branches or complete visible boundaries. Only then:

```bash
python "$TOOLS/hccepose_contract.py" review \
  --smoke-root "$EXP/smoke-v3/hopev2/obj_000001" \
  --reviewer "YOUR_NAME" --acknowledge-reviewed-overlays
python "$TOOLS/hccepose_contract.py" contract \
  --smoke-root "$EXP/smoke-v3/hopev2/obj_000001" --experiment-root "$EXP"

python "$TOOLS/prepare_frozen.py" prepare --source "$EXP/source.json" \
  --out "$EXP/objects/hopev2/obj_000001/data"
python "$TOOLS/hccepose_label_clipping.py" \
  --root "$EXP/objects/hopev2/obj_000001/data" --allow-nonstandard-backbone
python "$TOOLS/hccepose_render_contract.py" \
  --root "$EXP/objects/hopev2/obj_000001/data" --allow-nonstandard-backbone
```

Keep the `EXP/objects/<dataset>/obj_XXXXXX/data` layout: the trainer locates
`EXP/contract/<dataset>/PASSED.json` relative to it. Geometry review must cover your
new data; a previous object's overlay is not automatic approval of every object.
The GPU ABI contract is dataset-scoped and can be reused only with unchanged code.
The example does not run real validation or any test export.

Checkpoints go to `.../obj_000001/work/last.pth`; `PROGRESS.json` includes attempted
iterations, successful updates and AMP skips. `COMPLETE.json` is written at the
fixed endpoint. To recover an interrupted **same-recipe** run, inspect it first,
then explicitly add `--resume` to `hccepose_render_contract.py`. The existing
execution lock, code/input/label identities and complete optimizer/scaler/loss/RNG
state are preserved. Exact mid-epoch sampler replay is not claimed. A truly fresh
run must use a new work tree; do not delete old evidence or mix identities.

## Port validation

CPU regression tests cover transform units/frame preservation, symmetry offsets,
strict missing-map behavior, visibility/cheirality, conditional clipping and
review/renderer attestation. The port retains the training function bodies from
the original experimental driver. Its new portable CLI/path glue is **not claimed
to have completed a new GPU training campaign**; run the smoke and contract on
your machine before production use.
