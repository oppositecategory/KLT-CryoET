# 3D KLT Particle Picking for Cryo-Electron Tomography

This repository develops a three-dimensional, reference-free particle detector
for cryo-electron tomograms using a Karhunen–Loève transform (KLT) model. It
extends the spectral-estimation and data-adaptive template construction ideas in
[the original KLT particle-picking work](https://arxiv.org/abs/1912.06500) to
three-dimensional volumes.

The implementation contains two execution paths:

- `KLTParticleDetector3D` is the compact, in-memory reference pipeline for
  tomograms that fit in device memory.
- `MultiGPUKLTParticleDetector3D` is the out-of-core implementation for large
  MRC tomograms. It streams haloed subvolumes from host storage and uses all
  local GPUs on one machine.

The code is research software under active development. The in-memory path is
useful for small experiments and algorithm validation; the streamed path is
intended for full experimental tomograms.

## Method overview

The detector learns its signal and noise statistics directly from the input
tomogram. At a high level, it performs the following steps:

1. Apply radial band-pass preprocessing.
2. Partition the tomogram into non-overlapping cubic patches.
3. Estimate a radial power spectral density (RPSD) for every patch.
4. Factor the patch spectra with alternating least squares (ALS) into particle
   and noise components.
5. Construct a whitening filter from the estimated noise PSD.
6. Repeat the RPSD estimation and ALS fit on the whitened data.
7. Solve radial Fredholm eigenproblems and combine their radial eigenfunctions
   with spherical harmonics to construct 3D KLT templates.
8. Evaluate the KLT likelihood score throughout the tomogram.
9. Rank local maxima and apply particle-scale non-maximum suppression (NMS).

The complete templates have the form

$$
\psi_{\ell,n,m}(r,\theta,\phi)
= R_{\ell,n}(r)Y_\ell^m(\theta,\phi),
$$

where $\ell$ is the angular degree, $n$ indexes radial eigenfunctions for a
fixed $\ell$, and $m=-\ell,\ldots,\ell$ indexes spherical harmonics.
Templates are retained by degeneracy-weighted KLT eigenvalue energy.

## Repository layout

```text
kltpicker_3d/
  tomogram.py              In-memory end-to-end detector
  streaming.py             Volume sources and reusable subvolume execution
  multi_gpu.py             Out-of-core multi-GPU KLT pipeline
  spectral_estimation.py   Autocorrelation and radial PSD estimation
  alt_least_squares.py     Particle/noise RPSD factorization
  fredholm_solver.py       Radial Fredholm eigensolver
  utils.py                 Filtering, interpolation, calibration, and NMS
  tests/                   Numerical, streaming, and end-to-end tests

experiments/
  EMPIAR-10045.py                  Full EMPIAR-10045 experiment
  EMPIAR-10045_whitening_kernel.ipynb

tomotwin_experiments/      Smaller validation and parameter studies
outdated_experiments/      Earlier exploratory notebooks retained for history
```

## Installation

Python 3.10 or newer is required.

Create an environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e '.[dev]'
```

JAX must be installed for the CUDA version available on the target machine.
The generic dependency may install a CPU build, so follow the current
[JAX installation instructions](https://docs.jax.dev/en/latest/installation.html)
when GPU execution is required. Verify the installation with:

```bash
python -c 'import jax; print(jax.__version__); print(jax.devices())'
```

## In-memory detector

Use `KLTParticleDetector3D` when the complete tomogram and intermediate score
volumes fit in memory. Input arrays use `(z, y, x)` order.

```python
import mrcfile
import numpy as np

from kltpicker_3d.tomogram import KLTParticleDetector3D

with mrcfile.open("tomogram.mrc", permissive=True) as mrc:
    volume = np.asarray(mrc.data, dtype=np.float32)
    voxel_size_angstrom = float(mrc.voxel_size.x)

detector = KLTParticleDetector3D(
    volume,
    particle_diameter=270.0,       # angstrom
    mgscale=1.0 / voxel_size_angstrom,
    num_particles=500,
    legendre_order=150,
    max_order=4,                   # ell = 0, 1, 2, 3
    template_energy_fraction=0.99,
    threshold=-1.0,
)

num_detected, particles_zyx_score = detector.process_tomogram()
```

The returned array has columns `(z, y, x, normalized_score)`. Internally,
`particle_diameter * mgscale` converts the physical diameter to voxels.

Default geometry is derived from the particle diameter $D$ in voxels:

- PSD patch side: largest odd integer not greater than $0.8D$.
- Fredholm support radius: $0.4D$.
- Template side: smallest odd cube enclosing the Fredholm support.
- NMS radius: $D/2$.

All of these defaults can be overridden independently.

## Large tomograms and local multi-GPU execution

The streamed implementation is designed for tomograms that cannot be resident
on a GPU. `MrcVolumeSource` memory-maps the MRC file, and
`MultiGPUSubvolumeProcessor` divides the spatial domain into non-overlapping
cores. Each operation requests the halo required by its finite spatial support.
Only the valid core output is retained, which prevents internal seams caused by
FFT circular-boundary assumptions.

This is distributed computation in the general sense, but it is specifically a
single-host, multi-device implementation: one Python/JAX process controls all
visible local GPUs. It is not a multi-node implementation.

The main experiment driver provides checkpointing, logging, and evaluation:

```bash
python -u experiments/EMPIAR-10045.py \
  --input /path/to/IS002_291013_008.mrc \
  --ground-truth /path/to/IS002_291013_008.coords \
  --dry-run
```

`--dry-run` validates the input, ground truth, devices, memory plan, subvolume
schedule, and geometry without running a numerical stage.

After activating an environment with GPU-enabled JAX, run the experiment with
explicit dataset paths:

```bash
env \
  PYTHONUNBUFFERED=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -u experiments/EMPIAR-10045.py \
    --input /path/to/IS002_291013_008.mrc \
    --ground-truth /path/to/IS002_291013_008.coords \
    --resume \
    --core-patch-shape 2 2 2 \
    --score-template-batch-size 1
```

Use `--results-dir` to select where checkpoints and analysis outputs are stored:

```bash
python -u experiments/EMPIAR-10045.py \
  --input /path/to/IS002_291013_008.mrc \
  --ground-truth /path/to/IS002_291013_008.coords \
  --results-dir /path/to/results \
  --resume
```

Ground-truth files are expected to contain three whitespace-separated columns
in `(x, y, z)` order. Internal coordinates use `(z, y, x)` order.

### Spatial cores, patches, and halos

These terms refer to different geometries:

- **PSD patch**: the cubic unit used to estimate one RPSD.
- **Core patch shape**: the number of PSD patches along `(z, y, x)` owned by
  one spatial subvolume.
- **Loaded subvolume**: the core plus the operation-specific halo on both sides.
- **Template batch**: the number of templates evaluated concurrently on each
  GPU; it is unrelated to the spatial core shape.

For EMPIAR-10045 tomogram 08, the current defaults produce a PSD patch side of
93. With `--core-patch-shape 2 2 2`, each core is therefore
`(186, 186, 186)` voxels. The current scoring halo is

$$
R_{\mathrm{score}}
=R_{\mathrm{whitening}}+R_{\mathrm{template}}+R_{\mathrm{local\ max}}
=37+48+1=86,
$$

so each GPU loads a `(358, 358, 358)` volume and returns only its central
`(186, 186, 186)` result.

At the outer tomogram boundary, missing values are padded to preserve a common
static shape across devices. At internal subvolume boundaries, the halo is read
from the real neighboring tomogram data.

### Template construction and block QR

For `max_order=4`, the detector includes $\ell=0,1,2,3$, giving

$$
\sum_{\ell=0}^{3}(2\ell+1)=16
$$

distinct `(ell, m)` groups. These are not merely 16 templates: each group
contains multiple radial modes indexed by `n`. In the current EMPIAR run, the
99% energy criterion retained 39 `(ell, n)` radial modes, which expand through
their `m` multiplicities to 153 complete `(ell, n, m)` templates.

The multi-GPU score model performs QR independently within each fixed `(ell,m)`
block while varying `n`, then diagonalizes the transformed signal covariance.
The resulting scoring basis is stored as complex64 and repartitioned evenly
over the GPUs.

During scoring, the subvolume FFT is reused. Each GPU multiplies it by its
template spectra and performs inverse FFTs for the spatial responses. With 153
templates and seven GPUs, each GPU receives approximately 22 templates.
`--score-template-batch-size 1` evaluates one of those local templates at a
time; larger values use batched inverse FFTs but increase peak memory sharply.

### Important memory controls

| Argument | Meaning |
|---|---|
| `--core-patch-shape Z Y X` | Explicit spatial core size in PSD-patch units |
| `--memory-fraction` | Fraction used by automatic spatial-core planning |
| `--resident-volume-copies` | Conservative multiplier in spatial memory planning |
| `--patches-per-microbatch` | Patch batch used inside streamed RPSD extraction |
| `--score-template-batch-size` | Templates processed concurrently per GPU |
| `--score-memory-fraction` | Memory budget for automatic template FFT batching |
| `--candidate-capacity-per-subvolume` | Highest local maxima retained before global NMS |

If `--core-patch-shape` is supplied, it overrides automatic spatial-core
selection. If `--score-template-batch-size` is omitted, the code estimates a
batch size from device memory. cuFFT also needs temporary workspace, so an
explicit batch size of one is the safest starting point on 11 GiB GPUs.

## EMPIAR-10045 pipeline stages and outputs

The experiment writes atomic checkpoints to
`results/empiar-10045-bandpass-block-qr` by default.

| Stage | Main output | Description |
|---:|---|---|
| 0 | `00_manifest.json` | Input, geometry, devices, and parameters |
| 0 | `00_bandpass_filter.npy` | Finite-support band-pass kernel |
| 1 | `01_initial_patch_rpsds.pkl` | Band-passed patch RPSDs and variances |
| 2 | `02_initial_als.pkl` | Initial particle/noise RPSD model |
| 3 | `03_whitening_filter.npy` | Finite band-pass/whitening kernel |
| 4 | `04_whitened_patch_rpsds.pkl` | Whitened patch RPSDs |
| 5 | `05_whitened_als.pkl` | Calibrated whitened particle/noise model |
| 6 | `06_templates.npy` | Expanded KLT templates |
| 6b | `06b_block_qr_templates.npy` | Block-orthogonal scoring basis |
| 6b | `06b_block_qr_score_model.npz` | Likelihood weights, eigenvalues, and offset |
| 7 | `07_candidates_top4096.npy` | Globally indexed local score maxima |
| 8 | `08_particles_top4096_zyx.npy` | Ranked picks after global NMS |
| 8 | `08_particles_xyz.csv` | User-facing coordinates and scores |
| 9 | `09_evaluation.json` | Recall and stage timings |
| 9 | `09_matches.csv` | Prediction-to-truth assignments |
| 9 | `09_final_result.pkl` | Combined final analysis object |

`--resume` loads completed stage checkpoints. Checkpoints are written only when
an entire stage completes; interruption in the middle of a stage restarts that
stage. Use `--overwrite` to intentionally replace existing incompatible
artifacts. Raw RPSDs from the earlier unfiltered implementation are not
compatible with the finite band-pass pipeline.

### Candidate handling and recall evaluation

Every spatial subvolume extracts elementary `3 x 3 x 3` local maxima and keeps
its highest configurable number of candidates. Their coordinates are converted
from local subvolume indices to global tomogram indices before aggregation.
Candidates from every subvolume and GPU are then globally sorted by score.

Particle-scale NMS is applied only after this global merge, so suppression also
works across subvolume boundaries and is independent of GPU execution order.
The default EMPIAR experiment requests exactly the known 454 particles to
measure recall first. The default matching and NMS radius is half the particle
diameter, approximately 59.3 voxels or 135 Å for tomogram 08.

## Running in the background

For long experiments, detach the process from the SSH terminal and retain all
launcher output:

```bash
mkdir -p results/empiar-10045-bandpass-block-qr

(
  echo "$(date) | launcher started"
  env \
    PYTHONUNBUFFERED=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python -u experiments/EMPIAR-10045.py \
      --input /path/to/IS002_291013_008.mrc \
      --ground-truth /path/to/IS002_291013_008.coords \
      --resume \
      --core-patch-shape 2 2 2 \
      --score-template-batch-size 1
  echo "$(date) | process exited with status $?"
) >> results/empiar-10045-bandpass-block-qr/background.log 2>&1 < /dev/null &

echo $! | tee results/empiar-10045-bandpass-block-qr/run.pid
```

Monitor both launcher and structured experiment logs with:

```bash
tail -F results/empiar-10045-bandpass-block-qr/background.log
tail -F results/empiar-10045-bandpass-block-qr/empiar-10045.log
```

Heavy Python, SciPy, and JAX imports can take tens of seconds before the first
structured message appears.

## Tests

Install the test dependency and run:

```bash
python -m pytest
```

The tests cover:

- PSD calibration and finite whitening support;
- radial Fredholm discretization and weighted orthogonality;
- streamed RPSD extraction versus in-memory reference calculations;
- haloed filtering versus filtering a complete volume;
- distributed `(ell,m)` block QR and covariance preservation;
- fused frequency-domain scoring versus sequential convolution;
- globally consistent coordinates and NMS across subvolumes;
- miniature in-memory and streamed end-to-end pipelines.

GPU-specific sharding tests are skipped when fewer than two devices are
available. Most numerical tests can run on CPU by setting:

```bash
JAX_PLATFORM_NAME=cpu python -m pytest
```

## Numerical and coordinate conventions

- Volumes and internal coordinates use `(z, y, x)` order.
- Deposited coordinate files and exported CSV files use `(x, y, z)` order.
- Real tomogram and finite-filter arrays use float32 where practical.
- FFT-domain arrays and the block-QR scoring basis use complex64.
- The code requires isotropic voxel spacing. Anisotropic MRC volumes must be
  resampled before processing.
- FFT convolution is evaluated with enough real-data halo to make the retained
  core equivalent to finite linear convolution. Padding is needed only beyond
  the physical tomogram boundary.

## Troubleshooting

### JAX sees no GPU

Confirm that the installed JAX build supports the machine's CUDA stack and that
`CUDA_VISIBLE_DEVICES` exposes the intended devices:

```bash
nvidia-smi
python -c 'import jax; print(jax.devices())'
```

The EMPIAR script refuses accidental CPU execution unless `--allow-cpu` is
passed explicitly.

### cuFFT plan or allocation failure

Reduce `--score-template-batch-size`, reduce `--core-patch-shape`, or lower the
automatic memory fractions. cuFFT scratch space is additional to the visible
input and template arrays. Another process occupying GPU memory can also cause
plan creation to fail.

### Resume starts the current stage again

Resume is stage-granular. Check that the stage's final checkpoint exists in the
selected results directory. A progress bar alone does not imply that a
checkpoint has been committed.

### Logging remains empty

First verify that the process exists, then test imports and storage access.
Network-mounted MRC files can place Python in uninterruptible I/O wait even when
all GPUs are healthy. The structured log begins after top-level imports.

## Research context

The repository is intended to support research into reference-free particle
detection in heterogeneous cryo-electron tomograms, including comparisons with
learned detectors and analysis of the learned particle/noise spectra. The
notebooks under `tomotwin_experiments` and `experiments` contain ongoing
parameter studies and diagnostic analyses; they should be treated as research
artifacts rather than stable public APIs.
