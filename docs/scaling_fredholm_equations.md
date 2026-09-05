# GPU System Optimization for Large-Scale 3D KLT Particle Detection

## Purpose

This document records the current GPU architecture, the scaling problem exposed by the EMPIAR-10045 experiment, and a concrete optimization plan for running the 3D KLT detector on full Cryo-ET tomograms.

The central issue is no longer only that the tomogram is too large to materialize on a GPU. Streaming solves that part. The new spectral analysis shows that an accurate KLT model contains far more angular and radial modes than the first full-volume experiment used. Expanding those modes over the spherical-harmonic index $m$ can produce hundreds of thousands of Cartesian templates. That changes the dominant constraints from tomogram storage to template storage, FFT workspace, convolution throughput, and checkpoint I/O.

The immediate objective is not to retain 99% of the theoretical covariance trace at any cost. It is to build a resource-aware system that can measure recall as the retained template budget grows, identify the useful operating region, and make each scoring run as efficient as possible.

## Terminology

The different indices and truncations must remain distinct:

- $K$ is the Gauss-Legendre quadrature order. It is currently 150 and determines the size of each radial Fredholm eigenproblem.
- `max_order` is an exclusive angular ceiling. For example, `max_order=135` solves $\ell=0,\ldots,134$.
- $n$ indexes the radial eigenpairs $\left(\lambda_{\ell n},R_{\ell n}\right)$ inside angular block $\ell$.
- $m=-\ell,\ldots,\ell$ indexes the spherical harmonics. Each $(\ell,n)$ radial mode expands into $2\ell+1$ Cartesian templates.
- Numerical-rank truncation removes eigenvalues that are indistinguishable from floating-point error.
- Energy truncation retains a leading globally ranked set of meaningful $(\ell,n)$ modes.
- The template cap limits the number of $m$-expanded Cartesian filters that can be scored.

The full covariance eigenfunctions have the form

$$
\psi_{\ell n m}(r,\theta,\phi)
=R_{\ell n}(r)Y_\ell^m(\theta,\phi).
$$

For fixed $(\ell,n)$, the eigenvalue $\lambda_{\ell n}$ is repeated for all $2\ell+1$ values of $m$. Its contribution to the covariance trace is therefore

$$
E_{\ell n}=(2\ell+1)\lambda_{\ell n}.
$$

The factor $2\ell+1$ is a true eigenvalue multiplicity, not an arbitrary weighting factor.

## EMPIAR-10045 problem scale

The full experiment uses EMPIAR-10045 tomogram 08 with the following representative geometry:

| Quantity | Value |
|---|---:|
| Tomogram shape $(z,y,x)$ | $(513,3710,3838)$ |
| Tomogram size | approximately 27.2 GiB |
| Patch side | 93 voxels |
| Patch grid | $(5,39,41)$ |
| Total patches | 7,995 |
| Template side | 97 voxels |
| Whitening radius | 37 voxels |
| Template radius | 48 voxels |
| Local-maximum halo | 1 voxel |
| Total scoring halo | 86 voxels |
| Representative core shape | $(186,186,186)$ |
| Representative loaded shape | $(358,358,358)$ |

The initial streamed RPSD and ALS stages do not need to be repeated when only the KLT angular ceiling or template budget changes. The valid reusable checkpoints are:

- `04_whitened_patch_rpsds.pkl`
- `05_whitened_als.pkl`

Fredholm modes, Cartesian templates, QR score filters, scores, and candidates must be regenerated whenever the spectral truncation or likelihood scale changes.

## What the high-order spectrum revealed

The high-order EMPIAR analysis uses the checkpointed whitened particle PSD, $K=150$, and angular orders $\ell=0,\ldots,179$. The theoretical trace is calculated independently from the PSD:

$$
\operatorname{tr}(C)=|B_a|C(0),
$$

with

$$
C(0)=\frac{4\pi}{(2\pi)^3}\int_0^c G(r)r^2\,dr.
$$

The accumulated Fredholm trace through $\ell=179$ agrees with this value to floating-point precision, confirming the corrected Fourier normalization and the $2\ell+1$ multiplicity accounting.

### Angular-ceiling convergence

| Target fraction of full trace | Required `max_order` | Largest included $\ell$ |
|---:|---:|---:|
| 80% | 67 | 66 |
| 90% | 98 | 97 |
| 95% | 116 | 115 |
| 99% | 135 | 134 |

These values describe how much spectrum is exposed by the angular solve. They do not state how many templates are affordable to score.

### Numerical radial rank

The numerical-rank threshold is

$$
\tau=K\,\epsilon_{\mathrm{float64}}
\max_{\ell,n}\lambda_{\ell n}.
$$

For the full 180-order sweep:

- Nominal radial solutions: 27,000.
- Strictly positive solutions: 3,839.
- Numerically resolved solutions: 3,705.
- Largest meaningful radial index: $n=53$.
- Fraction of trace discarded by the numerical threshold: approximately $3.3\times10^{-13}$.

For the 99% angular ceiling, $\ell=0,ldots,134$:

- Nominal radial solutions: 20,250.
- Numerically resolved radial modes: 3,594.
- Surviving radial modes per $\ell$: minimum 6, median 25, maximum 54.
- Expanding every resolved mode over $m$ would produce 339,076 templates.

The large requirement is therefore not a need to raise $K$ beyond 150. It is the combination of many meaningful $(\ell,n)$ modes and their $2\ell+1$ angular multiplicities.

### Global energy truncation and template cost

Modes are globally sorted by $\lambda_{\ell n}$. Retaining a complete $(\ell,n)$ multiplet costs $2\ell+1$ templates and gains $(2\ell+1)\lambda_{\ell n}$ trace energy. Sorting by eigenvalue is consequently also sorting by energy gained per Cartesian template.

Using the full high-order spectrum:

| Full-trace target | Radial modes | Expanded templates |
|---:|---:|---:|
| 80% | 404 | 26,762 |
| 90% | 832 | 79,130 |
| 95% | 1,393 | 134,841 |
| 99% | 2,322 | 208,188 |

Using only $\ell\le134$:

- All 3,594 resolved radial modes expand to 339,076 templates.
- Retaining 99% of the energy available under that ceiling uses 2,303 radial modes and 203,535 templates.
- Retaining 99% of the full theoretical trace uses 2,763 radial modes and 239,535 templates.

The second and third quantities differ because the angular ceiling itself exposes only approximately 99.098% of the full trace.

## What has actually been scored so far

The latest completed EMPIAR checkpoint used:

| Setting or result | Value |
|---|---:|
| `max_order` | 4 |
| Solved angular orders | $\ell=0,1,2,3$ |
| Configured energy fraction | 99% |
| Configured template cap | 1,000 |
| Available radial modes | 212 |
| Available expanded templates | 842 |
| Retained radial modes | 39 |
| Retained final templates | 153 |
| Retained fraction of exposed energy | 99.0809% |
| Retained fraction of full high-order trace | approximately 9.745% |

The apparent 99% retention applied only to the spectrum exposed by four angular orders. Those orders contain only approximately 9.835% of the full high-order covariance trace. This is the main reason that the next experiment must solve a much larger angular range before selecting a resource-constrained template set.

## Current multi-GPU architecture

### Streamed RPSD estimation

The tomogram is memory-mapped on the host and partitioned into fixed core regions. Haloed subvolumes are loaded only when needed. Static shapes allow JAX to compile one kernel geometry and reuse it across rounds. During RPSD extraction, independent subvolumes can be assigned across devices because each patch produces an independent radial spectrum.

### Fredholm solution and template construction

The radial Fredholm equations are solved independently for each $\ell$. Energy selection and the template cap continue to count the complete $2\ell+1$ multiplicity. The multi-GPU execution representation materializes only $m\geq0$: $m=0$ has multiplicity one, while each $m>0$ complex template represents its conjugate $\pm m$ pair with multiplicity two.

For a real whitened tomogram, the paired responses obey

$$
|r_{\ell n,-m}(x)|^2=|r_{\ell n,m}(x)|^2.
$$

Consequently, positive-$m$ quadratic and likelihood-offset contributions are multiplied by two. This preserves the complete covariance model while avoiding the negative-$m$ template, spectrum, and inverse FFT.

The high-order Fredholm solve is relatively cheap compared with full-volume scoring. It should be performed once at a deliberately high ceiling and cached. Different template budgets can then reuse the eigenpairs and repeat only template selection, template construction, QR, and scoring.

### Distributed $(\ell,m)$ block QR

For a fixed retained $(\ell,m)$ block with $m\geq0$, the radial templates are arranged as columns:

$$
A_{\ell m}
=\begin{bmatrix}
\psi_{\ell n_1m}&\cdots&\psi_{\ell n_{N_\ell}m}
\end{bmatrix}.
$$

Each block is independently factorized:

$$
A_{\ell m}=Q_{\ell m}R_{\ell m}.
$$

The small signal covariance is formed in this orthonormal basis and diagonalized to produce the final score basis, weights, adjusted eigenvalues, and likelihood offset. Blocks with equal radial width $N_\ell$ share a compiled shape. Up to one block per GPU is dispatched concurrently, and multiple rounds cover all blocks.

GPU assignment is handled by asynchronous JAX dispatch. There is no collective in the QR stage because the $(\ell,m)$ blocks are independent. Results are gathered to host and written directly into a memory-mapped output array.

The relevant implementation is `distributed_block_qr_score_parameters` in [`multi_gpu.py`](../kltpicker_3d/multi_gpu.py).

### Template-sharded scoring

The scoring stage currently shards templates, not space. For every streamed haloed subvolume:

1. The same subvolume is replicated to every GPU.
2. Each GPU holds a disjoint resident shard of the score templates and weights.
3. Each GPU computes the subvolume FFT and fused whitening spectrum locally.
4. Each GPU processes its templates in batches and accumulates one partial score volume.
5. `jax.lax.psum` performs one all-reduce over the partial score volumes.
6. Every GPU receives the complete score volume, but only replica zero performs local-maximum extraction.
7. Candidates are translated from local to global tomogram coordinates and accumulated on the host.
8. Global ranked NMS is applied after candidates from every subvolume have been collected.

For GPU $g$, the partial score is

$$
S_g(x)=\sum_{j\in\mathcal T_g}
\mu_j w_j\left|r_j(x)\right|^2,
$$

where $\mu_j=1$ for $m=0$ and $\mu_j=2$ for $m>0$.

The collective constructs

$$
S(x)=\sum_g S_g(x)-b,
$$

where $b$ is the likelihood offset.

The all-reduce appears as `jax.lax.psum` in `compute_fused_klt_score_shard`. The device axis and replicated/sharded arguments are defined by the enclosing `jax.pmap` in `score_candidates`.

### Fused FFT scoring

For each GPU and subvolume, the implementation computes:

$$
\widehat V=\mathcal F(V),
\qquad
\widehat V_w=\widehat V\,\widehat W,
$$

then, for every local template batch,

$$
r_j=\mathcal F^{-1}
\left(\widehat V_w\,\widehat H_j\right).
$$

The valid spatial region is cropped, squared magnitudes are weighted, and the batch is reduced directly into the partial score volume. The subvolume FFT is shared by all local templates, while template FFTs and inverse response FFTs are batched.

FFT convolution is appropriate for a $97^3$ kernel. A direct convolution over a representative $186^3$ output would require approximately

$$
186^3\times97^3\approx5.9\times10^{12}
$$

multiply-accumulates per template and subvolume. The main question is therefore not whether to replace FFT convolution with direct convolution, but how to remove avoidable FFT padding, workspace, batching, and memory-traffic costs.

## Checkpoint layout and storage behavior

The current experiment writes:

| Checkpoint | Contents |
|---|---|
| `06_templates.npy` | Original Cartesian templates |
| `06_template_metadata.npz` | Eigenvalues, orders, $m$ values, and truncation metadata |
| `06b_block_qr_templates.npy` | QR-transformed scoring templates |
| `06b_block_qr_score_model.npz` | Score weights, adjusted eigenvalues, normalization, and offset |

Both template arrays are currently `complex64` with shape `(template_count, 97, 97, 97)`. One template occupies approximately 6.96 MiB.

| Template count | Size of one template array | Two template checkpoints |
|---:|---:|---:|
| 153 | 1.04 GiB | 2.08 GiB |
| 1,000 | 6.80 GiB | 13.60 GiB |
| 4,000 | 27.20 GiB | 54.40 GiB |
| 8,000 | 54.40 GiB | 108.80 GiB |
| 208,188 | approximately 1.38 TiB | approximately 2.77 TiB |

Consequently, a literal 99%-energy model is not practical under the current Cartesian materialization strategy even before convolution begins.

The QR output is created as a host memory map, which avoids an additional complete in-memory output copy. Scoring currently creates padded host shards and transfers the complete resident template shard to every GPU before the subvolume loop. The templates are not reread for every subvolume, but the host shard construction still creates a substantial temporary copy.

## Current bottlenecks and limitations

### 1. Angular under-modeling in the completed run

The completed 153-template experiment used only $\ell=0,ldots,3$. It cannot test whether the KLT model works on this dataset when the dominant high-order covariance structure is represented.

### 2. Spherical-harmonic multiplicity

The number of final filters grows much faster than the radial-mode count because each radial mode costs $2\ell+1$ templates. At high $\ell$, retaining one radial mode may add hundreds of filters.

### 3. Resident template memory

`score_template_batch_size` controls only the number of simultaneous full-grid FFT responses. It does not stream the underlying spatial-template shard. Every GPU must currently hold its complete local shard.

### 4. FFT temporary memory

For loaded shape $(358,358,358)$, one `complex64` full-grid array occupies approximately 0.342 GiB. The current conservative planner assumes:

- five fixed complex full-grid arrays;
- eight complex full-grid arrays per batched template;
- the real input and score buffers;
- the resident spatial-template shard.

This corresponds to approximately 2.73 GiB of estimated additional memory per template in the FFT batch. The estimate does not exactly include cuFFT plan workspace, allocator fragmentation, or the actual XLA buffer-liveness schedule.

### 5. Unfriendly FFT dimensions

The representative side length has factorization

$$
358=2\times179,
$$

where 179 is prime. cuFFT cannot recursively decompose this axis into its most efficient small-radix kernels and may require a generic prime algorithm or Bluestein-style transform. Padding to

$$
360=2^3\times3^2\times5
$$

may be faster despite processing more voxels. This must be benchmarked rather than assumed.

### 6. Repeated transforms

Every GPU independently computes the identical subvolume FFT. Every template is transformed again for every subvolume because the padded template spectrum has the full loaded-subvolume shape. Precomputing all template spectra would replace computation with prohibitive memory use: one $358^3$ `complex64` spectrum is approximately 0.342 GiB per template.

### 7. Sequential spatial schedule during template sharding

All GPUs cooperate on one subvolume and then advance to the next. This is appropriate when templates do not fit on one GPU, but it can underuse spatial parallelism for small models such as the current 153-template checkpoint.

### 8. QR and file-system scaling

Each $(\ell,m)$ block has approximately $97^3$ rows and up to roughly 54 numerically meaningful radial columns in the measured spectrum. QR is independently parallelizable, but the number of blocks and total output bytes become large. Complex QR is unsupported by some JAX/GPU combinations, requiring a slower host LAPACK fallback.

### 9. Candidate extraction

The current local-maximum path constructs full-volume masks and applies a global `top_k` over the flattened core. This is only once per subvolume and is likely secondary to thousands of FFTs, but it may become visible for smaller template models.

### 10. Static analytical memory estimates

The planner uses fixed multipliers and a configurable memory fraction, currently 0.8. It predicts capacity but does not measure cuFFT scratch space or peak live memory. A failed allocation inside a replicated computation may abort the entire process, so trial-and-error in the production process is unsafe.

## Optimization strategy

### Phase 0: preserve correctness and reproducibility

Before optimizing throughput:

1. Reuse only the valid stage-4 and stage-5 PSD/ALS checkpoints.
2. Solve a high angular ceiling once and cache all radial eigenpairs.
3. Apply the numerical-rank tolerance before energy selection.
4. Preserve complete $m$ multiplets during selection.
5. Rebuild stage 6 and later checkpoints whenever the selected spectrum, Fourier normalization, or likelihood offset changes.
6. Compare single-device and multi-device scores on the same small crop to numerical tolerance.
7. Verify that template-sharded and unsharded scoring produce the same score volume, candidate ordering, and NMS output.

### Phase 1: make experiments budget-driven

Do not select 99% energy as the first production target. Solve through a high angular ceiling so the global mode ordering is correct, then constrain the deployed model by an explicit Cartesian-template budget.

The initial sweep should include at least:

| Template cap | Fraction of full trace in current analysis |
|---:|---:|
| 1,000 | 58.82% |
| 2,000 | 64.28% |
| 4,000 | 67.40% |
| 8,000 | 70.19% |
| 16,000 | 75.09% |
| 32,000 | 81.74% |

For each point record:

- retained radial modes and $m$-expanded templates;
- retained trace fraction;
- raw and QR checkpoint sizes;
- QR time;
- scoring time per subvolume;
- total experiment time;
- peak host and device memory;
- recall at $N$, $2N$, $5N$, and $10N$ picks;
- score-distribution and nonstationarity diagnostics.

The best scientific operating point is the smallest budget after which recall improvements flatten, not necessarily a fixed theoretical energy percentage.

### Phase 2: empirical FFT and memory calibration

Create a standalone scoring calibration program using one representative haloed subvolume and realistic score templates. It should run each configuration in a separate subprocess so an allocation failure cannot kill the controlling process.

Sweep:

- template batch sizes $1,2,4,8,\ldots$;
- exact FFT shape versus smooth padded alternatives;
- representative core shapes;
- template counts per device;
- allocator memory fractions;
- GPU models used by the experiments.

Measure after a warm-up compilation:

- steady-state time per template batch;
- time per template and per subvolume;
- peak device memory;
- cuFFT plan and scratch allocations;
- host-to-device transfer time;
- FFT, inverse FFT, elementwise, and reduction time;
- NCCL all-reduce time;
- achieved memory bandwidth and GPU utilization.

Recommended tools:

- JAX profiler traces for operation timelines;
- `jax.profiler.save_device_memory_profile` for live JAX allocations;
- Nsight Systems for CUDA API, kernel, transfer, and NCCL timelines;
- Nsight Compute for selected elementwise or reduction kernels;
- NVML sampling for process-level device memory;
- `/usr/bin/time -v` or an equivalent process monitor for peak host RSS.

JAX preallocation can hide allocation behavior. Profiling with preallocation disabled may help explain memory, but final throughput must also be measured under the production allocator configuration.

### Phase 3: improve FFT geometry and batching

1. Add an FFT-shape planner that considers nearby smooth dimensions rather than always using the exact loaded shape.
2. Benchmark candidate dimensions rather than relying only on prime factorization.
3. Select `score_template_batch_size` from an empirical per-GPU profile keyed by loaded shape, template shape, dtype, and device model.
4. Maintain a safety margin for allocator fragmentation and cuFFT workspace.
5. Cache the whitening spectrum for a static FFT shape if profiling shows a measurable benefit.
6. Investigate buffer donation for the loaded subvolume and score accumulators.
7. Preserve one local accumulation across all batches and perform only one all-reduce per subvolume, as the current implementation already does.

### Phase 4: choose the correct parallel decomposition

No single sharding strategy is optimal for every template budget.

#### Spatial sharding

Each GPU receives a different subvolume and a complete copy of the templates.

Advantages:

- multiple subvolumes are processed concurrently;
- no score all-reduce is required;
- no redundant subvolume FFT across GPUs.

Use when the complete scoring model and FFT workspace fit comfortably on every GPU. This is likely preferable for the current 153-template model and may remain practical for moderate budgets on large-memory GPUs.

#### Template sharding

All GPUs receive the same subvolume and disjoint template shards. Partial score volumes are all-reduced.

Advantages:

- template memory is divided across devices;
- all GPUs cooperate on the expensive template sum;
- only one complete score map is retained.

Use when the complete model cannot fit on one GPU. This is the current scoring implementation.

#### Hybrid two-dimensional sharding

Partition GPUs into groups. Each group processes a different subvolume, while GPUs inside a group shard templates and all-reduce their partial scores.

For eight GPUs, candidate layouts include:

- one group of eight: maximum template capacity, one subvolume at a time;
- two groups of four: two subvolumes concurrently;
- four groups of two: four subvolumes concurrently;
- eight groups of one: pure spatial sharding.

The planner should choose the smallest template-sharding group that fits the model and FFT batch, then use remaining groups for spatial parallelism. Prime GPU counts naturally provide fewer balanced layouts.

### Phase 5: reduce host and template-storage pressure

1. Avoid simultaneously materializing both raw and QR-transformed template arrays when reproducibility requirements allow one to be regenerated.
2. Preserve memory-mapped reads through sharding rather than making a complete padded host copy.
3. Transfer template shards asynchronously from pinned host buffers.
4. Double-buffer host-to-device template chunks if templates must be streamed.
5. Record exact checkpoint sizes before launching expensive configurations.
6. Consider compressed or lower-precision checkpoint storage only after validating score and recall stability.

Template streaming introduces a loop-order trade-off:

- Subvolume outermost: load a subvolume once but repeatedly transfer template chunks.
- Template chunk outermost: transfer a chunk once but revisit every subvolume and preserve partial score volumes between passes.

The first uses more PCIe bandwidth; the second needs large persistent partial-score storage or repeated host I/O. Profiling must determine which resource is limiting on the target machine.

### Phase 6: targeted custom kernels

Do not replace cuFFT or QR with handwritten kernels. The best candidates surround those library operations.

#### Fused post-IFFT accumulation

Fuse the following operations:

1. crop the valid response region;
2. calculate complex magnitude squared;
3. multiply by score weights;
4. reduce across the template batch;
5. accumulate directly into the partial score volume.

A Pallas or custom CUDA kernel may reduce intermediate materialization and global-memory traffic. It cannot eliminate the inverse FFT output or the fundamental one-response-per-template cost.

#### Local maxima and top-K

Replace the full-volume mask plus flattened global `top_k` with a two-stage kernel:

1. detect and compact local maxima per tile;
2. perform top-K on the compacted candidate list.

This optimization should be attempted only if profiling shows candidate extraction is material after FFT tuning.

#### RPSD radial reduction

If the initial RPSD stage becomes important for repeated datasets, fuse squared FFT magnitude and radial-bin accumulation to avoid materializing unnecessary full patch spectra. This does not address the dominant repeated scoring cost.

### Phase 7: algorithmic representations worth investigating

System optimization alone cannot make a 200,000-template Cartesian model cheap. Longer-term options include:

- a real spherical-harmonic basis, potentially replacing complex templates with real filters and enabling real FFT paths;
- factorized block scoring that avoids writing both raw and QR-transformed spatial bases;
- compressed or mixed-precision score filters with accuracy validation;
- adaptive energy budgets per angular order;
- score approximations that preserve the covariance quadratic form without explicitly convolving every expanded template;
- coarse-to-fine scoring in which a small basis identifies promising regions before applying a larger basis.

These are mathematical changes and require correctness experiments against the existing likelihood, not only performance benchmarks.

## Suggested experiment sequence

The next work should proceed in this order:

1. **High-order model cache:** retain the existing 180-order eigenvalue analysis and extend the production template-selection path to use the same numerical tolerance.
2. **Correctness crop:** compare single-device and template-sharded scoring on the same EMPIAR crop with identical templates.
3. **Memory calibration:** benchmark batch size and FFT shape in isolated subprocesses using the actual scoring geometry.
4. **First high-order budget:** construct and score a 1,000-template model selected from the high-order spectrum.
5. **Recall measurement:** compare it with the existing 153-template, low-angular-order result.
6. **Budget sweep:** attempt 2,000, 4,000, and 8,000 templates only when memory and storage estimates permit.
7. **Parallel-layout comparison:** compare spatial, template, and hybrid sharding at budgets that fit multiple layouts.
8. **Kernel optimization:** use profiler evidence to decide whether post-IFFT fusion or candidate compaction is worthwhile.

## Acceptance criteria

An optimized configuration should satisfy all of the following:

- Scores agree with the reference implementation within a documented numerical tolerance.
- Candidate coordinates and global NMS are invariant to GPU count and subvolume partitioning, apart from explicitly handled floating-point ties.
- The selected mode set preserves complete $m$ multiplets.
- The reported retained energy is relative to the full high-order trace, not only the currently exposed angular spectrum.
- No stage relies on retrying an unsafe replicated OOM in the production process.
- Peak host memory, peak device memory, checkpoint size, and steady-state time are logged before or during every full experiment.
- Template batch size and FFT shape are chosen from measured device-specific data.
- Checkpoints encode the Fourier normalization, angular ceiling, numerical tolerance, energy target, template cap, QR method, and score-model version.

## Immediate recommended configuration

For the next correctness-oriented full experiment:

1. Solve and cache through `max_order=180` so global ranking is not biased by a low angular ceiling.
2. Apply the numerical eigenvalue threshold before global sorting.
3. Start with a hard cap of 1,000 complete Cartesian templates rather than requesting 99% energy.
4. Let the scoring batch-size planner choose a value only after an empirical calibration for the actual GPU and subvolume geometry.
5. Compare exact and smooth FFT shapes, including 358 and 360 for the representative geometry.
6. Use spatial sharding if the 1,000-template model fits on every device; otherwise use the existing template-sharded all-reduce path.
7. Treat recall improvement relative to the 153-template baseline as the decision signal for increasing the budget.

This configuration does not attempt to solve the final capacity problem. It establishes whether the newly discovered high-angular-order spectrum materially improves particle ranking before investing in much larger template storage and scoring infrastructure.
