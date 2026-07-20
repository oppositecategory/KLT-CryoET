# KLT Picker 3D style guide

This repository follows a DeepMind-inspired, JAX-native scientific Python
style. The baseline references are the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
and the
[JAX contribution guide](https://docs.jax.dev/en/latest/contributing.html).
Repository rules in this document take precedence when those references leave
a choice open.

## Tooling

- Ruff is the formatter, import sorter, and linter.
- Pyrefly is the static type checker.
- Pytest is the test runner.
- Pre-commit runs repository hygiene checks and Ruff on changed Python files.
- Python 3.10 is the minimum supported language version.
- Lines are formatted to 88 characters with four-space indentation.
- Strings and docstrings use double quotes.

Install the development tools from the repository root:

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

Run the checks manually with:

```bash
ruff format --check kltpicker_3d
ruff check kltpicker_3d
pyrefly check
pytest
```

Use `ruff format kltpicker_3d` to format code. Use `ruff check --fix
kltpicker_3d` only after reviewing the proposed changes; unsafe automatic fixes
are not enabled.

## Imports

Imports are grouped in this order:

1. Standard library.
2. Third-party packages.
3. `kltpicker_3d` modules.

Use absolute package imports. Wildcard imports are prohibited. Import modules
or explicit symbols, and keep imports at module scope unless a delayed import
is required to avoid a documented optional dependency or import cycle.

```python
import jax
import jax.numpy as jnp
import numpy as np

from kltpicker_3d import spectral_estimation
from kltpicker_3d.fredholm_solver import solve_radial_fredholm_equation
```

Importing a module must not load data, initialize random state, allocate large
arrays, or start a computation.

## Naming

- Packages, modules, functions, methods, parameters, and local variables use
  `lower_snake_case`.
- Classes use `UpperCamelCase`.
- Constants use `UPPER_SNAKE_CASE`.
- Internal modules, functions, attributes, and constants start with `_`.
- Acronyms are lowercase inside identifiers: `rpsd`, `psd`, `acf`, and `nms`.
- Public APIs use descriptive names even when a paper uses short mathematical
  notation.
- Single-letter mathematical names are limited to short equation-oriented
  scopes. Document their meaning or cite the equation that defines them.
- Coordinates always use `(z, y, x)` ordering unless a public API explicitly
  documents otherwise.
- Names include units when ambiguity is possible, for example
  `particle_diameter_voxels`, `frequency_radians`, and
  `low_frequency_fraction`.

Prefer:

```python
def factorize_rpsd(patch_spectra: jax.Array) -> RpsdFactorization:
    ...
```

Avoid:

```python
def factorize_RPSD(S):
    ...
```

Renaming an existing public API requires either a compatibility alias with a
deprecation path or an explicitly documented breaking change.

## Type annotations

All public functions, methods, and classes are typed. New private functions
should also be typed.

- Use `jax.Array` for device-side JAX arrays.
- Use `numpy.typing.NDArray` for functions that intentionally operate only on
  NumPy arrays.
- Use `ArrayLike` only at public boundaries that intentionally accept several
  array implementations.
- Use modern unions such as `float | None`.
- Avoid `Any` unless an interoperability boundary makes it necessary.
- Document array shapes, axis order, dtype expectations, and units in
  docstrings because ordinary Python annotations do not express them fully.

Pyrefly currently uses its default preset while the existing package is being
annotated. Change it to the strict preset only after the package passes the
default check without suppressing legitimate errors.

## Documentation

Use Google-style docstrings. Every public module, class, function, and method
has a docstring. Private helpers need docstrings when their mathematical,
numerical, or shape behavior is not obvious.

```python
def estimate_rpsd(
    patches: jax.Array,
    max_distance: int,
) -> jax.Array:
    """Estimate an isotropic radial power spectral density.

    Args:
        patches: Tomogram patches with shape `(num_patches, z, y, x)`.
        max_distance: Maximum spatial ACF lag, in voxels.

    Returns:
        Radial PSD samples with shape `(num_radial_bins,)`.

    Raises:
        ValueError: If the patches are not cubic or `max_distance` is invalid.
    """
```

Scientific docstrings specify, where relevant:

- Array shape and axis order.
- Units and coordinate conventions.
- Supported dtypes and precision assumptions.
- Frequency range and normalization convention.
- Mathematical assumptions and invariants.
- The paper, equation, or derivation that establishes non-obvious notation.

Comments explain why a numerical or scientific choice is necessary. They do
not merely translate the following line into prose. Use `TODO(owner): reason`
for actionable temporary work; do not leave commented-out implementations.

## JAX code

Numerical kernels should be pure functions with explicit inputs and outputs.

- Pass PRNG keys explicitly. Never store a module-level or hidden global key.
- Split keys at the call site that consumes the new key.
- Do not call `jax.config.update` from library modules.
- Use `jax.numpy` for values that may be traced or transformed.
- Use NumPy and SciPy only for intentional host-side setup, I/O, or operations
  unavailable in JAX.
- Do not implicitly convert traced arrays with `np.asarray`.
- Use `jax.lax` control flow when a condition depends on traced values.
- Apply `jax.jit` at meaningful computational boundaries rather than to every
  small helper.
- Declare genuinely static arguments explicitly and keep shapes stable across
  repeated calls to avoid unnecessary recompilation.
- Preserve compatibility with `jax.jit` and `jax.vmap`; preserve
  differentiability where the mathematics supports it.
- Do not mutate JAX arrays or rely on hidden mutable state.
- Do not use Python `print` or logging inside compiled functions.
  `jax.debug.print` is for temporary debugging only.
- Respect the caller's precision configuration. Library code must not silently
  enable 64-bit mode or force an unrelated dtype.
- Derive tolerances and epsilon values from the active dtype when possible.
- Guard normalization, division, convergence ratios, and eigendecompositions
  against zero and near-zero values.

Host-side orchestration and device-side numerical kernels should be visibly
separate. Conversions between NumPy and JAX are explicit at those boundaries.

## Design and organization

- Each module has one clear scientific responsibility.
- Public exports from `kltpicker_3d.__init__` are deliberate and minimal.
- Large orchestration methods are decomposed into independently testable
  functions.
- Prefer immutable configuration dataclasses to long lists of loosely related
  constructor parameters.
- Separate numerical kernels, I/O, evaluation, and visualization.
- Reusable experiment code belongs in importable modules, not only notebooks.
- Notebooks consume the installed package and never modify `sys.path`.
- Generated results and large data files do not live beside package source.

Functions should remain focused. Extract a helper when a block has an
independent invariant, transformation boundary, or testable scientific role;
do not split functions solely to satisfy an arbitrary line count.

## Testing

Tests use descriptive behavior-oriented names:

```python
def test_nms_preserves_original_peak_scores() -> None:
    ...
```

Every public numerical routine should have:

- A representative normal-case test.
- Shape and dtype checks.
- A boundary or degenerate-case test.
- A mathematical invariant test when one exists.

Use tolerance-aware comparisons with justified tolerances. Test JIT execution,
vectorization, and deterministic random behavior when supported. Useful
scientific invariants include symmetry, Hermitian structure, non-negativity,
normalization, monotonic ordering, and conservation of variance or energy.

Every bug fix includes a regression test that fails for the old behavior.

## Migration policy

The existing package predates this guide, so adoption is incremental:

1. Keep the test suite green.
2. Format and clean imports without changing behavior.
3. Refactor one module at a time.
4. Add annotations and docstrings while refactoring.
5. Run formatting, linting, type checking, and tests after each module.
6. Enable stricter checks only after existing violations are resolved.

Formatting-only and semantic refactoring changes should be kept in separate
commits whenever practical so scientific changes remain reviewable.
