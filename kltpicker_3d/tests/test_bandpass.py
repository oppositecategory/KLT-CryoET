import numpy as np

from kltpicker_3d.utils import bandpass_filter_3d


def test_radial_bandpass_removes_low_and_high_3d_frequencies():
    size = 32
    z = np.arange(size)[:, None, None]
    low = np.broadcast_to(np.cos(2 * np.pi * 2 * z / size), (size,) * 3)
    middle = np.broadcast_to(np.cos(2 * np.pi * 6 * z / size), (size,) * 3)
    high = np.broadcast_to(np.cos(2 * np.pi * 15 * z / size), (size,) * 3)
    volume = low + middle + high

    filtered = np.asarray(
        bandpass_filter_3d(
            volume,
            low_fraction=0.2,
            high_fraction=0.2,
            normalize=False,
        )
    )
    spectrum = np.abs(np.fft.fftn(filtered))

    assert spectrum[2, 0, 0] < 1e-3
    assert spectrum[15, 0, 0] < 1e-3
    assert spectrum[6, 0, 0] > 0.49 * volume.size
    assert abs(filtered.mean()) < 1e-6


def test_bandpass_rejects_invalid_cutoffs():
    volume = np.zeros((5, 5, 5), dtype=np.float32)

    for low, high in [(-0.1, 0.1), (0.1, -0.1), (0.6, 0.4)]:
        try:
            bandpass_filter_3d(volume, low, high)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid cutoff fractions must be rejected")
