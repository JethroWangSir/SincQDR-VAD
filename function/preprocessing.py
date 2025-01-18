import random
import numpy as np
import torch

def shift_perturbation(waveform, min_shift_ms=-5.0, max_shift_ms=5.0, rng=None):
    """
    Perturbs audio by shifting the audio in time by a random amount between min_shift_ms and max_shift_ms.
    The final length of the audio is kept unaltered by padding the audio with zeros.


    Args:
        min_shift_ms (float): Minimum time in milliseconds by which audio will be shifted
        max_shift_ms (float): Maximum time in milliseconds by which audio will be shifted
        rng: Random number generator
    """

    shifted_waveform = waveform.clone()
    _rng = random.Random() if rng is None else rng
    shift_ms = _rng.uniform(min_shift_ms, max_shift_ms)
    shift_samples = int(shift_ms * 16000 // 1000)
    if abs(shift_samples) > waveform.shape[-1]:
        # TODO: do something smarter than just ignore this condition
        return
    if shift_samples < 0:
        shifted_waveform[:, -shift_samples:] = waveform[:, :shift_samples]
        shifted_waveform[:, :-shift_samples] = 0
    elif shift_samples > 0:
        shifted_waveform[:, :-shift_samples] = waveform[:, shift_samples:]
        shifted_waveform[:, -shift_samples:] = 0

    return shifted_waveform.float()


def white_noise_perturbation(waveform, min_level=-90, max_level=-46, rng=None):
    """
    Perturbation that adds white noise to an audio file in the training dataset.

    Args:
        min_level (int): Minimum level in dB at which white noise should be added
        max_level (int): Maximum level in dB at which white noise should be added
        rng: Random number generator
    """

    _rng = np.random.RandomState() if rng is None else rng
    noise_level_db = _rng.randint(min_level, max_level, dtype='int32')
    noise_signal = _rng.randn(*waveform.shape) * (10.0 ** (noise_level_db / 20.0))
    waveform += noise_signal

    return waveform.float()


def spec_augment(input_spec, length, freq_masks=2, time_masks=2, freq_width=15, time_width=25, mask_value=0.0, rng=None):
    """
    Zeroes out(cuts) random continuous horisontal or
    vertical segments of the spectrogram as described in
    SpecAugment (https://arxiv.org/abs/1904.08779).

    params:
    freq_masks - how many frequency segments should be cut
    time_masks - how many time segments should be cut
    freq_width - maximum number of frequencies to be cut in one segment
    time_width - maximum number of time steps to be cut in one segment.
        Can be a positive integer or a float value in the range [0, 1].
        If positive integer value, defines maximum number of time steps
        to be cut in one segment.
        If a float value, defines maximum percentage of timesteps that
        are cut adaptively.
    """

    # Random number generator
    _rng = random.Random() if rng is None else rng

    # Shape: (B, D, T)
    batch_size, num_freqs, num_time_steps = input_spec.shape

    for idx in range(batch_size):
        # Apply frequency masks
        for _ in range(freq_masks):
            x_left = _rng.randint(0, max(0, num_freqs - freq_width))  # Start index
            w = _rng.randint(0, freq_width)  # Random mask width
            input_spec[idx, x_left: x_left + w, :] = mask_value  # Apply mask

        # Apply time masks
        for _ in range(time_masks):
            # Adaptive time width calculation
            if isinstance(time_width, float) and 0 <= time_width <= 1:
                adaptive_width = max(1, int(length[idx] * time_width))
            elif isinstance(time_width, int):
                adaptive_width = time_width
            else:
                raise ValueError("time_width must be an integer or float in range [0, 1].")

            y_left = _rng.randint(0, max(1, length[idx] - adaptive_width))  # Start index
            w = _rng.randint(0, adaptive_width)  # Random mask width
            input_spec[idx, :, y_left: y_left + w] = mask_value  # Apply mask

    return input_spec


def spec_cutout(input_spec, rect_masks=5, rect_time=25, rect_freq=15, rng=None):
    """
    Zeroes out(cuts) random rectangles in the spectrogram
    as described in (https://arxiv.org/abs/1708.04552).

    params:
    rect_masks - how many rectangular masks should be cut
    rect_freq - maximum size of cut rectangles along the frequency dimension
    rect_time - maximum size of cut rectangles along the time dimension
    """

    if rng is None:
        rng = random.Random()

    sh = input_spec.shape
    assert len(sh) == 3, "Input spectrogram must have 3 dimensions: (B, D, T)"

    with torch.no_grad():  # Disable gradient computation for augmentation
        for idx in range(sh[0]):  # Loop over batch dimension
            for _ in range(rect_masks):
                # Randomly select rectangle's top-left corner
                rect_x = rng.randint(0, sh[1] - rect_freq)
                rect_y = rng.randint(0, sh[2] - rect_time)

                # Randomly select rectangle's size
                w_x = rng.randint(0, rect_freq)
                w_y = rng.randint(0, rect_time)

                # Zero out the selected rectangle
                input_spec[idx, rect_x : rect_x + w_x, rect_y : rect_y + w_y] = 0.0

    return input_spec
