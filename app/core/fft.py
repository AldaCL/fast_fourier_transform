import numpy as np

def fft_cooley_tukey(signal: list):
    N = len(signal)
    if N <= 1: 
        return signal

    # Split: FFT of even and odd parts
    even_part = fft_cooley_tukey(signal[0::2])
    odd_part = fft_cooley_tukey(signal[1::2])

    # Combine: FFT of original signal
    combined = np.zeros(N, dtype=complex)
    # N/2 operations in the for cicle to add the two parts
    for k in range(N // 2):
        t = np.exp(-2j * np.pi * k / N) * odd_part[k]
        combined[k] = even_part[k] + t
        combined[k + N // 2] = even_part[k] - t

    return combined
