import numpy as np
import matplotlib.pyplot as plt


def compute_dft_direct(input_signal: np.array):
    """
    Compute the Discrete Fourier Transform (DFT) of the given input sequence.
    This function uses the direct method of computing the DFT, which is
    inefficient for large input sequences.
    
    
    Args:
    input_sequence (list or np.array): The input sequence (time domain).

    Returns:
    np.array: The DFT of the input sequence (frequency domain).
    """
    n = len(input_signal)
    # Initialize output array with zeros
    output_sequence = np.zeros(n, dtype=complex)

    # Compute DFT
    for k in range(n):
        for j in range(n):
            angle = -2j * np.pi * j * k / n
            output_sequence[k] += input_signal[j] * np.exp(angle)

    return output_sequence

def generate_signal(num_points:int, max_freq:int =0):
    """
    Generate a complex signal composed of multiple sinusoidal waves.

    Args:
    num_points (int): Number of data points in the signal.
    max_freq (int): Maximum frequency of the sinusoidal components.
    noise (float): Noise level (gaussian noise).

    Returns:
    np.array: A complex signal composed of multiple sinusoidal components.
    """
    t = np.linspace(0, 1, num_points)
    breakpoint()
    signal = np.zeros(num_points, dtype=complex)
    for freq in range(1, max_freq+1):
        signal += np.exp(2j * np.pi * freq * t)
    return signal

def plot_signal_and_dft(signal, dft_result):
    """
    Plot the original signal and its DFT.

    Args:
    signal (np.array): The original signal.
    dft_result (np.array): The DFT of the signal.
    """
    num_points = len(signal)
    t = np.linspace(0, 1, num_points)
    freq = np.arange(num_points)

    plt.figure(figsize=(12, 6))

    # Plot the original signal
    plt.subplot(1, 2, 1)
    plt.plot(t, signal, label='Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Time Domain')
    plt.legend()

    # Plot the magnitude of the DFT result
    plt.subplot(1, 2, 2)
    plt.plot(freq, np.abs(dft_result), label='DFT Magnitude')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('Frequency Domain')
    plt.legend()

    plt.tight_layout()
    plt.show()




# Usage
if __name__ == "__main__":
    input_signal_example = np.array([0, 1, 2, 3])
    dft_result = compute_dft_direct(input_signal=input_signal_example)
    print(dft_result)
    
    
    # Generate a more complex signal
    num_points = 1024
    complex_signal = generate_complex_signal(num_points)

    # Compute its DFT
    dft_of_complex_signal = compute_dft_direct(complex_signal)
    
    # Plot the signal and its DFT
    plot_signal_and_dft(complex_signal, dft_of_complex_signal)