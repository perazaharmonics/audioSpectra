# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.signal import convolve
from scipy.fftpack import ifft, fftshift
from scipy.interpolate import interp1d
from multiprocessing import Pool
from functools import partial

######################################################################
## Discrete Wavelet Transform
## Description: The Daubechies wavelets are a family of orthogonal wavelets characterized by a maximal number of vanishing moments for some given support. In simpler terms, this means that the wavelets are designed to represent polynomial data of a given order efficiently.
## For the purpose of this discussion, let's look at the simplest of Daubechies wavelets: the D4, often referred to as db1. The D4 wavelet has two vanishing moments, which makes it optimal for representing piecewise constant functions.
## Low-Pass: The D4 wavelet is defined by four coefficients , h0, h1, h2, and h3. 
## High-Pass: the D4 wavelet is defined by the same coefficients, but in bit-reversed order and alternating signs: g3, g2, g1, and g0.
## Conclusion: These coefficients are used in a filter bank to either decompose a signal into approximation and detail coefficients (analysis) or reconstruct a signal from approximation and detail coefficients (synthesis).
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
######################################################################

######################################################################
## Discrete Wavelet Transform
## Description: The Discrete Wavelet Transform (DWT) is a time-frequency representation of signals. 
# It uses a filter bank to decompose a signal into approximation and detail coefficients. The DWT is computed by convolving the signal with the low-pass and high-pass filters, and then downsampling the result.
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
######################################################################

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(np.ceil(np.log2(x)))

def pad_to_pow2(signal):
    original_length = len(signal)
    padded_length = int(next_power_of_2(original_length))
    padding_zeros = padded_length - original_length
    padded_signal = np.pad(signal, (0, padding_zeros), mode='symmetric')
    return padded_signal, original_length

def remove_padding(signal, original_length):
    return signal[:original_length]

 # Scales the coeeficients to the range [0, 1]
def normalize_minmax(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

# Scales the coeeficients to have zero mean and unit variance
def normalize_zscore(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (data - mean_val) / std_val

def awgn(signal, desired_snr_db):
    """
    Add AWGN noise to a signal to achieve a desired SNR.
    
    :param signal: The clean input signal
    :param desired_snr_db: Desired SNR in dB
    :return: Noisy signal
    """
    
    # Calculate signal power
    signal_power = np.mean(signal ** 2)
    
    # Calculate desired noise power based on the desired SNR
    desired_noise_power = signal_power / (10 ** (desired_snr_db / 10))
    
    # Generate AWGN noise with the desired power
    noise = np.sqrt(desired_noise_power) * np.random.randn(*signal.shape)
    
    # Add noise to the signal
    noisy_signal = signal + noise
    
    return noisy_signal

def dwt_multilevel(signal, wavelet_func, levels):
    n = len(signal)
    n_pad = int(next_power_of_2(n))
    
    # Zero-padding to make the length a power of 2
    if n_pad != n:
        pad_values = n_pad - n
        signal = np.pad(signal, (0, pad_values), 'constant')
        
    coeffs = []
    current_signal = signal
    
    for i in range(levels):
        approx, detail = wavelet_func(current_signal)
        coeffs.append((approx, detail))
        current_signal = approx
        if len(current_signal) < 2:
            break  # Terminate if the signal length becomes less than 2
    
    return coeffs


def haar(signal):
    # haar low-pass filter coefficients
    h = [1 / np.sqrt(2), 1 / np.sqrt(2)]  # low-pass filter coefficients
    # haar high-pass filter coefficients
    g = [1 / np.sqrt(2), - 1 / np.sqrt(2)]  # high-pass filter coefficients
    
    
    # Initialize
    approx = []
    detail = []

    # Generate coefficients
    for i in range(0, len(signal) // 2):
        approx_sum = h[0]*signal[2*i] + h[1]*signal[2*i + 1]
        detail_sum = g[0]*signal[2*i] + g[1]*signal[2*i + 1]
        approx.append(approx_sum)
        detail.append(detail_sum)

    return approx, detail

def db1(signal):

    # db1 (D4) filter coefficients
    h = [(1 + np.sqrt(3))/4, (3 + np.sqrt(3))/4, (3 - np.sqrt(3))/4, (1 - np.sqrt(3))/4]  # low-pass filter coefficients
    g = [h[3], -h[2], h[1], -h[0]]  # high-pass filter coefficients

    # Initialize
    approx = []
    detail = []

    # Generate coefficients
    for i in range(0, len(signal) // 2):
        approx_sum = 0
        detail_sum = 0
        for k in range(len(h)):
            index = 2 * i + k - len(h) // 2 + 1
            if 0 <= index < len(signal):
                approx_sum += h[k] * signal[index]
                detail_sum += g[k] * signal[index]
        approx.append(approx_sum)
        detail.append(detail_sum)

    return approx, detail

def db6(signal):
    # db6 (D12) filter coefficients
    # db6 low-pass filter coefficients
    h = [
        -0.001077301085308,
        0.0047772575109455,
        0.0005538422011614,
        -0.031582039318486,
        0.027522865530305,
        0.097501605587322,
        -0.129766867567262,
        -0.226264693965440,
        0.315250351709198,
        0.751133908021095,
        0.494623890398453,
        0.111540743350109
    ]

    # db6 high-pass filter coefficients
    g = [
        h[11],
        -h[10],
        h[9],
        -h[8],
        h[7],
        -h[6],
        h[5],
        -h[4],
        h[3],
        -h[2],
        h[1],
        -h[0]
    ]


    

    # Initialize
    N = len(h)
    approx = []
    detail = []

    # Generate coefficients
    for i in range(0, len(signal) // 2):
        approx_sum = 0
        detail_sum = 0
        for k in range(N):
            index = 2 * i + k - N // 2 + 1
            if 0 <= index < len(signal):
                approx_sum += h[k] * signal[index]
                detail_sum += g[k] * signal[index]
        approx.append(approx_sum)
        detail.append(detail_sum)

    return approx, detail
#########################################################################
## Symlet Wavelet 5 (Sym5)
## Description: The Symlet wavelets are a family of orthogonal wavelets that are defined from the scaling function of the Daubechies wavelets. The Symlet wavelets are characterized by a maximal number of vanishing moments for some given support. In simpler terms, this means that the wavelets are designed to represent polynomial data of a given order efficiently.
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html
#########################################################################

def sym5(signal):
    # Sym5 filter coefficients
    h = [
        0.027333068345078,
        0.029519490925774,
        -0.039134249302383,
        0.199397533977394,
        0.723407690402421,
        0.633978963458212,
        0.016602105764522,
        -0.175328089908450,
        -0.021101834024759,
        0.019538882735287
    ]
    g = [h[i] * (-1)**i for i in range(len(h))][::-1]

    # Initialization
    approx = []
    detail = []
    
    # Generate coefficients
    for i in range(0, len(signal) // 2):
        approx_sum = 0
        detail_sum = 0
        for k in range(len(h)):
            index = 2 * i + k - len(h) // 2 + 1
            if 0 <= index < len(signal):
                approx_sum += h[k] * signal[index]
                detail_sum += g[k] * signal[index]
        approx.append(approx_sum)
        detail.append(detail_sum)
    
    return approx, detail

#########################################################################
## Symlet Wavelet 8 (Sym8)
## Description: The Symlet wavelets are a family of orthogonal wavelets that are defined from the scaling function of the Daubechies wavelets. The Symlet wavelets are characterized by a maximal number of vanishing moments for some given support. In simpler terms, this means that the wavelets are designed to represent polynomial data of a given order efficiently.
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html
#########################################################################

def sym8(signal):
    # Sym8 filter coefficients
    h = [
        -0.003382415951359,
        -0.000542132331635,
        0.031695087811492,
        0.007607487325284,
        -0.143294238350809,
        -0.061273359067908,
        0.481359651258372,
        0.777185751700523,
        0.364441894835331,
        -0.051945838107709,
        -0.027219029917056,
        0.049137179673476,
        0.003808752013890,
        -0.014952258336792,
        -0.000302920514551,
        0.001889950332900
    ]
    g = [h[i] * (-1)**i for i in range(len(h))][::-1]

    # Initialization
    approx = []
    detail = []
    
    # Generate coefficients
    for i in range(0, len(signal) // 2):
        approx_sum = 0
        detail_sum = 0
        for k in range(len(h)):
            index = 2 * i + k - len(h) // 2 + 1
            if 0 <= index < len(signal):
                approx_sum += h[k] * signal[index]
                detail_sum += g[k] * signal[index]
        approx.append(approx_sum)
        detail.append(detail_sum)
    
    return approx, detail

#########################################################################
## Coiflet Wavelet 5 (Coif5)
def coif5(signal):
    # Coif5 filter coefficients
    h = [
        -0.000720549445364,
        -0.001823208870703,
        0.005611434819394,
        0.023680171946334,
        -0.059434418646456,
        -0.076488599078311,
        0.417005184421393,
        0.812723635445542,
        0.386110066821162,
        -0.067372554721963,
        -0.041464936781959,
        0.016387336463522,
    ]
    g = [h[i] * (-1)**i for i in range(len(h))][::-1]

    # Initialization
    approx = []
    detail = []
    
    # Generate coefficients
    for i in range(0, len(signal) // 2):
        approx_sum = 0
        detail_sum = 0
        for k in range(len(h)):
            index = 2 * i + k - len(h) // 2 + 1
            if 0 <= index < len(signal):
                approx_sum += h[k] * signal[index]
                detail_sum += g[k] * signal[index]
        approx.append(approx_sum)
        detail.append(detail_sum)
    
    return approx, detail


#######################################################################################
## Synthesis
## Description: The synthesis filter bank is the inverse of the analysis filter bank. It is used to reconstruct a signal from approximation and detail coefficients.
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
#######################################################################################
def idwt_multilevel(coeffs, wavelet_func):
    signal = coeffs[0][0]  # Get the approximation coefficients of the highest level
    
    for i in range(len(coeffs) - 1, 0, -1):
        approx, detail = coeffs[i]
        signal = wavelet_func(approx, detail)
    
    return signal

#######################################################################################
## Synthesis Haar
## Description: The synthesis filter bank is the inverse of the analysis filter bank. It is used to reconstruct a signal from approximation and detail coefficients.
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
#######################################################################################


def inverse_haar(approx, detail):
    # Haar wavelet analysis filters
    h = [0.7071067811865476, 0.7071067811865476]
    g = [-0.7071067811865476, 0.7071067811865476]
    
    # Reversing filters for synthesis
    h_inv = h[::-1]
    g_inv = g[::-1]
    
    reconstructed_signal = []
    for a, d in zip(approx, detail):
        reconstructed_signal.extend([(a * h_inv[0] + d * g_inv[0]), (a * h_inv[1] + d * g_inv[1])])
    
    return reconstructed_signal



#######################################################################################
## Synthesis db1
## Description: The synthesis filter bank is the inverse of the analysis filter bank. It is used to reconstruct a signal from approximation and detail coefficients.
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
#######################################################################################

def inverse_db1(approx, detail):
 # Inverse Daubechies (db1) transform
    reconstructed_signal = []
    h0 = (1 + np.sqrt(3)) / (4 * np.sqrt(2))
    h1 = (3 + np.sqrt(3)) / (4 * np.sqrt(2))
    h2 = (3 - np.sqrt(3)) / (4 * np.sqrt(2))
    h3 = (1 - np.sqrt(3)) / (4 * np.sqrt(2))
    
    for i in range(len(approx) // 2):
        a, d = approx[i], detail[i]
        reconst_point = h0 * a + h1 * d
        reconstructed_signal.append(reconst_point)
        reconst_point = h2 * a + h3 * d
        reconstructed_signal.append(reconst_point)
    
    return reconstructed_signal


#######################################################################################
## Synthesis db6
## Description: The synthesis filter bank is the inverse of the analysis filter bank. It is used to reconstruct a signal from approximation and detail coefficients.
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
#######################################################################################

def inverse_db6(approx, detail):
    # Your db6 (D12) filter coefficients
    h = [
        -0.001077301085308,
        0.0047772575109455,
        0.0005538422011614,
        -0.031582039318486,
        0.027522865530305,
        0.097501605587322,
        -0.129766867567262,
        -0.226264693965440,
        0.315250351709198,
        0.751133908021095,
        0.494623890398453,
        0.111540743350109
    ]
    
    # Reverse the coefficients to get synthesis filters (bit reveral and alternating signs) to satisfy QMF condition for perfect reconstruction (PR)
    # Define Low-Low Filter
    h_inv = h[::-1]
    # Define High-Low Filter 
    g_inv = [-1 * coeff if idx % 2 else coeff for idx, coeff in enumerate(h_inv)]

    N = len(h_inv)
    
    reconstructed_signal = [0.0] * (2 * len(approx))  # Initialize with zeros

    # Loop through the approximation and detail coefficients
    for i in range(len(approx)):
        for k in range(N):
            index1 = (2 * i + k) % len(reconstructed_signal)
            index2 = (2 * i - k) % len(reconstructed_signal)
            
            reconstructed_signal[index1] += approx[i] * h_inv[k] + detail[i] * g_inv[k]
            reconstructed_signal[index2] += approx[i] * h_inv[k] - detail[i] * g_inv[k]

    return reconstructed_signal

#############################################################################
## Inverse Symlet 5 Wavelet Transform
## Description: The Symlet wavelets are a family of orthogonal wavelets that are defined from the scaling function of the Daubechies wavelets. The Symlet wavelets are characterized by a maximal number of vanishing moments for some given support. In simpler terms, this means that the wavelets are designed to represent polynomial data of a given order efficiently.
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html
#############################################################################

def inv_sym5(approx, detail):
    h_inv = [
        0.019538882735287,
        -0.021101834024759,
        -0.175328089908450,
        0.016602105764522,
        0.633978963458212,
        0.723407690402421,
        0.199397533977394,
        -0.039134249302383,
        0.029519490925774,
        0.027333068345078
    ]
    g_inv = [h_inv[i] * (-1)**(i+1) for i in range(len(h_inv))]
    
    # Initialization
    reconstructed_signal = [0] * (len(approx) * 2)
    
    # Generate original signal
    for i in range(len(approx)):
        for k in range(len(h_inv)):
            index = 2 * i + k - len(h_inv) // 2 + 1
            if 0 <= index < len(reconstructed_signal):
                reconstructed_signal[index] += approx[i] * h_inv[k] + detail[i] * g_inv[k]
                
    return reconstructed_signal

def inv_sym8(approx, detail):
    # Sym8 inverse filter coefficients
    h_inv = [
        0.001889950332900,
        -0.000302920514551,
        -0.014952258336792,
        0.003808752013890,
        0.049137179673476,
        -0.027219029917056,
        -0.051945838107709,
        0.364441894835331,
        0.777185751700523,
        0.481359651258372,
        -0.061273359067908,
        -0.143294238350809,
        0.007607487325284,
        0.031695087811492,
        -0.000542132331635,
        -0.003382415951359
    ]
    
    g_inv = [h_inv[i] * (-1)**(i+1) for i in range(len(h_inv))]
    
    # Initialization
    reconstructed_signal = [0] * (len(approx) * 2)
    
    # Generate original signal
    for i in range(len(approx)):
        for k in range(len(h_inv)):
            index = 2 * i + k - len(h_inv) // 2 + 1
            if 0 <= index < len(reconstructed_signal):
                reconstructed_signal[index] += approx[i] * h_inv[k] + detail[i] * g_inv[k]
                
    return reconstructed_signal

#############################################################################
## Inverse Coiflet 5 Wavelet Transform
## Description: The Coiflet wavelets are a family of orthogonal wavelets that are defined from the scaling function of the Daubechies wavelets. The Coiflet wavelets are characterized by a maximal number of vanishing moments for some given support. In simpler terms, this means that the wavelets are designed to represent polynomial data of a given order efficiently.
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html
#############################################################################


def inv_coif5(approx, detail):
    # Coif5 inverse filter coefficients
    h_inv = [
        0.016387336463522,
        -0.041464936781959,
        -0.067372554721963,
        0.386110066821162,
        0.812723635445542,
        0.417005184421393,
        -0.076488599078311,
        -0.059434418646456,
        0.023680171946334,
        0.005611434819394,
        -0.001823208870703,
        -0.000720549445364
    ]
    
    g_inv = [h_inv[i] * (-1)**(i+1) for i in range(len(h_inv))]
    
    # Initialization
    reconstructed_signal = [0] * (len(approx) * 2)
    
    # Generate original signal
    for i in range(len(approx)):
        for k in range(len(h_inv)):
            index = 2 * i + k - len(h_inv) // 2 + 1
            if 0 <= index < len(reconstructed_signal):
                reconstructed_signal[index] += approx[i] * h_inv[k] + detail[i] * g_inv[k]
                
    return reconstructed_signal



# Synthesis Implementation
def synthesis(coeffs, wavelet_name):
    if wavelet_name == "haar":
        return idwt_multilevel(coeffs, inverse_haar)
    elif wavelet_name == "db1":
        return idwt_multilevel(coeffs, inverse_db1)
    elif wavelet_name == "db6":
        return idwt_multilevel(coeffs, inverse_db6)
    elif wavelet_name == "sym5":
        return idwt_multilevel(coeffs, inv_sym5)
    elif wavelet_name == "sym8":
        return idwt_multilevel(coeffs, inv_sym8)
    elif wavelet_name == "coif5":
        return idwt_multilevel(coeffs, inv_coif5)
    else:
        raise ValueError(f"Unknown wavelet type: {wavelet_name}")
def generate_time_vector(original_t, target_length):
    interpolator = interp1d(np.linspace(0, 1, len(original_t)), original_t)
    new_t = interpolator(np.linspace(0, 1, target_length))
    return new_t




######################################################################
## UI Block
## Description: The following code snippet demonstrates the use of the DWT and CWT on a simple sine wave.
######################################################################



# Select wavelet from user input
def select_wavelet():
    print("Please select the DWT type:")
    print("1: Haar")
    print("2: db1 (Daubechies 4)")
    print("3: db6 (Daubechies 12)")
    print("4: Sym5")
    print("5: Sym8")
    print("6: Coif5")
    choice = input("Enter the number corresponding to your choice: ")
    return choice



# Define a thresholding function (you can use different thresholding methods)
def threshold(coefficients, threshold_value, threshold_type='hard'):
    if threshold_type == 'hard':
        # Hard thresholding
        
        return np.where(np.abs(coefficients) < threshold_value, 0, coefficients)
    elif threshold_type == 'soft':
        # Soft thresholding
        
        return np.sign(coefficients) * np.maximum(0, np.abs(coefficients) - threshold_value)
    else:
        
        raise ValueError("Invalid threshold_type")

# Define a denoising function that applies thresholding to the wavelet coefficients
def denoise_coeff(coeffs, start_index, end_index, threshold_value, threshold_type='hard'):
    denoised_coeffs = list(coeffs)  # Clone the coefficients
    approx, detail = denoised_coeffs[start_index]
    
    # Apply thresholding only on the detail coefficients
    denoised_detail = threshold(detail, threshold_value, threshold_type)
    denoised_coeffs[start_index] = (approx, denoised_detail)
    
    for idx in range(start_index + 1, end_index):
        approx, detail = denoised_coeffs[idx]
        denoised_detail = threshold(detail, threshold_value, threshold_type)
        denoised_coeffs[idx] = (approx, denoised_detail)
        
    return denoised_coeffs

# Define a function to parallelize the denoising process for wavelet coefficients
def parallel_denoise_coeff(coeffs, num_processes, threshold_value, threshold_type='hard'):
    pool = mp.Pool(num_processes)
    chunk_size = len(coeffs) // num_processes
    
    # Create a list of arguments for the denoise_coeff function
    args_list = [(coeffs, i, min(i + chunk_size, len(coeffs)), threshold_value, threshold_type) for i in range(0, len(coeffs), chunk_size)]
    
    # Apply denoising to coefficients in parallel
    denoised_chunks = pool.starmap(denoise_coeff, args_list)
    pool.close()
    pool.join()
    
    # Merge the denoised coefficients from all chunks
    denoised_coeffs = [denoised_chunks[i][j] for i in range(num_processes) for j in range(len(denoised_chunks[i]))]
    
    return denoised_coeffs

def apply_wavelet(signal, wavelet_choice):
    approx = []
    detail = []
    if wavelet_choice == '1':
        approx, detail = haar(signal)  # Haar wavelet
    elif wavelet_choice == '2':
        approx, detail = db1(signal)  # db1 (Daubechies 4) wavelet
    elif wavelet_choice == '3':
        approx, detail = db6(signal)  # db6 (Daubechies 12) wavelet
    elif wavelet_choice == '4':
        approx, detail = sym5(signal)  # sym5 (Symlet 5) wavelet
    elif wavelet_choice == '5':
        approx, detail = sym8(signal)  # sym8 (Symlet 8) wavelet
    elif wavelet_choice == '6':
        approx, detail = coif5(signal)  # coif5 (Coiflet 5) wavelet
    else:
        print("Invalid choice. Using Haar as default.")
        approx, detail = haar(signal)  # Haar wavelet

    return approx, detail
######################################################################
## Implementation
## Description: The following code snippet demonstrates the use of the DWT and CWT on a simple sine wave.
######################################################################

if __name__ == '__main__':

    wavelet_functions = {
        '1': haar,
        '2': db1,
        '3': db6,
        '4': sym5,
        '5': sym8,
        '6': coif5,
        # ... you can easily add more choices in the future
    }
    # Select wavelet from user input
    wavelet_choice = select_wavelet()
    selected_wavelet_func = wavelet_functions.get(wavelet_choice, haar)  # Default to haar if wavelet_choice is not recognized
    # Creating a dictionary to map the wavelet choice to the wavelet function
    wavelet_map = {
    '1': 'haar',
    '2': 'db1',
    '3': 'db6',
    '4': 'sym5',
    '5': 'sym8',
    '6': 'coif5'
}
    
    duration = 1
    
    
    # Frequencies for A minor chord
    f_A = 440  # Frequency for A4
    f_C = 523.25  # Frequency for C5
    f_E = 659.25  # Frequency for E5
    fs = 10 * f_E  # Sampling frequency
    N_pwr=0.1 # Noise power
    # Create the A minor chord signal

    # Define the time vector
    t = np.linspace(0, duration, int(fs*duration))

    signal = np.sin(2 * np.pi * f_A * t) + np.sin(2 * np.pi * f_C * t) + np.sin(2 * np.pi * f_E * t)
    N_signal = awgn(signal, N_pwr)

    # Specify denoising parameters
    threshold_value = 0.9  # Adjust this threshold value as needed
    threshold_type = 'soft'  # Choose 'soft' or 'hard' thresholding
    num_processes =  3 # Define the number of parallel processes

   
    levels = 9  # Number of levels in the DWT  
    coeffs = dwt_multilevel(N_signal, selected_wavelet_func, levels)
    denoised_coeffs = parallel_denoise_coeff(coeffs, num_processes, threshold_value, threshold_type)
    # After Decomposing the signal, apply the denoising function
    
    
    
    # Initialize the original time vector t with the length of the signal
    tvec = [t]

    # Generate the time vectors at each level of the DWT
    for approx, detail in coeffs:
        target_length = len(approx)  # or len(detail), they should be the same
        new_t = generate_time_vector(t, target_length)
        tvec.append(new_t)


    ##########################################################
    ## Perform the DWT
    ##########################################################
    approx, detail = apply_wavelet(N_signal, wavelet_choice)
    
    ##########################################################
    ## Perform the IDWT
    ##########################################################
    reconstructed_signal = synthesis(denoised_coeffs, wavelet_map[wavelet_choice])



    ##########################################################
    # Plotting for DWT
    ##########################################################
    # Generate the time vectors at each level of the DWT
    for i in range(1, levels + 1):
        tvec.append(generate_time_vector(t, i))

    # Now, you can use these time_vectors while plotting
    plt.figure(figsize=(12, 8))

    # Plot the original signal
    plt.subplot(2 * levels + 1, 1, 1)
    plt.title("Original Signal")
    plt.plot(tvec[0], N_signal)

    # Plot the coefficients
    for i, (approx, detail) in enumerate(coeffs):
        stretched_t = tvec[i + 1]  # Using the appropriate time vector

        plt.subplot(2 * levels + 1, 1, 2 * i + 2)
        plt.title(f"Level {i+1} Approximation Coefficients")
        plt.plot(stretched_t, approx)

        plt.subplot(2 * levels + 1, 1, 2 * i + 3)
        plt.title(f"Level {i+1} Detail Coefficients")
        plt.plot(stretched_t, detail)

    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(14, 8))

    # Plot original signal
    plt.subplot(3, 1, 1)
    plt.title('Original Signal')
    plt.plot(t, N_signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Use the last time_vector for the last-level approximation and detail coefficients
    last_tvec = tvec[-1]
    t_approx = generate_time_vector(original_t=t, target_length=len(approx))
    t_detail = generate_time_vector(original_t=t, target_length=len(detail))
    t_recon = generate_time_vector(original_t=t, target_length=len(reconstructed_signal))
    # Plot Approximation Coefficients
    plt.subplot(3, 1, 2)
    plt.title('Noisy Approximation Coefficients')
    plt.plot(t_approx, approx)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')



    # Plot Detail Coefficients
    plt.subplot(3, 1, 3)
    plt.title('NoisyDetail Coefficients')
    plt.plot(t_detail, detail)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

    # Plot the original signal
    plt.figure(figsize=(10, 5))
    plt.plot(t, signal)
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

    # Plot the reconstructed signal
    plt.figure(figsize=(10, 5))
    plt.plot( t_recon, reconstructed_signal)
    plt.title('Denoised Reconstructed Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()



