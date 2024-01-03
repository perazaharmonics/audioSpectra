import jax
import jax.numpy as jnp
import cr.sparse as crs
import cr.sparse.data as crdata
import cr.sparse.dict as crdict
import cr.sparse.wt as crwt


# Replace the wavelet decomposition with CR-Wavelet
def cr_wavelet_decomposition(audio_signal, wavelet_name, level):
    # Create a CR-Wavelet dictionary
    dictionary = crwt.cr_wavelet_dictionary(audio_signal.size, wavelet_name, level)
    # Perform the CR-Wavelet decomposition
    coeffs = crs.analysis(dictionary, audio_signal)
    return coeffs

# Discrete Coiflet 5 wavelet decomposition with level 5
coeffs = cr_wavelet_decomposition(audio_signal, 'coif8', level=5)

approx_coeffs = coeffs[0]  # This is the approximation coefficients array
detail_coeffs = coeffs[1:]  # This is a list of detail coefficients arrays

# Reshape each array in detail_coeffs to be 2D
detail_coeffs_reshaped = [coeff.reshape(1, -1) for coeff in detail_coeffs]

# Convert to JAX arrays
audio_signal_jax = jnp.array(audio_signal)
approx_coeffs_jax = jnp.array(approx_coeffs)
detail_coeffs_jax = [jnp.array(coeff) for coeff in detail_coeffs_reshaped]

# Define CR-Sparse dictionary
dict_size = len(approx_coeffs_jax) + sum(coeff.size for coeff in detail_coeffs_jax)
dictionary = crdict.identity(dict_size)

# Perform the sparse coding using CR-Sparse
sparsity = 0.1  # Adjust the sparsity level as needed
code = crs.omp(dictionary, audio_signal_jax, sparsity)

# You can now use 'code' for further processing

# Calculate temporal features
def calculate_zcr(wavelet_coeff):
    zero_crossings = jnp.where(jnp.diff(jnp.sign(wavelet_coeff)))[0]
    return len(zero_crossings) / len(wavelet_coeff)

def calculate_rms(wavelet_coeff):
    return jnp.sqrt(jnp.mean(jnp.square(wavelet_coeff)))

zcr_subbands = [calculate_zcr(coeff) for coeff in detail_coeffs_jax]
rms_subbands = [calculate_rms(coeff) for coeff in detail_coeffs_jax]

# Convert temporal features to JAX arrays
zcr_subbands_jax = jnp.array(zcr_subbands)
rms_subbands_jax = jnp.array(rms_subbands)

# Concatenate temporal features
temporal_features_jax = jnp.concatenate([zcr_subbands_jax, rms_subbands_jax])

# Normalize the temporal features if needed
# temporal_features_jax_normalized = (temporal_features_jax - jnp.mean(temporal_features_jax)) / jnp.std(temporal_features_jax)

if temporal_features_jax.shape[0] == 0:
    raise ValueError("No temporal features found.")