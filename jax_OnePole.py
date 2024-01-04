import dawdreamer as daw

from functools import partial
import itertools
from pathlib import Path
import os
import time

import jax
default_device = 'gpu'
jax.config.update('jax_platform_name', default_device)
import jax.numpy as jnp
from jax import random, vmap, jit, grad, value_and_grad, lax

from flax import linen as nn
from flax.training import train_state # Useful dataclass to keep train state
from flax.core.frozen_dict import unfreeze 
import optax
from dawdreamer.faust import createLibContext, destroyLibContext, FaustContext
from dawdreamer.faust.box import *

from tqdm.notebook import tqdm
from scipy.io import wavfile

import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from IPython.display import HTML
from IPython.display import Audio
import IPython.display as ipd

SAMPLE_RATE = 44100

from dawutils import *
import dawutils as du
from jax import random
import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++ DAW MODEL ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Function to show audio in the notebook
def show_audio(data, autoplay=False):
    if data.size == 0:
        print("Empty audio data, no audio to show.")
        return

    if abs(data).max() > 1.:
        data /= abs(data).max()

    ipd.display(Audio(data=data, rate=SAMPLE_RATE, normalize=False, autoplay=autoplay))


# Function to save audio to a file
def save_audio(data, filename):
    if abs(data).max() > 1.:
        data /= abs(data).max()
    sf.write(filename, data, SAMPLE_RATE)


def pole2cutoff(poles, fs):
    """
    Convert pole locations to real-valued cutoff frequencies for a digital filter.

    :param poles: The pole locations in the Z-domain (should be within the unit circle).
    :param fs: Sampling frequency.
    :return: Real-valued cutoff frequencies in Hz.
    """
    # Ensure that the poles are within the unit circle for stability
    if np.any(np.abs(poles) >= 1):
        raise ValueError("All poles must be inside the unit circle for stability.")

    # Calculate the angle of each pole and convert to cutoff frequency
    theta = np.angle(poles)
    f_cutoff = fs * theta / (2 * np.pi)

    # Ensure frequencies are within the Nyquist range
    f_cutoff = np.clip(f_cutoff, 0, fs/2)

    # Ensure the output is real-valued
    f_cutoff_real = np.real(f_cutoff)
    return f_cutoff_real

def make_sine(freq: float, T: int, sr=SAMPLE_RATE):
    """Return sine wave based on freq in Hz and duration T in samples"""
    return jnp.sin(jnp.pi*2.*freq*jnp.arange(T)/sr)

overall_start_time = time.time()
print(f"Start Time {overall_start_time} seconds.")

#+++++++++++++++++++++++++++++++++++++++++++ GROUND TRUTH MODEL ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
faust_code = """
declare filename "untitled.dsp";
declare name "untitled";
import("stdfaust.lib");
myOnePole(x) =  +~(*(-a1)); 

 
a1 = hslider("[0]a1[style:knob][scale:linear]",0.5, -0.93, 0.93, 0.0001);

process = _ : myOnePole(a1);
"""

with FaustContext():
    box = boxFromDSP(faust_code)

    assert box.inputs == 1
    assert box.outputs == 1

    # Now we convert it to C++ code and specify a class name MyDSP
    module_name = "OnePole"
    cpp_code = boxToSource(box, 'cpp', module_name)
    print(cpp_code)
    
with FaustContext():
    box = boxFromDSP(faust_code)

    assert box.inputs == 1
    assert box.outputs == 1

    # Now we convert it to C++ code and specify a class name MyDSP
    module_name = "OnePole"
    jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])
    print(jax_code)

def faust2jax(faust_code: str):
    """
    Convert faust code into a batched JAX model and a single-item inference function.

    Inputs:
    * faust_code: string of faust code.
    """
    
    module_name = "MyDSP"
    with FaustContext():
    
      box = boxFromDSP(faust_code)
      
      jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])
    
    custom_globals = {}

    exec(jax_code, custom_globals)  # security risk!

    MyDSP = custom_globals[module_name]
    # MyDSP is now a class definition which can be instantiated or subclassed.

    """
    In our vmap code, we specify `in_axes=(0, None)` to batch along `x` but not along `T`.
    In other words, the `x` argument will be a batch, but our `T` will remain a simple integer (not a batch of integers)
    We choose `variable_axes={'params': None, 'intermediates': 0}` in order to share parameters among the batch and continue to sow intermediate vars
    We choose `split_rngs={'params': False}` to use the same random number generator for each item in the batch.
    """
    MyDSP = nn.vmap(MyDSP, in_axes=(0, None), variable_axes={'params': None, 'intermediates': 0}, split_rngs={'params': False})

    # Now we can create a model that handles batches of input.
    model_batch = MyDSP(sample_rate=SAMPLE_RATE)

    # let's jit compile the model's `apply` method    
    jit_inference_fn = jax.jit(partial(model_batch.apply, mutable='intermediates'), static_argnums=[2])

    # We jitted the model's "apply" function, which is of the form `apply(params, x, T)`. 
    # T (the number of samples of the output) is a constant, so we specified static_argnums=[2].
    
    # We specify mutable='intermediates' to access intermediate variables, which are
    # human-interpretable.
    # Our architecture file normalized all of the parameters to be between -1 and 1.
    # During a forward pass, the parameters are remapped to their original ranges
    # and stored as intermediate variables via the `sow` method.

    return model_batch, jit_inference_fn

hidden_model, jit_hidden = faust2jax(faust_code)

seed = 42
key = random.PRNGKey(seed)
key, subkey = random.split(key)
noise1 = random.uniform(subkey, shape=(100,100))
key, subkey = random.split(key)
noise2 = random.uniform(subkey, shape=(100,100))
key, subkey = random.split(key)
noise3 = random.uniform(subkey, shape=(100,100))


# T is the number of audio samples of input and output
T = int(SAMPLE_RATE*1.0)  # 1 second of audio

batch_size = 8

# The middle dimension is the number of channels of input
input_shape = (batch_size, hidden_model.getNumInputs(), T)

key = random.PRNGKey(42)

key, subkey = random.split(key)
noises = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)

key, subkey = random.split(key)
params = hidden_model.init({'params': subkey}, noises, T)

print('normalized params: ', params)

audio, mod_vars = jit_hidden(params, noises, T)
print('un-normalized params: ', mod_vars['intermediates'])

print('input audio (LOUD!!):')
show_audio(np.array(noises[0]))

print('output audio (less loud):')
show_audio(np.array(audio[0]))

# Training the model
# Repeat the code much earlier, except create a model whose cutoff is 10,000 Hz.
init_pole = -0.01 #@param {type: 'number'}
faust_code2 = f"""

import("stdfaust.lib");
myOnePole(x) = + ~(x * (-a1)); 

a1 = hslider("a1[style:knob][scale:linear]", {init_pole}, -0.93, 0.93, 0.0001);

process = myOnePole(a1);

"""


print(faust_code2)
train_model, jit_train_inference = faust2jax(faust_code2)

batch_size = 2 #@param {type: 'integer'}
input_shape = (batch_size, train_model.getNumInputs(), T)

# Create some noises that will serve as our validation dataset
key, subkey = random.split(key)
noises = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)

key, subkey = random.split(key)
hidden_params = hidden_model.init({'params': subkey}, jnp.zeros_like(noises), T)['params']
key, subkey = random.split(key)
train_params = train_model.init({'params': subkey}, jnp.zeros_like(noises), T)['params']

print('hidden params:', hidden_params)
print('train params:', train_params)

# +++++++++++++++++++++++++++++++++++++++++++ TRAINING MODEL++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Adam optimizer parameters
learning_rate = 1e-3 #@param {type: 'number'} # You can adjust this value

eta = 0.3  # Scale of noise (Noise-SGD papers suggest using a sequence of training where eta = {1.0, 0.1, 0.01}. 
# Inceasing eta (noise power) at first (1.0) then decreasing down the set helps the model escape local minima.)
gamma = 0.9  # Similar to SGD momentum
seed = 42
# Create Train state
tx = optax.noisy_sgd(learning_rate, eta, gamma, seed = seed)
state = train_state.TrainState.create(apply_fn=train_model.apply, params=train_params, tx=tx)

# Initialize jax Just-In-Time (JIT) compiler
@jax.jit
def train_step(state, x, y):
    """Train for a single step."""
    
    def loss_fn(params):
        pred = train_model.apply({'params': params}, x, T)
        # L1 time-domain loss
        loss = (jnp.abs(pred - y)).mean()
        return loss, pred
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

losses = []
cutoffs = []

train_steps = 3000 #@param {type: 'integer'}
train_steps_per_eval = 100
pbar = tqdm(range(train_steps))
start_time = time.time()

for n in pbar:
    # Generate a batch of inputs using our hidden parameters (440 Hz cutoff)
    key, subkey = random.split(key)
    x = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)
    y, _ = jit_hidden({'params': hidden_params}, x, T)
    
    state, loss = train_step(state, x, y)
        
    if n % train_steps_per_eval == 0:
        # jit_train_inference accepts a batch of one item.
        audio, mod_vars = jit_train_inference({'params': state.params}, noises, T)
        print(list(mod_vars['intermediates'].keys()))
        a1 = np.array(mod_vars['intermediates']['dawdreamer/a1'])
        # the cutoff above is a batch of predicted cutoff values, so we'll take the mean
        cutoff = pole2cutoff(a1, SAMPLE_RATE)
        cutoff = cutoff.mean()
        losses.append(loss)
        cutoffs.append(cutoff)
        pbar.set_description(f"cutoffs: {cutoffs}")

print('Done!')

elapsed_time = time.time() - start_time
print(f"Training took {elapsed_time} seconds.")

plt.figure(figsize =(8, 4))

ax1 = plt.subplot(211)
ax1.plot(losses)
ax1.set_title("Evaluation Loss (L1 Time Domain)")
ax1.set_ylabel("Loss (Linear scale)")
plt.tick_params('x', labelbottom=False)

ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(losses)
ax2.set_ylabel("Loss (Log scale)")
ax2.set_xlabel("Evaluation steps")
ax2.set_yscale('log')

plt.show()

plt.figure(figsize =(8, 4))

ax1 = plt.subplot(211)
ax1.plot()
ax1.set_title("Cutoff Parameter")
ax1.set_ylabel("Hz (Linear scale)")
ax1.set_xlabel("Evaluation steps")
plt.tick_params('x', labelbottom=False)

ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(cutoffs)
ax2.set_ylabel("Hz (Log scale)")
ax2.set_xlabel("Evaluation steps")
ax2.set_yscale('log')

plt.show()
# Pick a single example
x = noises[0:1]
y = hidden_model.apply({'params': hidden_params}, x, T)

# Pick the first param to be the varying parameter.
# There happens to only be one parameter.
param_name = list(hidden_params.keys())[0]

@jax.jit
def loss_one_sample(params):
    pred, mod_vars = jit_train_inference({'params': params}, x, T)
    assert pred.shape == y.shape
    loss = jnp.abs(pred-y).mean()

    return loss, mod_vars['intermediates']

loss_many_samples = jax.vmap(loss_one_sample, in_axes=0, out_axes=0)

loss_landscape_batch = 500

batched_hidden_params = jax.tree_map(lambda x: jnp.tile(x, loss_landscape_batch), hidden_params)
batched_hidden_params = unfreeze(batched_hidden_params)
batched_hidden_params[param_name] = jnp.linspace(-1, 1, loss_landscape_batch)
landscape_losses, mod_vars = loss_many_samples(batched_hidden_params)

plt.figure(figsize=(6, 8))

ax1 = plt.subplot(2, 1, 1)
ax1.set_title("Loss Landscape")
ax1.plot(np.array(batched_hidden_params[param_name]), landscape_losses)
ax1.set_ylabel('Loss')
ax1.set_xlabel('Normalized parameter')

ax2 = plt.subplot(2, 1, 2)
ax2.plot(list(mod_vars.values())[0][0], landscape_losses)
ax2.set_ylabel('Loss')
ax2.set_xlabel(list(mod_vars.keys())[0])
ax2.scatter(cutoffs, losses, s=10, color='red', label="training")
ax2.legend()
plt.show()

# Constructing the Loss Landscape
x=noises[0:1]
y=hidden_model.apply({'params': hidden_params}, x, T)

# Pick the first param to be the varying parameter as an example
param_name = list(hidden_params.keys())[0]



@jax.jit
def loss_one_sample(params):
    pred, mod_vars = jit_train_inference({'params': params}, x, T)
    assert pred.shape == y.shape
    loss = jnp.abs(pred-y).mean()

    return loss, mod_vars['intermediates']

loss_many_samples = jax.vmap(loss_one_sample, in_axes=0, out_axes=0)

loss_landscape_batch = 500

batched_hidden_params = jax.tree_map(lambda x: jnp.tile(x, loss_landscape_batch), hidden_params)
batched_hidden_params = unfreeze(batched_hidden_params)
batched_hidden_params[param_name] = jnp.linspace(-1, 1, loss_landscape_batch)
landscape_losses, mod_vars = loss_many_samples(batched_hidden_params)

# Convert pole location to cutoff frequency

plt.figure(figsize=(6, 8))

ax1 = plt.subplot(2, 1, 1)
ax1.set_title("Loss Landscape")
ax1.plot(np.array(batched_hidden_params[param_name]), landscape_losses)
ax1.set_ylabel('Loss')
ax1.set_xlabel('Normalized parameter')

ax2 = plt.subplot(2, 1, 2)
ax2.plot(list(mod_vars.values())[0][0], landscape_losses)
ax2.set_ylabel('Loss')
ax2.set_xlabel(list(mod_vars.keys())[0])
ax2.scatter(cutoffs, losses, s=10, color='red', label="training")
ax2.legend()
plt.show()

overall_elapsed_time = time.time() - overall_start_time
print(f"The whole process took {overall_elapsed_time} seconds.")

""" We previously used adam optimizer to optimize the bell's cutoff frequency, which was 
constant in time. But what if it were automated? We can use JAX to infer the hidden automation"""

"""
TODO: Complete Automation of the OnePole Filter


"""

