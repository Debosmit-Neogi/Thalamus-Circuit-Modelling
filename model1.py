from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# Parameters
tau = 25*ms
El1 = -50*mV + randint(-10,30)*mV
El2 = -45*mV + randint(-10,30)*mV
Vr1 = -75*mV
Vt1 = -58*mV + randint(-10,20)*mV
Vr2 = -75*mV
Vt2 = -45*mV + randint(-5,20)*mV
R = 20*Mohm
dt = 20*ms
Ee = 0*mV
Ei = -70*mV
tau_ampa = 5*ms
tau_nmda = 1*ms
tau_gaba = 10*ms
tau_adapt1 = 1*ms
tau_adapt2 = 2*ms
dg_adapt1 = 0.005
dg_adapt2 = 0.006
tau_post = 2*ms
eta = 0.002
eta_inh = 0.001
kappa = 0.001
maxweight = 1.0
U = 0.5
tau_d = 2*ms
tau_f = 1.5*ms
tau_ref = 10*ms
spiking_probability = 0.3
noise_mean = 0.5
noise_std = 0.1
gamma = 1.0  # Target local network activity level
tau_iSTDP = 20*ms  # Time constant for inhibitory STDP
tau_H = 50*ms  # Time constant for the global secreted factor

# Parameters for synaptic plasticity
eta_exc = 0.01  # Excitatory learning rate
A = 0.005  # LTP rate
beta = 0.001  # Heterosynaptic plasticity strength
delta = 0.001  # Transmitter-induced plasticity strength
tau_x = 20*ms  # Time constant for pre- and post-synaptic traces
tau_cons = 1000*ms  # Synaptic consolidation time constant
P = 0.1  # Magnitude of the double-well potential
w_P = 0.5  # Upper stable fixed point for consolidation
tau_hom = 500*ms  # Homeostatic regulation time constant
w_min_exc = 0.0
w_max_exc = 1.0



# Equations
eqs1 = '''
dv/dt = (El1 - v + R * (I + ge * (Ee - v) + gi * (Ei - v) - g_adapt1 * (v - Vr1) - g_adapt2 * (v - Vr1))) / tau : volt
dge/dt = -ge / tau_ampa : siemens
dgi/dt = -gi / tau_gaba : siemens
dg_adapt1/dt = -g_adapt1 / tau_adapt1 : siemens
dg_adapt2/dt = -g_adapt2 / tau_adapt2 : siemens
threshold : volt
I : amp
'''

eqs2_exc = '''
dv/dt = (El2 - v + R * (I + ge * (Ee - v) + gi * (Ei - v) - g_adapt1 * (v - Vr2) - g_adapt2 * (v - Vr2))) / tau : volt
dge/dt = -ge / tau_ampa : siemens
dgi/dt = -gi / tau_gaba : siemens
dg_adapt1/dt = -g_adapt1 / tau_adapt1 : siemens
dg_adapt2/dt = -g_adapt2 / tau_adapt2 : siemens
I : amp
'''

eqs2_inh = '''
dv/dt = (El2 - v + R * (I + ge * (Ee - v) + gi * (Ei - v) - g_adapt1 * (v - Vr2) - g_adapt2 * (v - Vr2))) / tau : volt
dge/dt = -ge / tau_nmda : siemens
dgi/dt = -gi / tau_gaba : siemens
dg_adapt1/dt = -g_adapt1 / tau_adapt1 : siemens
dg_adapt2/dt = -g_adapt2 / tau_adapt2 : siemens
I : amp
'''

# Neuron groups
N_thalamus = 100

N_thalamus_AD = 16
N_thalamus_AV = 42
N_thalamus_AM = 42

N_cortex = 500
N_inh_cortex = 100
N_exc_cortex = N_cortex - N_inh_cortex
N_stim = 100

stimulus = PoissonGroup(N_stim, rates=15*Hz) # Define stimulus neurons

thalamus_AD = NeuronGroup(N_thalamus, eqs1, threshold='v>Vt1', reset='v=Vr1', method='euler', refractory=tau_ref)
thalamus_AV = NeuronGroup(N_thalamus, eqs1, threshold='v>Vt1', reset='v=Vr1', method='euler', refractory=tau_ref)
thalamus_AM = NeuronGroup(N_thalamus, eqs1, threshold='v>Vt1', reset='v=Vr1', method='euler', refractory=tau_ref)

cortex_exc = NeuronGroup(N_exc_cortex, eqs2_exc, threshold='v>Vt2', reset='v=Vr2', method='euler', refractory=tau_ref)
cortex_inh = NeuronGroup(N_inh_cortex, eqs2_inh, threshold='v>Vt2', reset='v=Vr2', method='euler', refractory=tau_ref)

thalamus.v = 'El1 + rand() * (Vr1 - El1)'
cortex_exc.v = 'El2 + rand() * (Vr2 - El2)'
cortex_inh.v = 'El2 + rand() * (Vr2 - El2)'

# Synapses

# Define synapses from stimulus to thalamus_AD with plasticity
syn_stim_to_thalamus_AD = Synapses(stimulus, thalamus_AD, '''
                                  w : siemens
                                  dApre/dt = -Apre/tau_ampa : 1 (event-driven)
                                  dApost/dt = -Apost/tau_nmda : 1 (event-driven)
                                  u : 1  # Short-term plasticity variable
                                  x : 1  # Short-term plasticity variable
                                  ''',
                                  on_pre='''
                                  ge_post += w * u * x
                                  u = U + u * (1 - U)
                                  x *= exp((lastupdate - t)/tau_d)
                                  Apre += dg_adapt1
                                  w = clip(w + eta * Apost * siemens, 0*siemens, maxweight*siemens)
                                  ''',
                                  on_post='''
                                  Apost += dg_adapt1
                                  w = clip(w + eta * Apre * siemens, 0*siemens, maxweight*siemens)
                                  ''')

# Connect stimulus to thalamus neurons
syn_stim_to_thalamus_AD.connect(p=0.1)
syn_stim_to_thalamus_AD.w = 'rand() * 0.07 * siemens'

# Define synapses from stimulus to thalamus_AV with plasticity
syn_stim_to_thalamus_AV = Synapses(stimulus, thalamus_AV, '''
                                  w : siemens
                                  dApre/dt = -Apre/tau_ampa : 1 (event-driven)
                                  dApost/dt = -Apost/tau_nmda : 1 (event-driven)
                                  u : 1  # Short-term plasticity variable
                                  x : 1  # Short-term plasticity variable
                                  ''',
                                  on_pre='''
                                  ge_post += w * u * x
                                  u = U + u * (1 - U)
                                  x *= exp((lastupdate - t)/tau_d)
                                  Apre += dg_adapt1
                                  w = clip(w + eta * Apost * siemens, 0*siemens, maxweight*siemens)
                                  ''',
                                  on_post='''
                                  Apost += dg_adapt1
                                  w = clip(w + eta * Apre * siemens, 0*siemens, maxweight*siemens)
                                  ''')

# Connect stimulus to thalamus neurons
syn_stim_to_thalamus_AV.connect(p=0.1)
syn_stim_to_thalamus_AV.w = 'rand() * 0.05 * siemens'

# Define synapses from stimulus to thalamus_AM with plasticity
syn_stim_to_thalamus_AM = Synapses(stimulus, thalamus_AM, '''
                                  w : siemens
                                  dApre/dt = -Apre/tau_ampa : 1 (event-driven)
                                  dApost/dt = -Apost/tau_nmda : 1 (event-driven)
                                  u : 1  # Short-term plasticity variable
                                  x : 1  # Short-term plasticity variable
                                  ''',
                                  on_pre='''
                                  ge_post += w * u * x
                                  u = U + u * (1 - U)
                                  x *= exp((lastupdate - t)/tau_d)
                                  Apre += dg_adapt1
                                  w = clip(w + eta * Apost * siemens, 0*siemens, maxweight*siemens)
                                  ''',
                                  on_post='''
                                  Apost += dg_adapt1
                                  w = clip(w + eta * Apre * siemens, 0*siemens, maxweight*siemens)
                                  ''')

# Connect stimulus to thalamus neurons
syn_stim_to_thalamus_AM.connect(p=0.1)
syn_stim_to_thalamus_AM.w = 'rand() * 0.03 * siemens'

# Define synapses from thalamus to cortex (excitatory) with plasticity
syn_stim_to_cortex = Synapses(stimulus, cortex_exc, '''
                                  w : siemens
                                  dApre/dt = -Apre/tau_ampa : 1 (event-driven)
                                  dApost/dt = -Apost/tau_nmda : 1 (event-driven)
                                  u : 1  # Short-term plasticity variable
                                  x : 1  # Short-term plasticity variable
                                  ''',
                                  on_pre='''
                                  ge_post += w * u * x
                                  u = U + u * (1 - U)
                                  x *= exp((lastupdate - t)/tau_d)
                                  Apre += dg_adapt1
                                  w = clip(w + eta * Apost * siemens, 0*siemens, maxweight*siemens)
                                  ''',
                                  on_post='''
                                  Apost += dg_adapt1
                                  w = clip(w + eta * Apre * siemens, 0*siemens, maxweight*siemens)
                                  ''')
# Connect stimulus to thalamus neurons
syn_stim_to_cortex.connect(p=0.1)
syn_stim_to_cortex.w = 'rand() * 0.01 * siemens'


syn_thalamus_AD_to_cortex_exc = Synapses(thalamus_AD, cortex_exc, '''
                                  w : siemens
                                  dApre/dt = -Apre/tau_ampa : 1 (event-driven)
                                  dApost/dt = -Apost/tau_nmda : 1 (event-driven)
                                  u : 1  # Short-term plasticity variable
                                  x : 1  # Short-term plasticity variable
                                  ''',
                                  on_pre='''
                                  ge_post += w * u * x
                                  u = U + u * (1 - U)
                                  x *= exp((lastupdate - t)/tau_d)
                                  Apre += dg_adapt1
                                  w = clip(w + eta * Apost * siemens, 0*siemens, maxweight*siemens)
                                  ''',
                                  on_post='''
                                  Apost += dg_adapt1
                                  w = clip(w + eta * Apre * siemens, 0*siemens, maxweight*siemens)
                                  ''')


syn_thalamus_AD_to_cortex_exc.connect(p=0.4)
syn_thalamus_AD_to_cortex_exc.w = 'rand()*0.007*siemens'

syn_thalamus_AV_to_cortex_exc = Synapses(thalamus_AV, cortex_exc, '''
                                  w : siemens
                                  dApre/dt = -Apre/tau_ampa : 1 (event-driven)
                                  dApost/dt = -Apost/tau_nmda : 1 (event-driven)
                                  u : 1  # Short-term plasticity variable
                                  x : 1  # Short-term plasticity variable
                                  ''',
                                  on_pre='''
                                  ge_post += w * u * x
                                  u = U + u * (1 - U)
                                  x *= exp((lastupdate - t)/tau_d)
                                  Apre += dg_adapt1
                                  w = clip(w + eta * Apost * siemens, 0*siemens, maxweight*siemens)
                                  ''',
                                  on_post='''
                                  Apost += dg_adapt1
                                  w = clip(w + eta * Apre * siemens, 0*siemens, maxweight*siemens)
                                  ''')


syn_thalamus_AV_to_cortex_exc.connect(p=0.4)
syn_thalamus_AV_to_cortex_exc.w = 'rand()*0.005*siemens'

syn_thalamus_AM_to_cortex_exc = Synapses(thalamus_AM, cortex_exc, '''
                                  w : siemens
                                  dApre/dt = -Apre/tau_ampa : 1 (event-driven)
                                  dApost/dt = -Apost/tau_nmda : 1 (event-driven)
                                  u : 1  # Short-term plasticity variable
                                  x : 1  # Short-term plasticity variable
                                  ''',
                                  on_pre='''
                                  ge_post += w * u * x
                                  u = U + u * (1 - U)
                                  x *= exp((lastupdate - t)/tau_d)
                                  Apre += dg_adapt1
                                  w = clip(w + eta * Apost * siemens, 0*siemens, maxweight*siemens)
                                  ''',
                                  on_post='''
                                  Apost += dg_adapt1
                                  w = clip(w + eta * Apre * siemens, 0*siemens, maxweight*siemens)
                                  ''')


syn_thalamus_AM_to_cortex_exc.connect(p=0.4)
syn_thalamus_AM_to_cortex_exc.w = 'rand()*0.003*siemens'




syn_thalamus_AD_to_cortex_inh = Synapses(thalamus_AD, cortex_inh, '''
    w : siemens
    dz_pre_trace/dt = -z_pre_trace/tau_iSTDP : 1 (event-driven)
    dz_post_trace/dt = -z_post_trace/tau_iSTDP : 1 (event-driven)
    u : 1  # Short-term plasticity variable
    x : 1  # Short-term plasticity variable
    ''',
    on_pre='''
    ge_post += w * u * x
    u = U + u * (1 - U)
    x *= exp((lastupdate - t)/tau_d)
    z_pre_trace += 1
    w = clip(w + eta_inh * (z_post_trace + 1) * siemens, 0*siemens, maxweight*siemens)
    ''',
    on_post='''
    z_post_trace += 1
    w = clip(w + eta_inh * z_pre_trace * siemens, 0*siemens, maxweight*siemens)
    ''')


syn_thalamus_AD_to_cortex_inh.connect(p=0.4)
syn_thalamus_AD_to_cortex_inh.w = 'rand()*0.005*siemens'

syn_thalamus_AV_to_cortex_inh = Synapses(thalamus_AV, cortex_inh, '''
    w : siemens
    dz_pre_trace/dt = -z_pre_trace/tau_iSTDP : 1 (event-driven)
    dz_post_trace/dt = -z_post_trace/tau_iSTDP : 1 (event-driven)
    u : 1  # Short-term plasticity variable
    x : 1  # Short-term plasticity variable
    ''',
    on_pre='''
    ge_post += w * u * x
    u = U + u * (1 - U)
    x *= exp((lastupdate - t)/tau_d)
    z_pre_trace += 1
    w = clip(w + eta_inh * (z_post_trace + 1) * siemens, 0*siemens, maxweight*siemens)
    ''',
    on_post='''
    z_post_trace += 1
    w = clip(w + eta_inh * z_pre_trace * siemens, 0*siemens, maxweight*siemens)
    ''')


syn_thalamus_AV_to_cortex_inh.connect(p=0.4)
syn_thalamus_AV_to_cortex_inh.w = 'rand()*0.005*siemens'

syn_thalamus_AM_to_cortex_inh = Synapses(thalamus_AM, cortex_inh, '''
    w : siemens
    dz_pre_trace/dt = -z_pre_trace/tau_iSTDP : 1 (event-driven)
    dz_post_trace/dt = -z_post_trace/tau_iSTDP : 1 (event-driven)
    u : 1  # Short-term plasticity variable
    x : 1  # Short-term plasticity variable
    ''',
    on_pre='''
    ge_post += w * u * x
    u = U + u * (1 - U)
    x *= exp((lastupdate - t)/tau_d)
    z_pre_trace += 1
    w = clip(w + eta_inh * (z_post_trace + 1) * siemens, 0*siemens, maxweight*siemens)
    ''',
    on_post='''
    z_post_trace += 1
    w = clip(w + eta_inh * z_pre_trace * siemens, 0*siemens, maxweight*siemens)
    ''')


syn_thalamus_AM_to_cortex_inh.connect(p=0.4)
syn_thalamus_AM_to_cortex_inh.w = 'rand()*0.005*siemens'

syn_cortex_recurrent_exc = Synapses(cortex_exc, cortex_exc, '''
                                w : siemens
                                dApre/dt = -Apre/tau_ampa : 1 (event-driven)
                                dApost/dt = -Apost/tau_nmda : 1 (event-driven)
                                u : 1  # Short-term plasticity variable
                                x : 1  # Short-term plasticity variable
                                ''',
                                on_pre='''
                                ge_post += w * u * x
                                u = U + u * (1 - U)
                                x *= exp((lastupdate - t)/tau_d)
                                Apre += dg_adapt1
                                w = clip(w + eta * Apost * siemens, 0*siemens, maxweight*siemens)
                                ''',
                                on_post='''
                                Apost += dg_adapt1
                                w = clip(w + eta * Apre * siemens, 0*siemens, maxweight*siemens)
                                ''')

syn_cortex_recurrent_exc.add_attribute('w_ij_ref')
syn_cortex_recurrent_exc.add_attribute('B_i')
syn_cortex_recurrent_exc.add_attribute('C_i')
syn_cortex_recurrent_exc.connect(p=0.1, condition='i != j')
syn_cortex_recurrent_exc.w = 'rand()*0.001*siemens'
syn_cortex_recurrent_exc.w_ij_ref = 'rand()*0.001*siemens'
syn_cortex_recurrent_exc.B_i = 1.0
syn_cortex_recurrent_exc.C_i = 0.0

syn_cortex_recurrent_inh = Synapses(cortex_inh, cortex_exc, '''
    w : siemens
    dz_pre_trace/dt = -z_pre_trace/tau_iSTDP : 1 (event-driven)
    dz_post_trace/dt = -z_post_trace/tau_iSTDP : 1 (event-driven)
    u : 1  # Short-term plasticity variable
    x : 1  # Short-term plasticity variable
    ''',
    on_pre='''
    gi_post += w * u * x
    u = U + u * (1 - U)
    x *= exp((lastupdate - t)/tau_d)
    z_pre_trace += 1
    w = clip(w + eta_inh * (z_post_trace + 1) * siemens, 0*siemens, maxweight*siemens)
    ''',
    on_post='''
    z_post_trace += 1
    w = clip(w + eta_inh * z_pre_trace * siemens, 0*siemens, maxweight*siemens)
    ''')


syn_cortex_recurrent_inh.connect(p=0.1, condition='i != j')
syn_cortex_recurrent_inh.w = 'rand()*0.001*siemens'

syn_cortex_to_thalamus_AD_exc = Synapses(cortex_exc, thalamus_AD, '''
                                  w : siemens
                                  dApre/dt = -Apre/tau_ampa : 1 (event-driven)
                                  dApost/dt = -Apost/tau_nmda : 1 (event-driven)
                                  u : 1  # Short-term plasticity variable
                                  x : 1  # Short-term plasticity variable
                                  ''',
                                  on_pre='''
                                  ge_post += w * u * x
                                  u = U + u * (1 - U)
                                  x *= exp((lastupdate - t)/tau_d)
                                  Apre += dg_adapt1
                                  w = clip(w + eta * Apost * siemens, 0*siemens, maxweight*siemens)
                                  ''',
                                  on_post='''
                                  Apost += dg_adapt1
                                  w = clip(w + eta * Apre * siemens, 0*siemens, maxweight*siemens)
                                  ''')

syn_cortex_to_thalamus_AD_exc.add_attribute('w_ij_ref')
syn_cortex_to_thalamus_AD_exc.add_attribute('B')


syn_cortex_to_thalamus_AD_exc.connect(p=0.2)
syn_cortex_to_thalamus_AD_exc.w = 'rand()*0.004*siemens'


syn_cortex_to_thalamus_AV_exc = Synapses(cortex_exc, thalamus_AV, '''
                                  w : siemens
                                  dApre/dt = -Apre/tau_ampa : 1 (event-driven)
                                  dApost/dt = -Apost/tau_nmda : 1 (event-driven)
                                  u : 1  # Short-term plasticity variable
                                  x : 1  # Short-term plasticity variable
                                  ''',
                                  on_pre='''
                                  ge_post += w * u * x
                                  u = U + u * (1 - U)
                                  x *= exp((lastupdate - t)/tau_d)
                                  Apre += dg_adapt1
                                  w = clip(w + eta * Apost * siemens, 0*siemens, maxweight*siemens)
                                  ''',
                                  on_post='''
                                  Apost += dg_adapt1
                                  w = clip(w + eta * Apre * siemens, 0*siemens, maxweight*siemens)
                                  ''')

syn_cortex_to_thalamus_AV_exc.add_attribute('w_ij_ref')
syn_cortex_to_thalamus_AV_exc.add_attribute('B')


syn_cortex_to_thalamus_AV_exc.connect(p=0.2)
syn_cortex_to_thalamus_AV_exc.w = 'rand()*0.003*siemens'


syn_cortex_to_thalamus_AM_exc = Synapses(cortex_exc, thalamus_AM, '''
                                  w : siemens
                                  dApre/dt = -Apre/tau_ampa : 1 (event-driven)
                                  dApost/dt = -Apost/tau_nmda : 1 (event-driven)
                                  u : 1  # Short-term plasticity variable
                                  x : 1  # Short-term plasticity variable
                                  ''',
                                  on_pre='''
                                  ge_post += w * u * x
                                  u = U + u * (1 - U)
                                  x *= exp((lastupdate - t)/tau_d)
                                  Apre += dg_adapt1
                                  w = clip(w + eta * Apost * siemens, 0*siemens, maxweight*siemens)
                                  ''',
                                  on_post='''
                                  Apost += dg_adapt1
                                  w = clip(w + eta * Apre * siemens, 0*siemens, maxweight*siemens)
                                  ''')

syn_cortex_to_thalamus_AM_exc.add_attribute('w_ij_ref')
syn_cortex_to_thalamus_AM_exc.add_attribute('B')


syn_cortex_to_thalamus_AM_exc.connect(p=0.2)
syn_cortex_to_thalamus_AM_exc.w = 'rand()*0.002*siemens'

# syn_cortex_to_thalamus_inh = Synapses(cortex_inh, thalamus, '''
#     w : siemens
#     dz_pre_trace/dt = -z_pre_trace/tau_iSTDP : 1 (event-driven)
#     dz_post_trace/dt = -z_post_trace/tau_iSTDP : 1 (event-driven)
#     u : 1  # Short-term plasticity variable
#     x : 1  # Short-term plasticity variable
#     ''',
#     on_pre='''
#     gi_post += w * u * x
#     u = U + u * (1 - U)
#     x *= exp((lastupdate - t)/tau_d)
#     z_pre_trace += 1
#     w = clip(w + eta_inh * (z_post_trace + 1) * siemens, 0*siemens, maxweight*siemens)
#     ''',
#     on_post='''
#     z_post_trace += 1
#     w = clip(w + eta_inh * z_pre_trace * siemens, 0*siemens, maxweight*siemens)
#     ''')


# syn_cortex_to_thalamus_inh.connect(p=0.2)
# syn_cortex_to_thalamus_inh.w = 'rand()*0.002*siemens'

# Create a time array
duration = 500*ms
# time_array_in_seconds = np.arange(0, duration/second, dt/second)

# # External current inputs
# I_ext_thalamus = TimedArray(1*nA*(np.sin(10*np.pi*2*time_array_in_seconds)), dt=dt)
# I_ext_cortex = TimedArray(1*nA*(np.sin(1*np.pi*2*time_array_in_seconds)), dt=dt)

# Monitors
spike_mon_thalamus_AD = SpikeMonitor(thalamus_AD)
spike_mon_thalamus_AV = SpikeMonitor(thalamus_AV)
spike_mon_thalamus_AM = SpikeMonitor(thalamus_AM)
spike_mon_cortex_exc = SpikeMonitor(cortex_exc)
spike_mon_cortex_inh = SpikeMonitor(cortex_inh)
rate_mon_inh = PopulationRateMonitor(cortex_inh)

v_mon_thalamus_AD = StateMonitor(thalamus_AD, 'v', record=True)
v_mon_thalamus_AD = StateMonitor(thalamus_AV, 'v', record=True)
v_mon_thalamus_AD = StateMonitor(thalamus_AM, 'v', record=True)

v_mon_cortex_exc = StateMonitor(cortex_exc, 'v', record=True)
v_mon_cortex_inh = StateMonitor(cortex_inh, 'v', record=True)

w_mon_thalamus_AD_to_cortex_exc = StateMonitor(syn_thalamus_AD_to_cortex_exc, 'w', record=True)
w_mon_thalamus_AD_to_cortex_inh = StateMonitor(syn_thalamus_AD_to_cortex_inh, 'w', record=True)
w_mon_thalamus_AV_to_cortex_exc = StateMonitor(syn_thalamus_AV_to_cortex_exc, 'w', record=True)
w_mon_thalamus_AV_to_cortex_inh = StateMonitor(syn_thalamus_AV_to_cortex_inh, 'w', record=True)
w_mon_thalamus_AM_to_cortex_exc = StateMonitor(syn_thalamus_AM_to_cortex_exc, 'w', record=True)
w_mon_thalamus_AM_to_cortex_inh = StateMonitor(syn_thalamus_AM_to_cortex_inh, 'w', record=True)

# w_mon_thalamus_to_cortex_inh = StateMonitor(syn_thalamus_to_cortex_inh, 'w', record=True)
w_mon_cortex_recurrent_exc = StateMonitor(syn_cortex_recurrent_exc, 'w', record=True)
w_mon_cortex_recurrent_inh = StateMonitor(syn_cortex_recurrent_inh, 'w', record=True)
w_mon_cortex_to_thalamus_AD_exc = StateMonitor(syn_cortex_to_thalamus_AD_exc, 'w', record=True)
w_mon_cortex_to_thalamus_AV_exc = StateMonitor(syn_cortex_to_thalamus_AV_exc, 'w', record=True)
w_mon_cortex_to_thalamus_AM_exc = StateMonitor(syn_cortex_to_thalamus_AM_exc, 'w', record=True)

# w_mon_cortex_to_thalamus_inh = StateMonitor(syn_cortex_to_thalamus_inh, 'w', record=True)

# Global secreted factor H and inhibitory scaling factor G
# H = TimedArray(rate_mon_inh.smooth_rate(window='flat', width=50*ms), dt=defaultclock.dt)
# G = TimedArray((gamma / H), dt=defaultclock.dt)

# Network operation to update input
# @network_operation(dt=dt)
# def update_input():
#     thalamus.I = I_ext_thalamus(defaultclock.t)
#     cortex_exc.I = I_ext_cortex(defaultclock.t)
#     cortex_inh.I = I_ext_cortex(defaultclock.t)

# Network definition
net = Network(stimulus, thalamus, cortex_exc, cortex_inh, syn_stim_to_cortex, syn_stim_to_thalamus_AD, syn_stim_to_thalamus_AV, 
              syn_stim_to_thalamus_AM,
              syn_thalamus_AD_to_cortex_exc,syn_thalamus_AV_to_cortex_exc, syn_thalamus_AM_to_cortex_exc, syn_thalamus_AD_to_cortex_inh,
              syn_thalamus_AV_to_cortex_inh, syn_thalamus_AM_to_cortex_inh,
              syn_cortex_recurrent_exc, syn_cortex_recurrent_inh,
              syn_cortex_to_thalamus_AM_exc, syn_cortex_to_thalamus_AD_exc, syn_cortex_to_thalamus_AV_exc,
               spike_mon_cortex_exc, spike_mon_cortex_inh,
              rate_mon_inh, v_mon_thalamus_AD, v_mon_thalamus_AV, v_mon_thalamus_AM, v_mon_cortex_exc, v_mon_cortex_inh,
              w_mon_thalamus_AD_to_cortex_exc, w_mon_thalamus_AD_to_cortex_inh, w_mon_thalamus_AV_to_cortex_exc,
              w_mon_thalamus_AV_to_cortex_inh, w_mon_thalamus_AM_to_cortex_exc, w_mon_thalamus_AM_to_cortex_inh,
              w_mon_cortex_recurrent_exc, w_mon_cortex_recurrent_inh,
              w_mon_cortex_to_thalamus_AD_exc, w_mon_cortex_to_thalamus_AV_exc, w_mon_cortex_to_thalamus_AM_exc,
              spike_mon_thalamus_AD, spike_mon_thalamus_AV, spike_mon_thalamus_AM,rate_mon_inh
              )


# Run simulation
net.run(duration)

# Additional code for plotting and analysis
# Spike raster plot for thalamus 

plt.figure(figsize=(30, 20))
plt.subplot(3, 3, 1)
plt.plot(spike_mon_thalamus_AD.t/ms, spike_mon_thalamus_AD.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Thalamus AD Spikes')


plt.subplot(3, 3, 2)
plt.plot(spike_mon_thalamus_AV.t/ms, spike_mon_thalamus_AM.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Thalamus AM Spikes')


plt.subplot(3, 3, 3)
plt.plot(spike_mon_thalamus_AM.t/ms, spike_mon_thalamus_AM.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Thalamus AV Spikes')



# Spike raster plot for excitatory cortex
plt.subplot(3, 3, 4)
plt.plot(spike_mon_cortex_exc.t/ms, spike_mon_cortex_exc.i, '.b')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Excitatory Cortex Spikes')

# Spike raster plot for inhibitory cortex
plt.subplot(3, 3, 5)
plt.plot(spike_mon_cortex_inh.t/ms, spike_mon_cortex_inh.i, '.r')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Inhibitory Cortex Spikes')

avg_thalamus_AD_membrane_potential = 0
avg_thalamus_AV_membrane_potential = 0
avg_thalamus_AM_membrane_potential = 0
# avg_cortex_membrane_potential = 0
avg_exc_cortex_membrane_potential = 0
avg_inh_cortex_membrane_potential = 0

for i in range(len(v_mon_thalamus_AD.v)):
  avg_thalamus_AD_membrane_potential = avg_thalamus_AD_membrane_potential + v_mon_thalamus_AD.v[i]
avg_thalamus_AD_membrane_potential = avg_thalamus_AD_membrane_potential/len(v_mon_thalamus_AD.v)

for i in range(len(v_mon_thalamus_AV.v)):
  avg_thalamus_AV_membrane_potential = avg_thalamus_AV_membrane_potential + v_mon_thalamus_AV.v[i]
avg_thalamus_AV_membrane_potential = avg_thalamus_AV_membrane_potential/len(v_mon_thalamus_AV.v)

for i in range(len(v_mon_thalamus_AM.v)):
  avg_thalamus_AM_membrane_potential = avg_thalamus_AM_membrane_potential + v_mon_thalamus_AM.v[i]
avg_thalamus_AM_membrane_potential = avg_thalamus_AM_membrane_potential/len(v_mon_thalamus_AM.v)

for i in range(len(v_mon_cortex_exc.v)):
  avg_exc_cortex_membrane_potential = avg_exc_cortex_membrane_potential + v_mon_cortex_exc.v[i]
avg_exc_cortex_membrane_potential = avg_exc_cortex_membrane_potential/len(v_mon_cortex_exc.v)

for i in range(len(v_mon_cortex_inh.v)):
  avg_inh_cortex_membrane_potential = avg_inh_cortex_membrane_potential + v_mon_cortex_inh.v[i]
avg_inh_cortex_membrane_potential = avg_inh_cortex_membrane_potential/len(v_mon_cortex_inh.v)

# Membrane potential trace for thalamus neuron
plt.subplot(3, 3, 6)
plt.plot(v_mon_thalamus_AD.t/ms, avg_thalamus_AD_membrane_potential)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (V)')
plt.title('Thalamus AD Membrane Potential')

plt.subplot(3, 3, 7)
plt.plot(v_mon_thalamus_AV.t/ms, avg_thalamus_AV_membrane_potential)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (V)')
plt.title('Thalamus AV Membrane Potential')

plt.subplot(3, 3, 8)
plt.plot(v_mon_thalamus_AM.t/ms, avg_thalamus_AM_membrane_potential)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (V)')
plt.title('Thalamus AM Membrane Potential')

# Membrane potential trace for excitatory cortex neuron
plt.subplot(3, 3, 9)
# for i in range(len(v_mon_cortex_exc.v)):
plt.plot(v_mon_cortex_exc.t/ms, avg_exc_cortex_membrane_potential)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (V)')
plt.title('Excitatory Cortex Membrane Potential')

# Membrane potential trace for inhibitory cortex neuron
plt.subplot(3, 3, 10)
plt.plot(v_mon_cortex_inh.t/ms,avg_inh_cortex_membrane_potential)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (V)')
plt.title('Inhibitory Cortex Membrane Potential')

# # Synaptic weight evolution for thalamus to excitatory cortex
# plt.subplot(3, 3, 7)
# plt.plot(w_mon_thalamus_to_cortex_exc.t/ms, w_mon_thalamus_to_cortex_exc.w.T)
# plt.xlabel('Time (ms)')
# plt.ylabel('Synaptic weight (siemens)')
# plt.title('Thalamus to Excitatory Cortex Synaptic Weight')

# # Synaptic weight evolution for thalamus to inhibitory cortex
# plt.subplot(3, 3, 8)
# plt.plot(w_mon_thalamus_to_cortex_inh.t/ms, w_mon_thalamus_to_cortex_inh.w.T)
# plt.xlabel('Time (ms)')
# plt.ylabel('Synaptic weight (siemens)')
# plt.title('Thalamus to Inhibitory Cortex Synaptic Weight')

# # Synaptic weight evolution for excitatory cortex recurrent connections
# plt.subplot(3, 3, 9)
# plt.plot(w_mon_cortex_recurrent_exc.t/ms, w_mon_cortex_recurrent_exc.w.T)
# plt.xlabel('Time (ms)')
# plt.ylabel('Synaptic weight (siemens)')
# plt.title('Excitatory Cortex Recurrent Synaptic Weight')

plt.tight_layout()
plt.show()


