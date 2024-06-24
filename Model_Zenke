from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# Neuron parameters
v_rest = -60*mV
v_rest_inh = -80*mV
v_thresh = -70*mV
v_thresh_inh = -60*mV
v_inh = -80*mV
v_exc = 0*mV
v_el = -70*mV
tau_m = 20*ms
alpha = 0.3

# Synaptic parameters
tau_ampa = 5*ms
tau_nmda = 100*ms
tau_gaba = 10*ms
tau_a = 100*ms
delta_a = 0.1

# STP parameters
U = 0.2  # Utilization of synaptic efficacy
tau_f = 600*ms  # Facilitation time constant
tau_d = 200*ms  # Recovery from depression time constant

tau_plus = 20*ms
tau_minus = 20*ms
tau_slow = 100*ms
tau_iSTDP = 20*ms
tau_h = 10*ms
A = 0.001
B = 0.001
B_i = A
del_plas = 0.00002
eta_exc = 0.001
eta_inh = 0.001
beta = 0.05
w_p = 0.5
P = 20
tau_cons = 1200*second
maxweight = 5
gamma = 4

# Neuron equations
neuron_eq1 = '''
    w : 1
    dv/dt = ((v_rest - v) + g_exc*(v_exc - v) + (g_gaba + g_a)*(v_inh - v))/tau_m : volt
    dg_gaba/dt = -g_gaba/tau_gaba : 1
    dg_a/dt = -g_a/tau_a : 1
    g_exc = g_ampa*alpha + (1-alpha)*g_nmda : 1
    dg_ampa/dt = -g_ampa/tau_ampa : 1
    dg_nmda/dt = (-g_nmda + g_ampa)/tau_nmda : 1
    '''

thalamus_AD = NeuronGroup(20, neuron_eq1, threshold='v>v_thresh', reset='v = v_rest', method='euler', refractory=5*ms)
thalamus_AV = NeuronGroup(40, neuron_eq1, threshold='v>v_thresh+2*mV', reset='v = v_rest', method='euler', refractory=5*ms)
thalamus_AM = NeuronGroup(40, neuron_eq1, threshold='v>v_thresh+3*mV', reset='v = v_rest', method='euler', refractory=5*ms)
cortex_exc = NeuronGroup(450, neuron_eq1, threshold='v>v_thresh+4*mV', reset='v = v_rest', method='euler', refractory=5*ms)
cortex_inh = NeuronGroup(50, neuron_eq1, threshold='v>v_thresh_inh', reset='v = v_rest', method='euler', refractory=15*ms)
stim = PoissonGroup(100,  rates='10*Hz + 5*Hz*randn()')

stim_inh = PoissonGroup(20, rates='5*Hz + 1*Hz*randn()')

# thalamus.v = v_el*rand()
# cortex_exc.v = v_el*rand()
# cortex_inh.v = v_el*rand()
thalamus_AD.v = v_el + (rand(len(thalamus_AD)) - 0.5) * 10 * mV
thalamus_AV.v = v_el + (rand(len(thalamus_AV)) - 0.5) * 10 * mV
thalamus_AM.v = v_el + (rand(len(thalamus_AM)) - 0.5) * 10 * mV
cortex_exc.v = v_el + (rand(len(cortex_exc)) - 0.5) * 10 * mV
cortex_inh.v = v_el + (rand(len(cortex_inh)) - 0.5) * 10 * mV


# Synapse equations
synapse_eq_EE = '''
    alpha : 1
    dx/dt = (1 - x)/tau_d : 1 (event-driven)
    du/dt = (U - u)/tau_f : 1 (event-driven)
    dw_tilde/dt = (w - w_tilde - P * w_tilde * (w_p/2 - w_tilde) * (w_p - w_tilde)) / tau_cons : 1 (clock-driven)
    dz_j_plus/dt = -z_j_plus/tau_plus: 1 (event-driven)
    dz_i_minus/dt = -z_i_minus/tau_minus : 1 (event-driven)
    dz_i_slow/dt = -z_i_slow/tau_slow : 1 (event-driven)
    w_eff = w * u * x : 1
    '''

synapse_eq_EI = '''
    alpha : 1
    dx/dt = (1 - x)/tau_d : 1 (event-driven)
    du/dt = (U - u)/tau_f : 1 (event-driven)
    w_eff = w * u * x : 1
    '''

synapse_eq_IE = '''
    dz_j/dt = -z_j/tau_iSTDP: 1 (event-driven)
    dz_i/dt = -z_i/tau_iSTDP : 1 (event-driven)
    dH/dt = -H/tau_h : 1 (event-driven)
    '''
#  w_tilde += P * w_tilde * (w_p/2 - w_tilde) * (w_p - w_tilde)

# Create synapses
syn_stim_to_thalamus_AD = Synapses(stim, thalamus_AD, model=synapse_eq_EE,
             on_pre='''
             x *= (1 - u)
             u += U * (1 - u)
             g_gaba_post += w_eff
             z_j_plus += 1
             w = w + eta_exc*B_i*z_i_minus + del_plas
             ''',
             on_post='''
             g_a_post += delta_a
             z_i_minus += 1
             w = clip(w + eta_exc*A*z_j_plus*z_i_slow - beta*(w-w_tilde)*(z_i_minus)**3, 0, maxweight)
             z_i_slow += 1
             ''')

syn_stim_to_thalamus_AD.connect(p=0.4)
syn_stim_to_thalamus_AD.w = '0.6'

syn_stim_to_thalamus_AV = Synapses(stim, thalamus_AV, model=synapse_eq_EE,
             on_pre='''
             x *= (1 - u)
             u += U * (1 - u)
             g_gaba_post += w_eff
             z_j_plus += 1
             w = w + eta_exc*B_i*z_i_minus + del_plas
             ''',
             on_post='''
             g_a_post += delta_a
             z_i_minus += 1
             w = clip(w + eta_exc*A*z_j_plus*z_i_slow - beta*(w-w_tilde)*(z_i_minus)**3, 0, maxweight)
             z_i_slow += 1
             ''')

syn_stim_to_thalamus_AV.connect(p=0.4)
syn_stim_to_thalamus_AV.w = '0.4'

syn_stim_to_thalamus_AM = Synapses(stim, thalamus_AM, model=synapse_eq_EE,
             on_pre='''
             x *= (1 - u)
             u += U * (1 - u)
             g_gaba_post += w_eff
             z_j_plus += 1
             w = w + eta_exc*B_i*z_i_minus + del_plas
             ''',
             on_post='''
             g_a_post += delta_a
             z_i_minus += 1
             w = clip(w + eta_exc*A*z_j_plus*z_i_slow - beta*(w-w_tilde)*(z_i_minus)**3, 0, maxweight)
             z_i_slow += 1
             ''')

syn_stim_to_thalamus_AM.connect(p=0.4)
syn_stim_to_thalamus_AM.w = '0.2'

syn_stim_to_cortex_exc = Synapses(stim, cortex_exc, model=synapse_eq_EE,
             on_pre='''
             x *= (1 - u)
             u += U * (1 - u)
             g_gaba_post += w_eff
             z_j_plus += 1
             w = w + eta_exc*B_i*z_i_minus + del_plas
             ''',
             on_post='''
             g_a_post += delta_a
             z_i_minus += 1
             w = clip(w + eta_exc*A*z_j_plus*z_i_slow - beta*(w-w_tilde)*(z_i_minus)**3, 0, maxweight)
             z_i_slow += 1
             ''')

syn_stim_to_cortex_exc.connect(p=0.09)
syn_stim_to_cortex_exc.w = '0.008'

syn_stim_to_cortex_inh = Synapses(stim, cortex_inh, model=synapse_eq_EI,
            on_pre='''
             x *= (1 - u)
             u += U * (1 - u)
             ''',
             on_post='''
             g_a_post += delta_a
             ''')

syn_stim_to_cortex_inh.connect(p=0.09)
syn_stim_to_cortex_inh.w = '0.01'

syn_thalamus_AD_to_cortex_exc = Synapses(thalamus_AD, cortex_exc, model=synapse_eq_EE,
             on_pre='''
             x *= (1 - u)
             u += U * (1 - u)
             g_gaba_post += w_eff
             z_j_plus += 1
             w = w + eta_exc*B_i*z_i_minus + del_plas
             ''',
             on_post='''
             g_a_post += delta_a
             z_i_minus += 1
             w = clip(w + eta_exc*A*z_j_plus*z_i_slow - beta*(w-w_tilde)*(z_i_minus)**3, 0, maxweight)
             z_i_slow += 1
             ''')

syn_thalamus_AD_to_cortex_exc.connect(p=0.3)
syn_thalamus_AD_to_cortex_exc.w = '0.07'  # Adjust the base synaptic weight

syn_cortex_exc_to_thalamus_AD = Synapses(cortex_exc, thalamus_AD, model=synapse_eq_EE,
             on_pre='''
             x *= (1 - u)
             u += U * (1 - u)
             g_gaba_post += w_eff
             z_j_plus += 1
             w = w + eta_exc*B_i*z_i_minus + del_plas
             ''',
             on_post='''
             g_a_post += delta_a
             z_i_minus += 1
             w = clip(w + eta_exc*A*z_j_plus*z_i_slow - beta*(w-w_tilde)*(z_i_minus)**3, 0, maxweight)
             z_i_slow += 1

             ''')
syn_cortex_exc_to_thalamus_AD.connect(p=0.02)
syn_cortex_exc_to_thalamus_AD.w = '0.007*rand()'  # Adjust the base synaptic weight

syn_thalamus_AV_to_cortex_exc = Synapses(thalamus_AV, cortex_exc, model=synapse_eq_EE,
             on_pre='''
             x *= (1 - u)
             u += U * (1 - u)
             g_gaba_post += w_eff
             z_j_plus += 1
             w = w + eta_exc*B_i*z_i_minus + del_plas
             ''',
             on_post='''
             g_a_post += delta_a
             z_i_minus += 1
             w = clip(w + eta_exc*A*z_j_plus*z_i_slow - beta*(w-w_tilde)*(z_i_minus)**3, 0, maxweight)
             z_i_slow += 1
             ''')

syn_thalamus_AV_to_cortex_exc.connect(p=0.3)
syn_thalamus_AV_to_cortex_exc.w = '0.003'  # Adjust the base synaptic weight

syn_cortex_exc_to_thalamus_AV = Synapses(cortex_exc, thalamus_AV, model=synapse_eq_EE,
             on_pre='''
             x *= (1 - u)
             u += U * (1 - u)
             g_gaba_post += w_eff
             z_j_plus += 1
             w = w + eta_exc*B_i*z_i_minus + del_plas
             ''',
             on_post='''
             g_a_post += delta_a
             z_i_minus += 1
             w = clip(w + eta_exc*A*z_j_plus*z_i_slow - beta*(w-w_tilde)*(z_i_minus)**3, 0, maxweight)
             z_i_slow += 1

             ''')
syn_cortex_exc_to_thalamus_AV.connect(p=0.02)
syn_cortex_exc_to_thalamus_AV.w = '0.005*rand()'  # Adjust the base synaptic weight

syn_thalamus_AM_to_cortex_exc = Synapses(thalamus_AM, cortex_exc, model=synapse_eq_EE,
             on_pre='''
             x *= (1 - u)
             u += U * (1 - u)
             g_gaba_post += w_eff
             z_j_plus += 1
             w = w + eta_exc*B_i*z_i_minus + del_plas
             ''',
             on_post='''
             g_a_post += delta_a
             z_i_minus += 1
             w = clip(w + eta_exc*A*z_j_plus*z_i_slow - beta*(w-w_tilde)*(z_i_minus)**3, 0, maxweight)
             z_i_slow += 1
             ''')

syn_thalamus_AM_to_cortex_exc.connect(p=0.3)
syn_thalamus_AM_to_cortex_exc.w = '0.003'  # Adjust the base synaptic weight

syn_cortex_exc_to_thalamus_AM = Synapses(cortex_exc, thalamus_AM, model=synapse_eq_EE,
             on_pre='''
             x *= (1 - u)
             u += U * (1 - u)
             g_gaba_post += w_eff
             z_j_plus += 1
             w = w + eta_exc*B_i*z_i_minus + del_plas
             ''',
             on_post='''
             g_a_post += delta_a
             z_i_minus += 1
             w = clip(w + eta_exc*A*z_j_plus*z_i_slow - beta*(w-w_tilde)*(z_i_minus)**3, 0, maxweight)
             z_i_slow += 1

             ''')
syn_cortex_exc_to_thalamus_AM.connect(p=0.02)
syn_cortex_exc_to_thalamus_AM.w = '0.002*rand()'  # Adjust the base synaptic weight

syn_cortex_recurrent_exc = Synapses(cortex_exc, cortex_exc, model=synapse_eq_EE,
             on_pre='''
             x *= (1 - u)
             u += U * (1 - u)
             g_gaba_post += w_eff
             z_j_plus += 1
             w = w + eta_exc*B_i*z_i_minus + del_plas
             ''',
             on_post='''
             g_a_post += delta_a
             z_i_minus += 1
             z_i_slow += 1
             w = clip(w + eta_exc*A*z_j_plus*z_i_slow - beta*(w-w_tilde)*(z_i_minus)**3, 0, maxweight)
             ''')
syn_cortex_recurrent_exc.connect(p=0.1)
syn_cortex_recurrent_exc.w = '0.001*rand()'  # Adjust the base synaptic weight

syn_thalamus_AD_to_cortex_inh = Synapses(thalamus_AD, cortex_inh, model=synapse_eq_EI,
             on_pre='''
             x *= (1 - u)
             u += U * (1 - u)
             ''',
             on_post='''
             g_a_post += delta_a
             '''
)
syn_thalamus_AD_to_cortex_inh.connect(p=0.05)
syn_thalamus_AD_to_cortex_inh.w = '0.007'

syn_thalamus_AV_to_cortex_inh = Synapses(thalamus_AV, cortex_inh, model=synapse_eq_EI,
             on_pre='''
             x *= (1 - u)
             u += U * (1 - u)
             ''',
             on_post='''
             g_a_post += delta_a
             '''
)
syn_thalamus_AV_to_cortex_inh.connect(p=0.05)
syn_thalamus_AV_to_cortex_inh.w = '0.006'

syn_thalamus_AM_to_cortex_inh = Synapses(thalamus_AM, cortex_inh, model=synapse_eq_EI,
             on_pre='''
             x *= (1 - u)
             u += U * (1 - u)
             ''',
             on_post='''
             g_a_post += delta_a
             '''
)
syn_thalamus_AM_to_cortex_inh.connect(p=0.05)
syn_thalamus_AM_to_cortex_inh.w = '0.005'

syn_cortex_recurrent_inh = Synapses(cortex_inh, cortex_inh)
syn_cortex_recurrent_inh.connect(p=0.001)
syn_cortex_recurrent_inh.w = '0.00001'

syn_stim_inh_to_thalamus_AD = Synapses(stim_inh, thalamus_AD, model=synapse_eq_IE,
             on_pre='''
             z_j += 1
             w = w + eta_inh*(H-gamma)*z_j
             ''',
             on_post='''
             z_i += 1
             w = clip(w + eta_inh*(H-gamma)*z_i, 0, maxweight)
             ''')

syn_stim_inh_to_thalamus_AD.connect(p=0.2)
syn_stim_inh_to_thalamus_AD.w = '0.0008'

syn_stim_inh_to_thalamus_AV = Synapses(stim_inh, thalamus_AV, model=synapse_eq_IE,
             on_pre='''
             z_j += 1
             w = w + eta_inh*(H-gamma)*z_j
             ''',
             on_post='''
             z_i += 1
             w = clip(w + eta_inh*(H-gamma)*z_i, 0, maxweight)
             ''')

syn_stim_inh_to_thalamus_AV.connect(p=0.2)
syn_stim_inh_to_thalamus_AV.w = '0.0006'

syn_stim_inh_to_thalamus_AM = Synapses(stim_inh, thalamus_AM, model=synapse_eq_IE,
             on_pre='''
             z_j += 1
             w = w + eta_inh*(H-gamma)*z_j
             ''',
             on_post='''
             z_i += 1
             w = clip(w + eta_inh*(H-gamma)*z_i, 0, maxweight)
             ''')

syn_stim_inh_to_thalamus_AM.connect(p=0.2)
syn_stim_inh_to_thalamus_AM.w = '0.0004'

syn_cortex_inh_to_cortex_exc = Synapses(cortex_inh, cortex_exc, model=synapse_eq_IE,
             on_pre='''
             z_j += 1
             w = w + eta_inh*(H-gamma)*z_j
             ''',
             on_post='''
             z_i += 1
             w = clip(w + eta_inh*(H-gamma)*z_i, 0, maxweight)
             ''')

syn_cortex_inh_to_cortex_exc.connect(p=0.02)
syn_cortex_inh_to_cortex_exc.w = '0.0002'

# Monitors
M11 = StateMonitor(thalamus_AD, 'v', record=True)
M12 = StateMonitor(thalamus_AV, 'v', record=True)
M13 = StateMonitor(thalamus_AM, 'v', record=True)
M2 = StateMonitor(cortex_exc, 'v', record=True)
M3 = StateMonitor(cortex_inh, 'v', record=True)

spike_mon_G11 = SpikeMonitor(thalamus_AD)
spike_mon_G12 = SpikeMonitor(thalamus_AV)
spike_mon_G13 = SpikeMonitor(thalamus_AM)
spike_mon_G2 = SpikeMonitor(cortex_exc)
spike_mon_G3 = SpikeMonitor(cortex_inh)

rate_mon_G11 = PopulationRateMonitor(thalamus_AD)
rate_mon_G12 = PopulationRateMonitor(thalamus_AV)
rate_mon_G13 = PopulationRateMonitor(thalamus_AM)
rate_mon_G2 = PopulationRateMonitor(cortex_exc)
rate_mon_G3 = PopulationRateMonitor(cortex_inh)

statemon_syn_thalamus_AD_to_cortex_exc = StateMonitor(syn_thalamus_AD_to_cortex_exc, 'w', record=True)

statemon_syn_cortex_exc_to_thalamus_AD = StateMonitor(syn_cortex_exc_to_thalamus_AD, 'w', record=True)
statemon_syn_cortex_recurrent_exc = StateMonitor(syn_cortex_recurrent_exc, 'w', record=True)
statemon_syn_thalamus_to_cortex_inh = StateMonitor(syn_thalamus_AD_to_cortex_inh, 'w', record=True)
statemon_syn_cortex_recurrent_inh = StateMonitor(syn_cortex_recurrent_inh, 'w', record=True)
statemon_syn_cortex_inh_to_cortex_exc = StateMonitor(syn_cortex_inh_to_cortex_exc, 'w', record=True)

# net = Network(thalamus_AD, thalamus_AV, thalamus_AM,syn_stim_inh_to_thalamus_AV,syn_stim_inh_to_thalamus_AM,
#               syn_thalamus_AV_to_cortex_inh, syn_thalamus_AM_to_cortex_inh,syn_stim_to_thalamus_AV, syn_stim_to_thalamus_AM,

#               cortex_exc,cortex_inh, stim,stim_inh, syn_stim_inh_to_thalamus_AD, syn_stim_to_thalamus_AD, syn_stim_to_cortex_exc,syn_stim_to_cortex_inh,
#               syn_thalamus_AD_to_cortex_inh, syn_thalamus_AD_to_cortex_exc, syn_cortex_exc_to_thalamus_AD, syn_cortex_recurrent_inh, syn_cortex_recurrent_exc,
#               syn_cortex_inh_to_cortex_exc, statemon_syn_thalamus_AD_to_cortex_exc, statemon_syn_cortex_exc_to_thalamus_AD, statemon_syn_cortex_recurrent_exc,
#               statemon_syn_thalamus_to_cortex_inh, statemon_syn_cortex_recurrent_inh, statemon_syn_cortex_inh_to_cortex_exc,
#               M11, M12, M13, M2, M3, spike_mon_G11,spike_mon_G12,spike_mon_G13, spike_mon_G2, spike_mon_G3,rate_mon_G11,rate_mon_G12,
#               rate_mon_G13, rate_mon_G2,rate_mon_G3)
# Create the network
net = Network()

# Add neuron groups to the network
net.add(thalamus_AD)
net.add(thalamus_AV)
net.add(thalamus_AM)
net.add(cortex_exc)
net.add(cortex_inh)
net.add(stim)
net.add(stim_inh)

# Add synapses to the network
net.add(syn_stim_to_thalamus_AD)
net.add(syn_stim_to_thalamus_AV)
net.add(syn_stim_to_thalamus_AM)
net.add(syn_stim_to_cortex_exc)
net.add(syn_stim_to_cortex_inh)
net.add(syn_thalamus_AD_to_cortex_exc)
net.add(syn_cortex_exc_to_thalamus_AD)
net.add(syn_thalamus_AV_to_cortex_exc)
net.add(syn_cortex_exc_to_thalamus_AV)
net.add(syn_thalamus_AM_to_cortex_exc)
net.add(syn_cortex_exc_to_thalamus_AM)
net.add(syn_cortex_recurrent_exc)
net.add(syn_thalamus_AD_to_cortex_inh)
net.add(syn_thalamus_AV_to_cortex_inh)
net.add(syn_thalamus_AM_to_cortex_inh)
net.add(syn_cortex_recurrent_inh)
net.add(syn_stim_inh_to_thalamus_AD)
net.add(syn_stim_inh_to_thalamus_AV)
net.add(syn_stim_inh_to_thalamus_AM)
net.add(syn_cortex_inh_to_cortex_exc)

# Add monitors to the network
net.add(M11)
net.add(M12)
net.add(M13)
net.add(M2)
net.add(M3)
net.add(spike_mon_G11)
net.add(spike_mon_G12)
net.add(spike_mon_G13)
net.add(spike_mon_G2)
net.add(spike_mon_G3)
net.add(rate_mon_G11)
net.add(rate_mon_G12)
net.add(rate_mon_G13)
net.add(rate_mon_G2)
net.add(rate_mon_G3)
net.add(statemon_syn_thalamus_AD_to_cortex_exc)
net.add(statemon_syn_cortex_exc_to_thalamus_AD)
net.add(statemon_syn_cortex_recurrent_exc)
net.add(statemon_syn_thalamus_to_cortex_inh)
net.add(statemon_syn_cortex_recurrent_inh)
net.add(statemon_syn_cortex_inh_to_cortex_exc)



# Run the simulation
net.run(10*second)

# Helper function to calculate ISI and CV of ISI
def compute_isi_cv(spikes):
    ISIs = np.diff(spikes)
    CVs = np.std(ISIs) / np.mean(ISIs) if len(ISIs) > 1 else np.nan
    return ISIs, CVs

# Calculate ISI and CV of ISI for thalamus_AD
isis_ad = []
cv_isis_ad = []
firing_rates_ad = []

for i in range(len(thalamus_AD)):
    spikes_i = spike_mon_G11.spike_trains()[i]
    if len(spikes_i) > 1:
        ISIs, CVs = compute_isi_cv(spikes_i)
        isis_ad.extend(ISIs)
        cv_isis_ad.append(CVs)
    firing_rates_ad.append(len(spikes_i) / (net.t / second))

# Calculate ISI and CV of ISI for thalamus_AV
isis_av = []
cv_isis_av = []
firing_rates_av = []

for i in range(len(thalamus_AV)):
    spikes_i = spike_mon_G12.spike_trains()[i]
    if len(spikes_i) > 1:
        ISIs, CVs = compute_isi_cv(spikes_i)
        isis_av.extend(ISIs)
        cv_isis_av.append(CVs)
    firing_rates_av.append(len(spikes_i) / (net.t / second))

# Calculate ISI and CV of ISI for thalamus_AM
isis_am = []
cv_isis_am = []
firing_rates_am = []

for i in range(len(thalamus_AM)):
    spikes_i = spike_mon_G13.spike_trains()[i]
    if len(spikes_i) > 1:
        ISIs, CVs = compute_isi_cv(spikes_i)
        isis_am.extend(ISIs)
        cv_isis_am.append(CVs)
    firing_rates_am.append(len(spikes_i) / (net.t / second))

# Calculate ISI and CV of ISI for cortex_exc
isis_exc = []
cv_isis_exc = []
firing_rates_exc = []

for i in range(len(cortex_exc)):
    spikes_i = spike_mon_G2.spike_trains()[i]
    if len(spikes_i) > 1:
        ISIs, CVs = compute_isi_cv(spikes_i)
        isis_exc.extend(ISIs)
        cv_isis_exc.append(CVs)
    firing_rates_exc.append(len(spikes_i) / (net.t / second))

# Calculate ISI and CV of ISI for cortex_inh
isis_inh = []
cv_isis_inh = []
firing_rates_inh = []

for i in range(len(cortex_inh)):
    spikes_i = spike_mon_G3.spike_trains()[i]
    if len(spikes_i) > 1:
        ISIs, CVs = compute_isi_cv(spikes_i)
        isis_inh.extend(ISIs)
        cv_isis_inh.append(CVs)
    firing_rates_inh.append(len(spikes_i) / (net.t / second))




# Plot the results
def plot_firing_rates():
    plt.figure(figsize=(15, 3))
    plot(rate_mon_G11.t/ms, rate_mon_G11.smooth_rate(window='flat', width=10*ms)/Hz, label='Thalamus AD')
    plot(rate_mon_G12.t/ms, rate_mon_G12.smooth_rate(window='flat', width=10*ms)/Hz, label='Thalamus AV')
    plot(rate_mon_G13.t/ms, rate_mon_G13.smooth_rate(window='flat', width=10*ms)/Hz, label='Thalamus AM')
    plot(rate_mon_G2.t/ms, rate_mon_G2.smooth_rate(window='flat', width=10*ms)/Hz, label='Cortex Excitatory')
    plot(rate_mon_G3.t/ms, rate_mon_G3.smooth_rate(window='flat', width=10*ms)/Hz, label='Cortex Inhibitory')
    xlabel('Time (ms)')
    ylabel('Firing rate (Hz)')
    legend()
    plt.show()

def plot_membrane_potentials():
    plt.figure(figsize=(15, 5))
    plot(M11.t/ms, M11.v[0]/mV, '.k', label='Neuron 0 (Thalamus AD)')
    plot(M12.t/ms, M12.v[0]/mV, '.r', label='Neuron 0 (Thalamus AV)')
    plot(M13.t/ms, M13.v[0]/mV, '.g', label='Neuron 0 (Thalamus AM)')
    plot(M2.t/ms, M2.v[0]/mV, '.b', label='Neuron 0 (Cortex Exc)')
    plot(M3.t/ms, M3.v[0]/mV, '.c', label='Neuron 0 (Cortex Inh)')
    xlabel('Time (ms)')
    ylabel('Membrane potential (mV)')
    legend()
    plt.show()

def plot_spike_raster():
    plt.figure(figsize=(10, 1))
    plot(spike_mon_G11.t/ms, spike_mon_G11.i, '.k', markersize=1, label='Thalamus AD spikes')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    legend()
    plt.show()

    plt.figure(figsize=(10, 1))
    plot(spike_mon_G12.t/ms, spike_mon_G12.i, '.r', markersize=1, label='Thalamus AV spikes')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    legend()
    plt.show()

    plt.figure(figsize=(10, 1))
    plot(spike_mon_G13.t/ms, spike_mon_G13.i, '.g', markersize=1, label='Thalamus AM spikes')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    legend()
    plt.show()

    figure(figsize=(10, 2))
    plot(spike_mon_G2.t/ms, spike_mon_G2.i, '.k', markersize=1, label='Cortex Exc spikes')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    legend()
    plt.show()

    plt.figure(figsize=(10, 2))
    plot(spike_mon_G3.t/ms, spike_mon_G3.i, '.k', markersize=1, label='Cortex Inh spikes')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    legend()
    plt.show()

def plot_isi_histograms(isis_g1, isis_g2, isis_g3, isis_cortex_exc, isis_cortex_inh):
    plt.figure(figsize=(3, 3))
    plt.hist(isis_g1, bins=20, alpha=0.6, label='Thalamus AD ISI', color='k')
    plt.xlabel('ISI (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    plt.figure(figsize=(3, 3))
    plt.hist(isis_g2, bins=20, alpha=0.6, label='Thalamus AV ISI', color='r')
    plt.xlabel('ISI (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    plt.figure(figsize=(3, 3))
    plt.hist(isis_g3, bins=20, alpha=0.6, label='Thalamus AM ISI', color='g')
    plt.xlabel('ISI (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    plt.figure(figsize=(3, 3))
    plt.hist(isis_cortex_exc, bins=20, alpha=0.6, label='Cortex Exc ISI', color='b')
    plt.xlabel('ISI (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    plt.figure(figsize=(3, 3))
    plt.hist(isis_cortex_inh, bins=20, alpha=0.6, label='Cortex Inh ISI', color='c')
    plt.xlabel('ISI (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_cv_isi_histograms(cv_isis_g1, cv_isis_g2, cv_isis_g3, cv_isis_cortex_exc, cv_isis_cortex_inh):
    plt.figure(figsize=(3, 3))
    plt.hist(cv_isis_g1, bins=20, alpha=0.6, label='Thalamus AD CV ISI', color='k')
    plt.xlabel('CV ISI')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    plt.figure(figsize=(3, 3))
    plt.hist(cv_isis_g2, bins=20, alpha=0.6, label='Thalamus AV CV ISI', color='r')
    plt.xlabel('CV ISI')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    plt.figure(figsize=(3, 3))
    plt.hist(cv_isis_g3, bins=20, alpha=0.6, label='Thalamus AM CV ISI', color='g')
    plt.xlabel('CV ISI')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    plt.figure(figsize=(3, 3))
    plt.hist(cv_isis_cortex_exc, bins=20, alpha=0.6, label='Cortex Exc CV ISI', color='b')
    plt.xlabel('CV ISI')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    plt.figure(figsize=(3, 3))
    plt.hist(cv_isis_cortex_inh, bins=20, alpha=0.6, label='Cortex Inh CV ISI', color='c')
    plt.xlabel('CV ISI')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
