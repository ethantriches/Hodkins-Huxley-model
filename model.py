import numpy as np
from scipy.integrate import solve_ivp

class SodiumChannel:
    def __init__(self, g_Na, E_Na):
        self.g_Na = g_Na
        self.E_Na = E_Na
    
    def alpha_m(self, V):
        if np.abs(V - 25) < 1e-6:
            return 1.0  # limit as V approaches 25 mV
        return 0.1 * (25 - V) / (np.exp((25 - V) / 10) - 1)
    
    def beta_m(self, V):
        return 4 * np.exp(-V / 18)
    
    def alpha_h(self, V):
        return 0.07 * np.exp(-V / 20)
    
    def beta_h(self, V):
        if V == 30:
            return 1.0  # limit as V approaches 30 mV
        return 1 / (np.exp((30 - V) / 10) + 1)
    
    def current(self, V, m, h):
        return self.g_Na * m**3 * h * (V - self.E_Na)


class PotassiumChannel:
    def __init__(self, g_K, E_K):
        self.g_K = g_K
        self.E_K = E_K
    
    def alpha_n(self, V):
        if np.abs(V - 10) < 1e-6:
            return 0.1  # limit as V approaches 10 mV
        return 0.01 * (10 - V) / (np.exp((10 - V) / 10) - 1)
    
    def beta_n(self, V):
        return 0.125 * np.exp(-V / 80)
    
    def current(self, V, n):
        return self.g_K * n**4 * (V - self.E_K)


class LeakChannel:
    def __init__(self, g_L, E_L):
        self.g_L = g_L
        self.E_L = E_L
    
    def current(self, V):
        return self.g_L * (V - self.E_L)


class HodgkinHuxleyModel:
    def __init__(self, g_Na, E_Na, g_K, E_K, g_L, E_L, Cm=1.0):
        self.sodium_channel = SodiumChannel(g_Na, E_Na)
        self.potassium_channel = PotassiumChannel(g_K, E_K)
        self.leak_channel = LeakChannel(g_L, E_L)
        self.Cm = Cm
    
    def compute_currents(self, V, m, h, n):
        I_Na = self.sodium_channel.current(V, m, h)
        I_K  = self.potassium_channel.current(V, n)
        I_L  = self.leak_channel.current(V)
        return I_Na, I_K, I_L
    
    def dVdt(self, V, m, h, n, I_ext):
        I_Na, I_K, I_L = self.compute_currents(V, m, h, n)
        return (I_ext - I_Na - I_K - I_L) / self.Cm

    def dMdt(self, V, m):
        return self.sodium_channel.alpha_m(V) * (1 - m) - self.sodium_channel.beta_m(V) * m

    def dHdt(self, V, h):
        return self.sodium_channel.alpha_h(V) * (1 - h) - self.sodium_channel.beta_h(V) * h

    def dNdt(self, V, n):
        return self.potassium_channel.alpha_n(V) * (1 - n) - self.potassium_channel.beta_n(V) * n


class HodgkinHuxleyHelper:
    def __init__(self, model):
        self.model = model
    
    def compute_derivatives(self, t, y, I_ext):  # t first, y second
        V, m, h, n = y
        dVdt = self.model.dVdt(V, m, h, n, I_ext)
        dMdt = self.model.dMdt(V, m)
        dHdt = self.model.dHdt(V, h)
        dNdt = self.model.dNdt(V, n)
        return [dVdt, dMdt, dHdt, dNdt]

    def solve_ODE(self, initial_conditions, t, I_ext):
        sol = solve_ivp(
            fun=lambda t, y: self.compute_derivatives(t, y, I_ext=I_ext),  # t first, y second
            t_span=(t[0], t[-1]),
            y0=initial_conditions,
            t_eval=t,
            max_step=0.01
        )
        return sol.y.T