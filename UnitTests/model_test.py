import pytest
from model import HodgkinHuxleyModel, HodgkinHuxleyHelper
import numpy as np

def test_compute_currents():
    model = HodgkinHuxleyModel(g_Na=120, E_Na=115, g_K=36, E_K=-12, g_L=0.3, E_L=10.6)
    V = 0
    m = 0.05
    h = 0.6
    n = 0.32
    
    I_Na, I_K, I_L = model.compute_currents(V, m, h, n)
    
    assert np.isclose(I_Na, -120 * (0.05**3) * 0.6 * (0 - 115))
    assert np.isclose(I_K, -36 * (0.32**4) * (0 + 12))
    assert np.isclose(I_L, -0.3 * (0 - 10.6))
    
def test_dVdt():
    model = HodgkinHuxleyModel(g_Na=120, E_Na=115, g_K=36, E_K=-12, g_L=0.3, E_L=10.6)
    V = 0
    m = 0.05
    h = 0.6
    n = 0.32
    I_ext = 10
    
    dVdt = model.dVdt(V, m, h, n, I_ext)
    
    I_Na, I_K, I_L = model.compute_currents(V, m, h, n)
    expected_dVdt = (I_ext - I_Na - I_K - I_L) / model.Cm
    
    assert np.isclose(dVdt, expected_dVdt)
    
def test_dMdt():
    model = HodgkinHuxleyModel(g_Na=120, E_Na=115, g_K=36, E_K=-12, g_L=0.3, E_L=10.6)
    V = 0
    m = 0.05
    
    dMdt = model.dMdt(V, m)
    
    alpha_m = model.sodium_channel.alpha_m(V)
    beta_m = model.sodium_channel.beta_m(V)
    expected_dMdt = alpha_m * (1 - m) - beta_m * m
    
    assert np.isclose(dMdt, expected_dMdt)

def test_dHdt():
    model = HodgkinHuxleyModel(g_Na=120, E_Na=115, g_K=36, E_K=-12, g_L=0.3, E_L=10.6)
    V = 0
    h = 0.6
    
    dHdt = model.dHdt(V, h)
    
    alpha_h = model.sodium_channel.alpha_h(V)
    beta_h = model.sodium_channel.beta_h(V)
    expected_dHdt = alpha_h * (1 - h) - beta_h * h
    
    assert np.isclose(dHdt, expected_dHdt)
    
def test_dNdt():
    model = HodgkinHuxleyModel(g_Na=120, E_Na=115, g_K=36, E_K=-12, g_L=0.3, E_L=10.6)
    V = 0
    n = 0.32
    
    dNdt = model.dNdt(V, n)
    
    alpha_n = model.potassium_channel.alpha_n(V)
    beta_n = model.potassium_channel.beta_n(V)
    expected_dNdt = alpha_n * (1 - n) - beta_n * n
    
    assert np.isclose(dNdt, expected_dNdt)
    
def test_solve_ODE():
    model = HodgkinHuxleyModel(g_Na=120, E_Na=115, g_K=36, E_K=-12, g_L=0.3, E_L=10.6)
    helper = HodgkinHuxleyHelper(model)
    
    t = np.linspace(0, 100, 10000)
    y0 = [0, 0.05, 0.6, 0.32]
    I_ext = 10
    
    solution = helper.solve_ODE(y0, t, I_ext)
    
    assert solution.shape == (len(t), 4)