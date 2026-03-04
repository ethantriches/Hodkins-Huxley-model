# Standard HH parameters
from model import HodgkinHuxleyHelper, HodgkinHuxleyModel
from plot import Plotter
import numpy as np
import matplotlib.pyplot as plt

model = HodgkinHuxleyModel(g_Na=120, E_Na=115, g_K=36, E_K=-12, g_L=0.3, E_L=10.6)
helper = HodgkinHuxleyHelper(model)
plotter = Plotter()

t  = np.linspace(0, 100, 10000)
y0 = [0, 0.05, 0.6, 0.32]  # resting potential = -65 mV

solution = helper.solve_ODE(y0, t, I_ext=10)
current_solution = model.compute_currents(solution.T[0], solution.T[1], solution.T[2], solution.T[3])

V, m, h, n = solution.T

plotter.plot_results(t, V, m, h, n, current_solution[0], current_solution[1], current_solution[2], 10 * np.ones_like(t), np.sum(current_solution, axis=0))