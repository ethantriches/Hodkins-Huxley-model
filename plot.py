from matplotlib import pyplot as plt


class Plotter:
    def __init__(self):
        pass
    
    def plot_results(self, t, V, m, h, n, I_n, I_k, I_l, I_ext, I_total):
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(t, V)
        plt.title('Membrane Potential (V) over Time') 
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        
        plt.subplot(3, 1, 2)
        plt.plot(t, m, label='m (activation)')
        plt.plot(t, h, label='h (inactivation)')
        plt.plot(t, n, label='n (activation)')  
        plt.title('Gating Variables over Time')
        plt.xlabel('Time (ms)')
        plt.ylabel('Gating Variable Value')
        
        plt.subplot(3, 1, 3)
        plt.plot(t, I_n, label='I_Na')
        plt.plot(t, I_k, label='I_K')
        plt.plot(t, I_l, label='I_L')
        plt.plot(t, I_ext, label='I_ext', linestyle='--')
        plt.plot(t, I_total, label='I_total', linestyle='-.')
        plt.title('Currents over Time')
        plt.xlabel('Time (ms)')
        plt.ylabel('Current (µA/cm²)')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        