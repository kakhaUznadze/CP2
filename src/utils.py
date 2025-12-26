import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import json

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_metrics_csv(path, method_name, dt, metrics_list, runtime, final_h):
    """
    Saves a summary row to a CSV file.
    Appends if exists.
    """
    file_exists = os.path.isfile(path)
    
    avg_iters = np.mean([m['iters'] for m in metrics_list])
    max_iters = np.max([m['iters'] for m in metrics_list])
    failures = np.sum([0 if m['converged'] else 1 for m in metrics_list])
    
    fieldnames = ['Method', 'dt', 'Steps', 'Avg_Iters', 'Max_Iters', 'Runtime_s', 'Final_h', 'Failures_Count']
    
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            'Method': method_name,
            'dt': dt,
            'Steps': len(metrics_list),
            'Avg_Iters': f"{avg_iters:.4f}",
            'Max_Iters': max_iters,
            'Runtime_s': f"{runtime:.4f}",
            'Final_h': f"{final_h:.4f}",
            'Failures_Count': failures
        })

def save_simulation_results(save_dir, times, states, method_name):
    """Saves the full time series to CSV for human readability."""
    ensure_dir(save_dir)
    file_path = os.path.join(save_dir, f"results_{method_name}.csv")
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'WaterHeight', 'ValveOpening'])
        for t, s in zip(times, states):
            writer.writerow([t, s[0], s[1]])

def plot_results(times, states, method_name, save_dir):
    """Generates standard plots."""
    ensure_dir(save_dir)
    
    h = states[:, 0]
    v = states[:, 1]
    
    # Plot 1: Height over time
    plt.figure(figsize=(8, 4))
    plt.plot(times, h, label='Water Height h(t)', color='blue')
    plt.axhline(y=0.25, color='r', linestyle='--', label='Target 0.25m', alpha=0.5) # Hardcoded ref
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.title(f'Cistern Filling - Height ({method_name})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'height_{method_name}.png'))
    plt.close()
    
    # Plot 2: Valve over time
    plt.figure(figsize=(8, 4))
    plt.plot(times, v, label='Valve Opening v(t)', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Opening Fraction')
    plt.title(f'Valve Dynamics ({method_name})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'valve_{method_name}.png'))
    plt.close()
    
    # Plot 3: Phase Portrait
    plt.figure(figsize=(6, 6))
    plt.plot(h, v, color='purple')
    plt.xlabel('Height (m)')
    plt.ylabel('Valve Opening')
    plt.title(f'Phase Portrait: v vs h ({method_name})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'phase_{method_name}.png'))
    plt.close()
