import argparse
import time
import os
from src.model import CisternModel
from src.integrators import simulate
from src.utils import save_metrics_csv, plot_results, save_simulation_results, ensure_dir

def main():
    parser = argparse.ArgumentParser(description="Run Cistern Simulation with Fixed Point Iteration")
    parser.add_argument('--dt', type=float, default=0.1, help="Time step (s)")
    parser.add_argument('--t_end', type=float, default=300.0, help="End time (s)")
    parser.add_argument('--save_dir', type=str, default='outputs', help="Output directory")
    parser.add_argument('--relaxation', type=float, default=1.0, help="Relaxation factor (1.0 = none)")
    args = parser.parse_args()

    # Setup
    save_dir_figs = os.path.join(args.save_dir, 'figures')
    save_dir_tables = os.path.join(args.save_dir, 'tables')
    ensure_dir(save_dir_figs)
    ensure_dir(save_dir_tables)
    
    model = CisternModel() # Uses defaults
    t_span = (0.0, args.t_end)
    u0 = model.u0
    
    print(f"Running Fixed-Point Simulation...")
    print(f"dt={args.dt}, t_end={args.t_end}, relaxation={args.relaxation}")
    
    start_time = time.time()
    
    # Run Simulation
    times, states, metrics = simulate(
        model, 
        t_span, 
        u0, 
        args.dt, 
        method='fixed_point', 
        relaxation=args.relaxation
    )
    
    runtime = time.time() - start_time
    print(f"Done. Runtime: {runtime:.4f}s")
    
    # Save Results
    final_h = states[-1, 0]
    save_metrics_csv(
        os.path.join(save_dir_tables, 'metrics.csv'), 
        'FixedPoint', 
        args.dt, 
        metrics, 
        runtime, 
        final_h
    )
    
    plot_results(times, states, 'FixedPoint', save_dir_figs)
    save_simulation_results(args.save_dir, times, states, 'FixedPoint')
    
    print(f"Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()
