
### Method 1: Fixed-Point Iteration
This script runs the simulation using the Fixed-Point method.

**Basic run (uses defaults):**
```bash
python run_fixed_point.py
```

**Custom run (play with parameters):**
```bash
python run_fixed_point.py --dt 0.5 --t_end 600 --relaxation 0.8
```

### Method 2: Newton-Gauss-Seidel
This one uses the Newton-GS solver.

**Basic run:**
```bash
python run_newton_gs.py
```

**Custom run:**
```bash
python run_newton_gs.py --dt 0.1 --t_end 100
```

## What Outputs are Produced

- **`outputs/figures/`**: Contains plots of the simulation (Water Level vs Time). Great for visualizing what happened.
- **`outputs/tables/`**: Contains `metrics.csv`. This file logs performance stats like how long it took to run and the final water level.
