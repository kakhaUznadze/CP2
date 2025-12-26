from src.nonlinear_solvers import solve_fixed_point, solve_newton_gs
import numpy as np

def backward_euler_step(u_current, t_current, dt, model, method='fixed_point', **kwargs):
    """
    Performs one time step of Implicit Backward Euler integration.
    u_{n+1} = u_n + dt * f(t_{n+1}, u_{n+1})
    """
    t_next = t_current + dt
    
    if method == 'fixed_point':
        u_next, metrics = solve_fixed_point(u_current, dt, t_next, model, **kwargs)
    elif method == 'newton_gs':
        u_next, metrics = solve_newton_gs(u_current, dt, t_next, model, **kwargs)
    else:
        raise ValueError(f"Unknown nonlinear solver method: {method}")
        
    return t_next, u_next, metrics

def simulate(model, t_span, u0, dt, method='fixed_point', **solver_kwargs):
    """
    Simulates the system over t_span = [t_start, t_end].
    
    Returns:
        times: array of time points
        states: array of states (N, 2)
        metrics_list: list of metric dicts for each step
    """
    t_start, t_end = t_span
    
    times = [t_start]
    states = [u0]
    metrics_list = []
    
    t = t_start
    u = u0.copy()
    
    # Simple fixed time stepping
    # Calculate number of steps to avoid theoretical float drift issues in simple loop
    n_steps = int(round((t_end - t_start) / dt))
    
    for _ in range(n_steps):
        t_next, u_next, metrics = backward_euler_step(u, t, dt, model, method, **solver_kwargs)
        
        times.append(t_next)
        states.append(u_next)
        metrics_list.append(metrics)
        
        # Update for next step
        t = t_next
        u = u_next
        
    return np.array(times), np.array(states), metrics_list
