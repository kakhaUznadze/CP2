import numpy as np

def solve_fixed_point(u_old, dt, t_next, model, tol=1e-6, max_iter=50, relaxation=1.0):
    """
    Solves the implicit equation u = u_old + dt * f(t_next, u) using Fixed Point Iteration.
    
    Args:
        u_old: State at previous step
        dt: Time step
        t_next: Time at next step
        model: Physics model (has .f(t,u))
        tol: Convergence tolerance for norm(change)
        max_iter: Max iterations
        relaxation: Damping factor omega (0 < omega <= 1). u_new = (1-w)*u_prev + w*G(u_prev)
        
    Returns:
        u_next: Converged solution
        metrics: dict with 'iters', 'converged'
    """
    u_guess = u_old.copy() # Initial guess is previous step
    
    iters = 0
    converged = False
    
    for i in range(max_iter):
        iters += 1
        
        # G(u) = u_old + dt * f(t_next, u)
        g_val = u_old + dt * model.f(t_next, u_guess)
        
        # Update with relaxation: u_{k+1} = (1-w)u_k + w*G(u_k)
        u_new = (1.0 - relaxation) * u_guess + relaxation * g_val
        
        # Check convergence
        diff = np.linalg.norm(u_new - u_guess)
        if diff < tol:
            converged = True
            u_guess = u_new
            break
            
        u_guess = u_new
        
    return u_guess, {'iters': iters, 'converged': converged}


def gauss_seidel_2x2_solve(A, b, x_guess, tol=1e-8, max_iter=20):
    """
    Solves Ax = b for x using Gauss-Seidel for a 2x2 system.
    Manual implementation as requested.
    """
    x = x_guess.copy()
    
    for _ in range(max_iter):
        x_old_iter = x.copy()
        
        # Row 0: A[0,0]x[0] + A[0,1]x[1] = b[0]
        # x[0] = (b[0] - A[0,1]x[1]) / A[0,0]
        if abs(A[0,0]) > 1e-12:
            x[0] = (b[0] - A[0,1] * x[1]) / A[0,0]
            
        # Row 1: A[1,0]x[0] + A[1,1]x[1] = b[1]
        # x[1] = (b[1] - A[1,0]x[0]) / A[1,1]  <-- uses NEW x[0]
        if abs(A[1,1]) > 1e-12:
            x[1] = (b[1] - A[1,0] * x[0]) / A[1,1]
            
        if np.linalg.norm(x - x_old_iter) < tol:
            return x
            
    return x # Return best effort


def solve_newton_gs(u_old, dt, t_next, model, tol=1e-6, max_iter=20):
    """
    Solves the implicit equation F(u) = 0 using Newton's method.
    The linear system J*delta = -F is solved via Gauss-Seidel.
    
    F(u) = u - u_old - dt * f(t_next, u)
    J_F(u) = I - dt * J_f(u, t_next)
    """
    u_guess = u_old.copy() # Initial guess
    
    iters = 0
    converged = False
    
    inner_iters_total = 0
    
    for i in range(max_iter):
        iters += 1
        
        # 1. Compute Residual F(u)
        # F = u - u_old - dt * f(t, u)
        f_val = model.f(t_next, u_guess)
        F = u_guess - u_old - dt * f_val
        
        # Check if F is close to 0 (residual check)
        if np.linalg.norm(F) < tol:
            converged = True
            break
            
        # 2. Compute Jacobian J_F = I - dt * J_f
        J_f = model.jacobian(t_next, u_guess)
        J_F = np.eye(2) - dt * J_f
        
        # 3. Solve J_F * delta = -F using Gauss-Seidel
        # Initial guess for delta is 0
        delta_guess = np.zeros(2)
        delta = gauss_seidel_2x2_solve(J_F, -F, delta_guess)
        
        # 4. Update u
        u_new = u_guess + delta
        
        # Check step size convergence (optional but good)
        if np.linalg.norm(u_new - u_guess) < tol:
            converged = True
            u_guess = u_new
            break
            
        u_guess = u_new
        
    return u_guess, {'iters': iters, 'converged': converged}
