# Numerical Analysis of Toilet Cistern Filling Dynamics
**Course Project #2: Numerical Programming**

## 1. Introduction
This project models the filling dynamics of a toilet cistern (water tank) equipped with a float valve. This is a common real-world control system where a valve automatically closes as the water level rises to a target height.

Mathematically, this system is modeled as a system of Ordinary Differential Equations (ODEs). We simulate this system using the **Implicit Backward Euler** method to ensure stability even with stiff dynamics. The implicit modification imposes a nonlinear system of algebraic equations at each time step, which we solve using two different iterative methods:
1.  **Fixed-Point Iteration**
2.  **Newton-Gauss-Seidel Method** (Newton's method with a manual Gauss-Seidel linear solver)

We compare these methods in terms of computational efficiency (number of iterations) and robustness.

## 2. Mathematical Model
The system is defined by two state variables:
- $h(t)$: Water level in the cistern [m].
- $v(t)$: Valve opening fraction [0, 1] (unitless).

The governing equations are:
$$
\begin{aligned}
\frac{dh}{dt} &= \frac{1}{A} (Q_{in} - Q_{out}) \\
\frac{dv}{dt} &= k_{valve} (v_{target}(h) - v)
\end{aligned}
$$

### 2.1 Physics
- **Inflow**: $Q_{in} = Q_{max} \cdot v(t)$. The flow is proportional to the valve opening.
- **Outflow/Leak**: $Q_{out} = k_{leak} \sqrt{h(t)}$. Modeled using Torricelli's law (simulating a small leak or usage).
- **Valve Dynamics**: The valve adjusts its position $v$ towards a target $v_{target}(h)$ with a response speed $k_{valve}$.

### 2.2 Target Function (Float Mechanism)
The float mechanism implies that the valve should be fully open ($1$) when empty and closed ($0$) when full (at $h_{target}$).
To allow for differentiability (required for Newton's method), we approximate the logical cutoff with a smooth sigmoid function:
$$
v_{target}(h) = \frac{1}{1 + \exp\left(\frac{h - h_{target}}{w}\right)}
$$
where $w$ is a small smoothing width.

### 2.3 Parameters
| Parameter | Symbol | Value | Units | Description |
|-----------|--------|-------|-------|-------------|
| Area | $A$ | 0.12 | $m^2$ | Cross-sectional area (~30x40cm) |
| Max Flow | $Q_{max}$ | 2.0e-4 | $m^3/s$ | Max inflow rate |
| Target H | $h_{target}$ | 0.25 | $m$ | Desired fill level |
| Valve Speed | $k_{valve}$ | 2.0 | $s^{-1}$ | How fast valve reacts |
| Leak Coeff | $k_{leak}$ | 1.0e-5 | $m^{2.5}/s$ | Small leak coefficient |

## 3. Numerical Methods

### 3.1 Time Discretization: Backward Euler
We solve the IVP $u' = f(t, u)$ using the implicit backward Euler scheme:
$$
u_{n+1} = u_n + \Delta t \cdot f(t_{n+1}, u_{n+1})
$$
This requires solving for $u_{n+1}$ at every step.

### 3.2 Method A: Fixed-Point Iteration
We rewrite the equation as a fixed-point problem $u = G(u)$:
$$
G(u) = u_n + \Delta t \cdot f(t_{n+1}, u)
$$
We iterate $u^{(k+1)} = G(u^{(k)})$ until convergence ($||u^{(k+1)} - u^{(k)}|| < \epsilon$).
To improve stability, we can use relaxation:
$$
u^{(k+1)} = (1-\omega)u^{(k)} + \omega G(u^{(k)})
$$

### 3.3 Method B: Newton-Gauss-Seidel
We find the root of the residual function:
$$
F(u) = u - u_n - \Delta t \cdot f(t_{n+1}, u) = 0
$$
Newton's update is $\Delta u = - J_F(u)^{-1} F(u)$, where the Jacobian is:
$$
J_F = I - \Delta t \cdot J_f(u)
$$
We compute $J_f$ analytically (derivatives of the model).
Instead of inverting $J_F$, we solve the linear system $J_F \Delta u = -F$ using the **Gauss-Seidel** method manually.

## 4. Results & Comparison

### 4.1 Simulation Setup
- **Time Step**: $\Delta t = 1.0s$ (and tested others)
- **Duration**: $300s$ (enough to reach steady state)
- **Initial Condition**: Empty tank ($h=0$), valve open ($v=1$).

### 4.2 Plots
Figures (saved in `outputs/figures/`) show that both methods produce identical trajectories for $h(t)$ and $v(t)$.
The water level rises linearly initially, then slows down as the valve closes near $h=0.25m$.

### 4.3 Performance Table
| Method | $\Delta t$ | Avg Niters | Runtime (s) | Robustness |
|--------|------------|------------|-------------|------------|
| Fixed-Point | 1.0s | ~20 | low | Converges slowly, needed relaxation for large dt |
| Newton-GS | 1.0s | ~3 | slightly higher | Very fast quadratic convergence |

### 4.4 Discussion
- **Fixed-Point** is simple to implement but requires many iterations per step, especially as $\Delta t$ increases or the system becomes stiffer (fast valve dynamics).
- **Newton-GS** is more complex (requires Jacobian) but converges in very few iterations (usually 2-4).
- The "manual" Gauss-Seidel inner solver works well here because the system is small (2x2) and diagonally dominant enough for small $\Delta t$.

## 5. Conclusion
Both methods successfully solve the nonlinear system arising from the implicit discretization.
- The **Fixed-Point** method is viable for small functional dependencies but struggles with stiffness.
- The **Newton-Gauss-Seidel** method is far superior in terms of iteration count and robustness, making it the preferred choice for this control system simulation.

The model correctly predicts the filling behavior: water rises to the target level and the valve shuts off automatically, maintaining the steady state.
