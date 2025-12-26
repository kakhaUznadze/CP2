import numpy as np

class CisternModel:
    """
    Models the filling dynamics of a toilet cistern with a float valve.
    
    States u = [h, v]:
      h: Water height in the tank (m)
      v: Valve opening fraction (0 to 1)
      
    Parameters:
      A:       Cross-sectional area of cistern (m^2)
      Q_max:   Max inflow rate when valve is fully open (m^3/s)
      h_target: Target water level (m)
      k_valve: Valve response speed (1/s)
      k_leak:  Leak coefficient (m^2.5/s) - models small outflow/leak
      width_s:  Width parameter for sigmoid smoothing (m)
    """
    def __init__(self, config=None):
        if config is None:
            config = {}
            
        # Physical parameters with sensible defaults
        self.A = config.get('A', 0.12)           # ~30cm x 40cm tank
        self.Q_max = config.get('Q_max', 2.0e-4) # ~0.2 L/s
        self.h_target = config.get('h_target', 0.25) # 25cm target height
        self.k_valve = config.get('k_valve', 2.0)    # Response speed
        self.k_leak = config.get('k_leak', 1.0e-5)   # Small leak
        self.width_s = config.get('width_s', 0.01)   # Smoothing width for valve cutoff
        
        # Initial condition default
        self.u0 = np.array(config.get('u0', [0.0, 1.0])) # Empty tank, valve open

    def v_target_func(self, h):
        """
        Smooth target valve opening based on height.
        Ideally 1 when h << h_target, 0 when h >> h_target.
        Using sigmoid: 1 / (1 + exp((h - h_target)/width))
        But we want it to close as h increases.
        """
        # Sigmoid that goes from 1 to 0 around h_target
        # x = (h - h_target) / width_s
        # f(x) = 1 / (1 + exp(x))  -> 1 when h small (x neg big), 0 when h big (x pos big)
        
        # Avoid overflow in exp
        arg = (h - self.h_target) / self.width_s
        arg = np.clip(arg, -50, 50) 
        return 1.0 / (1.0 + np.exp(arg))

    def dv_target_dh(self, h):
        """Derivative of v_target_func with respect to h."""
        arg = (h - self.h_target) / self.width_s
        arg = np.clip(arg, -50, 50)
        ex = np.exp(arg)
        # d/dh [ (1+e^u)^-1 ] = -1 * (1+e^u)^-2 * e^u * du/dh
        # du/dh = 1/width_s
        denom = (1.0 + ex)**2
        return -ex / denom / self.width_s

    def f(self, t, u):
        """
        System RHS: u' = f(t, u)
        u[0] = h
        u[1] = v
        """
        h = u[0]
        v = u[1]
        
        # 1. dh/dt = (Qin - Qout) / A
        # Q_in = Q_max * v
        # Q_out = k_leak * sqrt(h)
        safe_h = max(h, 1e-9) # Avoid sqrt(negative)
        q_in = self.Q_max * v
        q_out = self.k_leak * np.sqrt(safe_h)
        dh_dt = (q_in - q_out) / self.A
        
        # 2. dv/dt = k_valve * (v_target - v)
        # Valve tries to match the target opening for current height
        vt = self.v_target_func(h)
        dv_dt = self.k_valve * (vt - v)
        
        return np.array([dh_dt, dv_dt])

    def jacobian(self, t, u):
        """
        Analytic Jacobian Matrix J = df/du
        J[i, j] = df_i / du_j
        
        f1 = (Q_max*v - k_leak*sqrt(h)) / A
        f2 = k_valve * (v_target(h) - v)
        """
        h = u[0]
        v = u[1]
        
        safe_h = max(h, 1e-9)
        
        # df1 / dh = - (k_leak / A) * (1 / (2*sqrt(h)))
        df1_dh = -(self.k_leak / self.A) * (0.5 / np.sqrt(safe_h))
        
        # df1 / dv = Q_max / A
        df1_dv = self.Q_max / self.A
        
        # df2 / dh = k_valve * dv_target_dh(h)
        df2_dh = self.k_valve * self.dv_target_dh(h)
        
        # df2 / dv = -k_valve
        df2_dv = -self.k_valve
        
        J = np.array([
            [df1_dh, df1_dv],
            [df2_dh, df2_dv]
        ])
        
        return J
