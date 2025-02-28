"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np
from flow_policy_3d.models import utils as mutils


class ConsistencyFM():
    def __init__(self, init_type='gaussian', noise_scale=1.0,  use_ode_sampler='rk45', sigma_var=0.0, ode_tol=1e-5, sample_N=None):
      if sample_N is not None:
        self.sample_N = sample_N
        #print('Number of sampling steps:', self.sample_N)
      self.init_type = init_type
      
      self.noise_scale = noise_scale
      self.use_ode_sampler = use_ode_sampler
      self.ode_tol = ode_tol
      self.sigma_t = lambda t: (1. - t) * sigma_var
      #print('Init. Distribution Variance:', self.noise_scale)
      #print('SDE Sampler Variance:', sigma_var)
      #print('ODE Tolerence:', self.ode_tol)
      
      self.consistencyfm_hyperparameters = {
        "delta": 1e-3,
        "num_segments": 2,
        "boundary": 1, # NOTE If wanting zero, use 0 but not 0. or 0.0, since the former is integar.
        "alpha": 1e-5,
      }

    def T(self):
      return 1.

    def ode(self, init_input, model, reverse=False):
      ### run ODE solver for reflow. init_input can be \pi_0 or \pi_1
      from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
      from scipy import integrate
      rtol=1e-5
      atol=1e-5
      method='RK45'
      eps=1e-3

      # Initial sample
      x = init_input.detach().clone()

      model_fn = mutils.get_model_fn(model, train=False)
      shape = init_input.shape
      device = init_input.device

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = model_fn(x, vec_t*999)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      if reverse:
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(x),
                                                     rtol=rtol, atol=atol, method=method)
      else:
        solution = integrate.solve_ivp(ode_func, (eps, self.T), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
      nfe = solution.nfev
      #print('NFE:', nfe) 

      return x

    @torch.no_grad()
    def euler_ode(self, init_input, model, reverse=False, N=100):
      ### run ODE solver for reflow. init_input can be \pi_0 or \pi_1
      eps=1e-3
      dt = 1./N

      # Initial sample
      x = init_input.detach().clone()

      model_fn = mutils.get_model_fn(model)
      shape = init_input.shape
      device = init_input.device
      
      for i in range(N):  
        num_t = i / N * (self.T - eps) + eps      
        t = torch.ones(shape[0], device=device) * num_t
        pred = model_fn(x, t*999)
        
        x = x.detach().clone() + pred * dt         

      return x

    def get_z0(self, batch):
      B, N, D = batch.shape 

      if self.init_type == 'gaussian':
          ### standard gaussian #+ 0.5
          cur_shape = (B, N, D)
          return torch.randn(cur_shape)*self.noise_scale
      else:
          raise NotImplementedError("INITIALIZATION TYPE NOT IMPLEMENTED") 
      
