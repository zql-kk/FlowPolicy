# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import torch
import numpy as np
from models import utils as mutils
from sde_lib import ConsistencyFM

def get_consistency_flow_matching_loss_fn(sde, reduce_mean=True, eps=1e-3):
  """Create a loss function for training with rectified flow.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
  hyperparameter = sde.consistencyfm_hyperparameters

  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A velocity model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    assert sde.reflow_flag == False, "not implemented"

    z0 = sde.get_z0(batch).to(batch.device)
    
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    r = torch.clamp(t + hyperparameter["delta"], max=1.0)

    t_expand = t.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
    r_expand = r.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
    xt = t_expand * batch + (1.-t_expand) * z0
    xr = r_expand * batch + (1.-r_expand) * z0
    
    segments = torch.linspace(0, 1, hyperparameter["num_segments"] + 1, device=batch.device)
    seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1) # .clamp(min=1) prevents the inclusion of 0 in indices.
    segment_ends = segments[seg_indices]
    
    segment_ends_expand = segment_ends.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
    x_at_segment_ends = segment_ends_expand * batch + (1.-segment_ends_expand) * z0
    
    def f_euler(t_expand, segment_ends_expand, xt, vt):
      return xt + (segment_ends_expand - t_expand) * vt
    def threshold_based_f_euler(t_expand, segment_ends_expand, xt, vt, threshold, x_at_segment_ends):
      if (threshold, int) and threshold == 0:
        return x_at_segment_ends
      
      less_than_threshold = t_expand < threshold
      
      res = (
        less_than_threshold * f_euler(t_expand, segment_ends_expand, xt, vt)
        + (~less_than_threshold) * x_at_segment_ends
        )
      return res
    
    model_fn = model
    
    rng_state = torch.cuda.get_rng_state()
    vt = model_fn(xt, t*999)
    
    torch.cuda.set_rng_state(rng_state) # Shared Dropout Mask
    with torch.no_grad():
      if (isinstance(hyperparameter["boundary"], int) 
          and hyperparameter["boundary"] == 0): # when hyperparameter["boundary"] == 0, vr is not needed
        vr = None
      else:
        vr = model_fn(xr, r*999)
        vr = torch.nan_to_num(vr)
      
    ft = f_euler(t_expand, segment_ends_expand, xt, vt)
    fr = threshold_based_f_euler(r_expand, segment_ends_expand, xr, vr, hyperparameter["boundary"], x_at_segment_ends)

    ##### loss #####
    losses_f = torch.square(ft - fr)
    losses_f = reduce_op(losses_f.reshape(losses_f.shape[0], -1), dim=-1)
    
    def masked_losses_v(vt, vr, threshold, segment_ends, t):
      if (threshold, int) and threshold == 0:
        return 0
    
      less_than_threshold = t_expand < threshold
      
      far_from_segment_ends = (segment_ends - t) > 1.01 * hyperparameter["delta"]
      far_from_segment_ends = far_from_segment_ends.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
      
      losses_v = torch.square(vt - vr)
      losses_v = less_than_threshold * far_from_segment_ends * losses_v
      losses_v = reduce_op(losses_v.reshape(losses_v.shape[0], -1), dim=-1)
      
      return losses_v
    
    losses_v = masked_losses_v(vt, vr, hyperparameter["boundary"], segment_ends, t)

    loss = torch.mean(
      losses_f + hyperparameter["alpha"] * losses_v
    )
    return loss

  return loss_fn


def get_step_fn(sde, reduce_mean=False, likelihood_weighting=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
  if isinstance(sde, ConsistencyFM):
    loss_fn = get_consistency_flow_matching_loss_fn(sde, reduce_mean=reduce_mean)
  else:
    raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    loss = loss_fn(model, batch)
    return loss

  return step_fn
