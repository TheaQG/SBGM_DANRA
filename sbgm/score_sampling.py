import torch
import tqdm
import logging

import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

def guided_score_fn(score_model,
                    x,
                    t,
                    y=None,
                    cond_img=None,
                    lsm_cond=None,
                    topo_cond=None,
                    scale=2.0,):
  '''
    A wrapper function for the score model to include classifier-free guidance.
    If model is trained without classifier-free guidance, the model will not use this function.
  '''
  # Define the null conditional image for classifier-free guidance.
  null_cond_img = torch.zeros_like(cond_img) if cond_img is not None else None
  # Define the null conditional lsm and topo for classifier-free guidance.
  null_lsm_cond = torch.zeros_like(lsm_cond) if lsm_cond is not None else None
  null_topo_cond = torch.zeros_like(topo_cond) if topo_cond is not None else None
  null_y = torch.zeros_like(y) - 1 if y is not None else None

  # Compute the score for the conditional and unconditional cases.
  score_cond = score_model(x, t, y, cond_img, lsm_cond, topo_cond)
  score_uncond = score_model(x, t, null_y, null_cond_img, null_lsm_cond, null_topo_cond)

  # The final score is a linear combination of the conditional and unconditional scores.
  return (1 + scale) * score_cond - scale * score_uncond


#@title Define the Euler-Maruyama sampler (double click to expand or collapse)

## The number of sampling steps.
num_steps =  500#@param {'type':'integer'}
def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=64, 
                           num_steps=num_steps, 
                           device='cuda', 
                           eps=1e-3,
                           img_size=64,
                           y=None,
                           cond_img=None,
                           lsm_cond=None,
                           topo_cond=None,
                           cfg=None
                           ):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns:
    Samples.    
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, 32, 32, device=device) \
    * marginal_prob_std(t)[:, None, None, None]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  mean_x = x  # Initialize mean_x to ensure it is always defined
  with torch.no_grad():
    for time_step in tqdm.tqdm(time_steps):
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)

      if cfg is None:
        cfg = {}
      if cfg.get('classifier_free_guidance', {}).get('enabled', False):
        # If classifier-free guidance is enabled, use the guided score function.
        scale = cfg['classifier_free_guidance'].get('guidance_scale', 2.0)
        score = guided_score_fn(score_model,
                                x,
                                batch_time_step,
                                y,
                                cond_img,
                                lsm_cond,
                                topo_cond,
                                scale=scale)
      else:
        # Else, use the standard score model (cheaper computation).
        score = score_model(x, batch_time_step, y, cond_img, lsm_cond, topo_cond)

      # Update the mean_x with the score model output.
      mean_x = x + (g**2)[:, None, None, None] * score * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
  # Do not include any noise in the last sampling step.
  return mean_x


#@title Define the Predictor-Corrector sampler (double click to expand or collapse)

signal_to_noise_ratio = 0.16 #@param {'type':'number'}

## The number of sampling steps.
num_steps =  800#@param {'type':'integer'}
def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64, 
               num_steps=num_steps, 
               snr=signal_to_noise_ratio,                
               device='cuda',
               eps=1e-3,
               img_size=64,
               y=None,
               cond_img=None,
               lsm_cond=None,
               topo_cond=None,
               cfg=None):
  """Generate samples from score-based models with Predictor-Corrector method.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient 
      of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.    
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns: 
    Samples.
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, img_size, img_size, device=device) * marginal_prob_std(t)[:, None, None, None]
  time_steps = np.linspace(1., eps, num_steps)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  x_mean = x  # Initialize x_mean to ensure it is always defined
  with torch.no_grad():
    for time_step in tqdm.tqdm(time_steps):
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      # Corrector step (Langevin MCMC)
      if cfg is None:
        cfg = {}
      if cfg.get('classifier_free_guidance', {}).get('enabled', False):
        # If classifier-free guidance is enabled, use the guided score function.
        scale = cfg['classifier_free_guidance'].get('guidance_scale', 2.0)
        score = guided_score_fn(score_model,
                                x,
                                batch_time_step,
                                y,
                                cond_img,
                                lsm_cond,
                                topo_cond,
                                scale=scale)
      else:
        # Else, use the standard score model (cheaper computation).
        score = score_model(x, batch_time_step, y, cond_img, lsm_cond, topo_cond)
      grad = score                                                                                       
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = np.sqrt(np.prod(x.shape[1:]))
      langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
      x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

      # Predictor step (Euler-Maruyama)
      g = diffusion_coeff(batch_time_step)

      if cfg.get('classifier_free_guidance', {}).get('enabled', False):
        # If classifier-free guidance is enabled, use the guided score function.
        scale = cfg['classifier_free_guidance'].get('guidance_scale', 2.0)
        score = guided_score_fn(score_model,
                                x,
                                batch_time_step,
                                y,
                                cond_img,
                                lsm_cond,
                                topo_cond,
                                scale=scale)
      else:
        # Else, use the standard score model (cheaper computation).
        score = score_model(x, batch_time_step, y, cond_img, lsm_cond, topo_cond)

      x_mean = x + (g**2)[:, None, None, None] * score * step_size

      # x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step, y, cond_img, lsm_cond, topo_cond) * step_size
      x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      
    
    # The last step does not include any noise
    return x_mean


#@title Define the ODE sampler (double click to expand or collapse)

from scipy import integrate

## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5 #@param {'type': 'number'}
def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                num_steps=100,
                batch_size=64, 
                atol=error_tolerance, 
                rtol=error_tolerance, 
                device='cuda', 
                z=None,
                eps=1e-3,
                img_size=64,
                y=None,
                cond_img=None,
                lsm_cond=None,
                topo_cond=None,
                cfg=None
                ):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.ones(batch_size, device=device)
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, 1, 32, 32, device=device) \
      * marginal_prob_std(t)[:, None, None, None]
  else:
    init_x = z
    
  shape = init_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
    with torch.no_grad():    
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)
  
  def ode_func(t, x):        
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t    
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
  
  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
  logger.info(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x
