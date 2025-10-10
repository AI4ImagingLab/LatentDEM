"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import os
import math
import torch.nn.functional as f

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

import pdb

### fasr-em used
from networks.network_dncnn import FDnCNN as net_kernel
import emutil.utils_agem as agem


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    # @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               ip_mask = None, measurements = None, operator = None, gamma = 1, inpainting = False, omega=1,
               general_inverse = None, noiser=None,
               ffhq256=False,n_iter=10,kernel_size=None,
               lamb=1 , beta = 5e6,k_0 = None, base_dir = None
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
            #    **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        else:
            print('Running unconditional generation...')
        
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates,k_0 = \
        self.ddim_sampling(conditioning,                               size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    ip_mask = ip_mask, measurements = measurements, operator = operator,
                                                    gamma = gamma,
                                                    inpainting = inpainting, omega=omega,
                                                    general_inverse = general_inverse, noiser = noiser,
                                                    ffhq256=ffhq256,
                                                    n_iter=n_iter,
                                                    kernel_size=kernel_size,lamb=lamb,beta=beta,k_0=k_0, base_dir=base_dir)
        return samples, intermediates,k_0

    ## lr
    # @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      ip_mask = None, measurements = None, operator = None, gamma = 1, inpainting=False, omega=1,
                      general_inverse = None, noiser=None,
                      ffhq256=False,n_iter=10,kernel_size=None,lamb=None,beta=None,k_0 = None,base_dir = None):
        device = self.model.betas.device
        print("device:",device)
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img],"k_0_inter":[]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        
        
        ### load the pnp denoiser
        if k_0 == None:
            k_0 = self.init_ker(ksize=kernel_size, device=device)
        reg_path = os.path.join(base_dir, "src/model_zoo/model_5000_for_exp64.pth")
        kernel_denoiser = torch.load(reg_path, map_location=device)
        


        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            
            #print('index:', index)
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      ip_mask = ip_mask, measurements = measurements, operator = operator, gamma = gamma,
                                      inpainting=inpainting, omega=omega,
                                      gamma_scale = index/total_steps,
                                      general_inverse=general_inverse, noiser=noiser,
                                      ffhq256=ffhq256,kernel_denoiser=kernel_denoiser,k_0=k_0,n_iter=n_iter,kernel_size=kernel_size,lamb=lamb, beta = beta)
            img, pred_x0,k_0 = outs
            # if callback: callback(i)
            # if img_callback: img_callback(pred_x0, i)
            
            # if index % 10 == 0 or index == total_steps - 1:
            #     intermediates['x_inter'].append(img)
            #     intermediates["k_0_inter"].append(k_0)

        return img, intermediates,k_0

    ######################
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      ip_mask=None, measurements = None,  gamma=1, inpainting=False,operator=None,
                      gamma_scale = None, omega = 1e-1,
                      general_inverse=False,noiser=None,
                      ffhq256=False,kernel_denoiser=None,k_0=None,n_iter=10,kernel_size=None, lamb=None, beta=None):
        b, *_, device = *x.shape, x.device
           
        ##########################################
        ## measurment consistency guided diffusion
        ##########################################
        if inpainting:
            return 0
        
        elif general_inverse:
            
            
            # print('Running general inverse module...')
            z_t = torch.clone(x.detach())
            z_t.requires_grad = True
            
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(z_t, t, c)
                
            else:
                
                x_in = torch.cat([z_t] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            
            
            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, z_t, t, c, **corrector_kwargs)
            
            
            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
            
            # current prediction for x_0
            pred_z_0 = (z_t - sqrt_one_minus_at * e_t) / a_t.sqrt()
            
            
            
            if quantize_denoised:
                pred_z_0, _, *_ = self.model.first_stage_model.quantize(pred_z_0)
            
            
            # direction pointing to x_t
            dir_zt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            
            z_prev = a_prev.sqrt() * pred_z_0 + dir_zt + noise
            ## you can use the skip tech mentioned in https://arxiv.org/abs/2407.01027,perform your own skipping setting
            # if index%16 !=0 and index>40:
                # return z_prev.detach(), pred_z_0.detach(),k_0
            ##############################################
            
            image_pred = self.model.differentiable_decode_first_stage(pred_z_0).to(torch.float32)
            # print("diffusion image out",image_pred.min(),image_pred.max())
            #here is the annealing tech
            # if index>600:
            #     scale01=0.1+((0.8-0.1)/400)*(1000-index)
                
            # else:
            #     scale01=0.8


            # Here we also provide the implementations of different anealing tech like exponential and cosine annealing

            start_value = 0.8
            end_value = 0.1
            start_index = 600  # Annealing starts after this index
            end_index = 1000   # Annealing ends at this index

            # --- Annealing Logic (inside your loop) ---
            # Assuming 'index' is your loop variable, e.g., for index in range(1001):
            if index > start_index:
                # Calculate progress as a fraction from 0.0 to 1.0
                progress = (index - start_index) / (end_index - start_index)

                # Apply exponential decay formula
                scale01 = start_value * (end_value / start_value) ** progress
            else:
                scale01 = start_value


            # start_value = 0.8
            # end_value = 0.1
            # start_index = 600
            # end_index = 1000

            # --- Annealing Logic (inside your loop) ---
            # Assuming 'index' is your loop variable, e.g., for index in range(1001):
            # if index > start_index:
            #     # Calculate progress as a fraction from 0.0 to 1.0
            #     progress = (index - start_index) / (end_index - start_index)

            #     # Apply cosine annealing formula
            #     scale01 = end_value + 0.5 * (start_value - end_value) * (1 + math.cos(progress * math.pi))
            # else:
            #     scale01 = start_value
            ####the M_step comes in here:
            sigma_em=0.02
            with torch.no_grad():
                
                
                    k_0 = self.M_step(measurements, image_pred, sigma_em, k_0, lamb, beta ,kernel_denoiser,kernel_size,n_iter,index=index)
                    k_0 = k_0.type(torch.float32)
                    k_0 = torch.FloatTensor(agem.kernel_shift(k_0[0,0].cpu().numpy(), 1)).to(device)[None, None]
            forward_model = lambda x: agem.fft_blur(x, k_0)
            # print(k_0.shape)
            
            
            ##for using the 256 image to process,original 见github
            # print('ldm',image_pred.shape)
            meas_pred = forward_model(image_pred)
            meas_pred = noiser(meas_pred)
   

            

            meas_error = torch.linalg.norm(meas_pred - measurements)
            # print('meas_error',meas_error)
            
            ortho_project = image_pred - forward_model(image_pred)
            parallel_project = measurements
            inpainted_image = parallel_project + ortho_project
            
            # encoded_z_0 = self.model.encode_first_stage(inpainted_image) if ffhq256 else self.model.encode_first_stage(inpainted_image).mean  
            encoded_z_0 = self.model.encode_first_stage(inpainted_image)
            encoded_z_0 = self.model.get_first_stage_encoding(encoded_z_0)
            inpaint_error = torch.linalg.norm(encoded_z_0 - pred_z_0)
            ##here changing the value of scale
            error = (inpaint_error * gamma + meas_error * (omega))*scale01

            # error =  meas_error * (omega)*scale01
            
            gradients = torch.autograd.grad(error, inputs=z_t)[0]
            z_prev = z_prev - gradients
            # print('Loss: ', error.item())
            # assert not math.isnan(error.item())
            
            return z_prev.detach(), pred_z_0.detach(),k_0
        
        
        #########################################
        else:
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                with torch.no_grad():
                    e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                ## lr
                with torch.no_grad():
                    e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                ## lr
                with torch.no_grad():
                    e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                ## 
                with torch.no_grad():
                    pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            return x_prev, pred_x0
    
    ######################
    
    #@torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    #@torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec
    
    
    ###helper function for fast em
    def init_ker(self, ksize=64, device='cuda'):
        k = torch.zeros((1, 1, ksize, ksize)).to(device)
        k[0, 0, ksize//2, ksize//2] = 1
        return k
    def load_network(self, net, path, device='cuda'):
        net.load_state_dict(torch.load(path))
        print(path)
        return net.to(device)
    def M_step(self, y, x, sigma, k, lamb, beta,kernel_denoiser, kernel_size, n_iter ,index):
        return agem.HQS_ker(y, x, sigma, k,denoiser=kernel_denoiser,lamb=lamb, beta=beta,reg='pnp', n_iter=n_iter,ksize=kernel_size,index=index)
    ##Change the n_iter number to 20