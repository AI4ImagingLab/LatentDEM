import argparse, os, sys, glob
import emutil.utils_agem as agem
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
# from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torch
import torch.nn.functional as F
import pickle
from PIL import Image

original_paths = sys.path
updated_paths = [
    p.replace('PSLD', 'PSLD_release') if ('PSLD' in p and 'PSLD_release' not in p) else p
    for p in original_paths
]
final_paths = list(dict.fromkeys(updated_paths))

# This is the crucial step: overwrite the old sys.path
sys.path = final_paths

from ldm.util import instantiate_from_config
from ldm.models.diffusion.psld import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from transformers import CLIPProcessor, CLIPModel

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

import pdb




def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x





def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_false',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=1000,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    ## 
    parser.add_argument(
        "--dps_path",
        type=str,
        default='../diffusion-posterior-sampling/',
        help="DPS codebase path",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default='configs/inpainting_config.yaml',
        help="task config yml file",
    )
    parser.add_argument(
        "--diffusion_config",
        type=str,
        default='configs/diffusion_config.yaml',
        help="diffusion config yml file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default='configs/model_config.yaml',
        help="model config yml file",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1e-1,
        help="inpainting error",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1,
        help="measurement error",
    )
    parser.add_argument(
        "--inpainting",
        type=int,
        default=0,
        help="inpainting",
    )
    parser.add_argument(
        "--general_inverse",
        type=int,
        default=1,
        help="general inverse",
    )
    parser.add_argument(
        "--file_id",
        type=str,
        default='00014.png',
        help='input image',
    )
    parser.add_argument(
        "--skip_low_res",
        action='store_true',
        help='downsample result to 256',
    )
    parser.add_argument(
        "--ffhq256",
        action='store_true',
        help='load SD weights trained on FFHQ',
    )
    ##

    opt = parser.parse_args()
    # pdb.set_trace()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        
    ## 
    if opt.ffhq256:
        print("Using FFHQ 256 finetuned model...")
        opt.config = "models/ldm/ffhq256/config.yaml"
        opt.ckpt = "models/ldm/ffhq256/model.ckpt"
    ##
    
    # seed_everything(opt.seed)
    seed = opt.seed
    np.random.seed(seed)
    lamb = 1
    beta = 5e6
    mask = None
    n_iter = 20
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    

    
    
    
    config = OmegaConf.load(f"{opt.config}")
    print('config:',config)
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('device_init:',device)
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
        print("sampler is  DPMSolverSample")
    elif opt.plms:
        sampler = PLMSSampler(model)
        print("sampler is PLMSSampler")
    else:
        # pdb.set_trace()
        sampler = DDIMSampler(model)
        print("sampler is DDIMSampler")
    
        

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    #########################################################
    ## DPS configs
    #########################################################
    sys.path.append(opt.dps_path)

    import yaml
    from guided_diffusion.measurements import get_noise, get_operator
    from util.img_utils import clear_color, mask_generator
    import torch.nn.functional as f
    import matplotlib.pyplot as plt


    def load_yaml(file_path: str) -> dict:
        with open(file_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    
    
    model_config=opt.dps_path+opt.model_config
    diffusion_config=opt.dps_path+opt.diffusion_config
    task_config=opt.dps_path+opt.task_config
    # pdb.set_trace()

    # Load configurations
    model_config = load_yaml(model_config)
    diffusion_config = load_yaml(diffusion_config)
    task_config = load_yaml(task_config)

    task_config['data']['root'] = opt.dps_path + 'dataforlatentDEM/samples/'
    img = plt.imread(task_config['data']['root']+opt.file_id)
    img = img - img.min()
    img = img / img.max()
    img = torch.FloatTensor(img)
    img = torch.unsqueeze(img, dim=0).permute(0,3,1,2)
    
    img = img[:,:3,:,:].to(device)
    img = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    kernel_orig=operator.get_kernel()
    y_n = agem.fft_blur(img, kernel_orig).to(device)
    y_n = noiser(y_n)
    

    kernel_size=measure_config['operator']['kernel_size']
    kernel_size=kernel_size*2
    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
        **measure_config['mask_opt']
        )
    
    y_n = f.interpolate(y_n, opt.H)
    img01 = y_n
    
    x_checked_image_torch = img01[:,:3,:,:].to(device)

    org_image = torch.clone(x_checked_image_torch[0].detach())
    
    org_image = org_image[None,:,:,:].to(device)
    org_image=org_image-org_image.min()
    org_image=org_image/org_image.max()
    org_image=(org_image-0.5)/0.5
    y_n = org_image
    print("noisy img input",org_image.min(),org_image.max())
    mask = None

    for img_dir in ['input', 'label','recover','process']:    
        os.makedirs(os.path.join(sample_path, img_dir),exist_ok=True)
    fname = str(beta)+str(lamb)+str(0).zfill(5) + '.png'
    plt.imsave(os.path.join(sample_path, 'input', 'gtimg'+str(seed)+fname), clear_color(img))
    plt.imsave(os.path.join(sample_path, 'input', 'y_n'+str(seed)+fname), clear_color(org_image))
    plt.imsave(os.path.join(sample_path, 'label', str(seed)+fname), kernel_orig[0,0].cpu().detach().numpy())
    
    
    # plt.imsave(os.path.join(sample_path, 'label', fname), kernel_orig) ##for the ldm process 

    ##record the original image
    ########################################################
    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            tic = time.time()
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.ffhq256:
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, intermediates ,k_0= sampler.sample(S=opt.ddim_steps,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        ip_mask = mask,
                                                        measurements = y_n,
                                                        operator = operator,
                                                        gamma = opt.gamma,
                                                        inpainting = opt.inpainting,
                                                        omega = opt.omega,
                                                        general_inverse=opt.general_inverse,
                                                        noiser=noiser,
                                                        ffhq256=opt.ffhq256,kernel_size=kernel_size)
                    else:
                        # pdb.set_trace()
                        if opt.scale != 1.0 :
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        
                        samples_ddim, intermediates ,k_0= sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        ip_mask = mask,
                                                        measurements = y_n,
                                                        operator = operator,
                                                        gamma = opt.gamma,
                                                        inpainting = opt.inpainting,
                                                        omega = opt.omega,
                                                        general_inverse=opt.general_inverse,
                                                        noiser=noiser,
                                                        n_iter=n_iter,
                                                        kernel_size=kernel_size,lamb=lamb,beta=beta,
                                                        base_dir = base_dir)


                   
                    k_0 = k_0[0,0].cpu().detach().numpy()
                    k_0 /= k_0.max()
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                    # x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                    
                    x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                    
                    
                    if not opt.skip_save:
                        for x_sample in x_checked_image_torch:
                            if opt.inpainting:  # Inpainting gluing logic as in SD inpaint.py
                                image = torch.clamp((org_image+1.0)/2.0, min=0.0, max=1.0)
                                image = image.cpu().numpy()

                                mask = mask.cpu().numpy()
                                
                                inpainted = mask*image+(1-mask)*x_sample.cpu().numpy()
                                inpainted = inpainted.transpose(0,2,3,1)[0]*255
                                Image.fromarray(inpainted.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:05}.png"))
                            else:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img02 = Image.fromarray(x_sample.astype(np.uint8))
                                # img = put_watermark(img, wm_encoder)
                                img02.save(os.path.join(sample_path,str(seed)+ f"temp_{base_count:05}.png"))

                            base_count += 1
                        plt.imsave(os.path.join(sample_path, 'label', str(beta)+str(seed)+'kernel_recover.png'), k_0)
                        

                    if not opt.skip_grid:
                        all_samples.append(x_checked_image_torch)
                    
                    # pdb.set_trace()
                    if not opt.skip_low_res:
                        if not opt.skip_save:
                            inpainted_image_low_res = f.interpolate(x_checked_image_torch.type(torch.float32), size=(opt.H//2, opt.W//2))
                            
                            for x_sample in inpainted_image_low_res:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                # img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, str(seed)+f"{base_count:05}_low_res.png"))
                                base_count += 1
                    ##watch the process of recover:
                    for count,(x_inter,k_inter) in enumerate(zip(intermediates['x_inter'],intermediates["k_0_inter"]),start = 1):
                        fname = str(count) + '.png'
                        plt.imsave(os.path.join(sample_path, 'recover','x_inter'+ fname), x_inter[0,0].cpu().detach().numpy())
                        plt.imsave(os.path.join(sample_path, 'recover', "k_inter"+fname), k_inter[0,0].cpu().detach().numpy())
                        

            if not opt.skip_grid:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                img = Image.fromarray(grid.astype(np.uint8))
                # img = put_watermark(img, wm_encoder)
                img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                grid_count += 1

            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")
    print("lamb and beta:",lamb,beta)


if __name__ == "__main__":
    main()
