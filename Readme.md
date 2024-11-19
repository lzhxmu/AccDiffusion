# AccDiffusion 
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://lzhxmu.github.io/accdiffusion/accdiffusion.html)
[![arXiv](https://img.shields.io/badge/arXiv-2311.16973-b31b1b.svg)](https://arxiv.org/abs/2407.10738v1)

Code release for ["AccDiffusion: An Accurate Method for Higher-Resolution Image Generation"](https://lzhxmu.github.io/accdiffusion/accdiffusion.html) 

## News
- **2024.11.19**: 🔥 [AccDiffusion v2](https://github.com/lzhxmu/AccDiffusion_v2) is available!
- **2024.07.18**: 🔥 AccDiffusion has been accepted to ECCV'24!

## Experiments environment
### Set up the dependencies as:
```
conda create -n AccDiffusion python=3.9
conda activate AccDiffusion
pip install -r requirements.txt
```

## Higher-image generation
```
python accdiffusion_sdxl.py --experiment_name="AccDiffusion" \
    --model_ckpt="stabilityai/stable-diffusion-xl-base-1.0" \ # your sdxl model ckpt path
    --prompt="A cute corgi on the lawn." \
    --view_batch_size=16 \
    --num_inference_steps=50 \
    --seed=88 \
    --resolution="4096,4096" \
    --upscale_mode="bicubic_latent" \
    --stride=64 \
    --c=0.3 \ # c can be adjusted based on the degree of repetition and quality of the generated image
    --use_progressive_upscaling  --use_skip_residual --use_multidiffusion  --use_dilated_sampling --use_guassian  --use_md_prompt --shuffle 

``` 
