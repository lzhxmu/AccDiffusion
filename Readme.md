# AccDiffusion 
Code release for "AccDiffusion: An Accurate Method for Higher-Resolution Image Generation" 



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
