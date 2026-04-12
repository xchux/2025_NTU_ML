# ML2025 Spring HW10: Diffusion

This homework explores personalized image generation using two diffusion-based methods: **BLIP Diffusion** and **Custom Diffusion**, applied to 6 custom objects.

## Methods

### Method 1 — BLIP Diffusion
Zero-shot subject-driven generation using [Salesforce/blipdiffusion](https://huggingface.co/docs/diffusers/en/api/pipelines/blip_diffusion). A single conditioning image and a text prompt are fed to the pipeline to generate 15 images per object.

Key hyperparameters (tunable):
- `num_inference_steps` — denoising steps (default: `10`)
- `guidance_scale` — text prompt adherence (default: `7.5`)

Results are saved to `results_blip/<object-id>/` and packed as `results_blip.zip`.

### Method 2 — Custom Diffusion
Fine-tuning of cross-attention layers (K/V projections) in [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) per object, plus a learned textual embedding (`<new1>` token initialized from `ktn`). Based on [diffusers Custom Diffusion guide](https://huggingface.co/docs/diffusers/en/training/custom_diffusion).

Key hyperparameters (tunable):
| Parameter | Default |
|---|---|
| `parameter_to_train` | `crossattn_kv` (`crossattn_kv` or `crossattn`) |
| `learning_rate` | `2e-5` |
| `max_train_steps` | `120` |
| `train_batch_size` | `2` |
| `num_inference_steps` | `80` |
| `guidance_scale` | `7.0` |

Checkpoints are saved under `output_custom/<object-id>/`, generated images under `results/<object-id>/`, and the final submission is packed as `results.zip`.

## Dataset

6 objects defined in `ml2025-hw10/metadata.json`:

| Object | Name | Text Condition | Baseline DINO | Baseline CLIP-T |
|---|---|---|---|---|
| object-1 | cat | on the grass | 68 | 18 |
| object-2 | pink sunglasses | on a cobblestone street | 60 | 17 |
| object-3 | backpack | in the jungle | 61 | 18 |
| object-4 | dog | with sunglasses | 68 | 19 |
| object-5 | toy | in the snow | 60 | 19 |
| object-6 | plushie | on a plate | — | — |

Data is located under `ml2025-hw10/data/<object-id>/`.

## Project Structure

```
ML2025_HW10.ipynb        # Main notebook
ml2025-hw10/
  metadata.json          # Object metadata (name, path, text_cond, baselines)
  data/<object-id>/      # Reference images per object
output_custom/
  <object-id>/
    pytorch_custom_diffusion_weights.safetensors
    <new1>.safetensors   # Learned token embedding
    checkpoint-120/      # Saved accelerator state
    logs/                # TensorBoard logs
results/                 # Custom Diffusion generated images (15 per object)
results_blip/            # BLIP Diffusion generated images (15 per object)
```

## Dependencies

Managed via `pyproject.toml` (requires Python ≥ 3.11):

```
accelerate
diffusers==0.33.1
numpy
Pillow
safetensors
tensorboard
torch
torchvision
tqdm
transformers==4.36.2
```

> **Compatibility notes:** The notebook includes patches for `transformers>=5` (missing `SiglipImageProcessor`/`SiglipVisionModel`) and changes to `Blip2QFormerModel`/`Blip2QFormerLayer` that were introduced after `transformers==4.36.2`.

## Submission

Each method produces a zip archive with 6 folders (`object-1` … `object-6`), each containing 15 JPEG images at 512×512 resolution.

```bash
# BLIP Diffusion submission
zip -r results_blip.zip results_blip

# Custom Diffusion submission
zip -r results.zip results
```
