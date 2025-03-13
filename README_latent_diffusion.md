# Latent Diffusion Model for Text Generation

This project implements a diffusion model for text generation by operating in the latent space of a pre-trained language model. The approach consists of the following steps:

1. Encode text to latent representations using a trained latent encoder
2. Train a diffusion model to generate these latent representations from random noise
3. Generate new latent representations using the diffusion model
4. Decode these latent representations back to text using a trained latent decoder

## Architecture

The implementation provides two different model architectures for the diffusion model:

1. **MLP-based UNet Architecture** (`LatentDiffusionModel`): A simple MLP-based network with skip connections, designed specifically for latent space diffusion.

2. **HuggingFace UNet Architecture** (`HFStyleUNetLatentDiffusion`): Uses the HuggingFace UNet2DModel by reshaping the 1D latent vectors to 2D tensors, which allows us to leverage the powerful convolutional UNet architecture from the diffusers library.

## Installation

### Requirements

```
torch>=2.0.0
diffusers>=0.18.0
transformers>=4.28.0
accelerate>=0.19.0
datasets>=2.12.0
huggingface_hub>=0.14.0
tqdm
rich
```

Install the requirements using:

```bash
pip install -r requirements.txt
```

## Training

To train a latent diffusion model, you need:

1. A pre-trained latent encoder model
2. A dataset of text samples

Use the `train_latent_diffusion.py` script:

```bash
python train_latent_diffusion.py \
    --encoder_model_id "your-encoder-model-id" \
    --output_dir "path/to/save/model" \
    --dataset_path "path/to/dataset" \
    --num_epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --use_hf_unet  # Optional: use the HuggingFace UNet architecture
```

### Training Process

The training process follows the standard diffusion model training approach:

1. Encode text samples to latent representations using the pre-trained encoder
2. Add noise to these latent representations according to a noise schedule
3. Train the diffusion model to predict the noise that was added
4. During inference, start with random noise and gradually denoise it to produce latent representations

### Key Training Parameters

- `--encoder_model_id`: ID of the pre-trained latent encoder (required)
- `--output_dir`: Directory to save model checkpoints (required)
- `--dataset_path`: Path to HuggingFace dataset containing text samples (required)
- `--num_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--gradient_accumulation_steps`: Number of steps for gradient accumulation (default: 1)
- `--lr_warmup_steps`: Number of warmup steps for learning rate scheduler (default: 500)
- `--save_image_epochs`: Save sample generated latents every N epochs (default: 10)
- `--save_model_epochs`: Save model checkpoint every N epochs (default: 50)
- `--mixed_precision`: Mixed precision type (fp16, bf16, no) (default: fp16)
- `--use_hf_unet`: Use HuggingFace UNet instead of custom MLP (flag)
- `--push_to_hub`: Push model to Hugging Face Hub (flag)
- `--hub_model_id`: Model ID for pushing to Hub (optional)
- `--seed`: Random seed (default: 42)

## Inference

To generate text using a trained latent diffusion model, use the `generate_with_diffusion.py` script:

```bash
python generate_with_diffusion.py \
    --diffusion_model_path "path/to/diffusion/model" \
    --decoder_model_id "your-decoder-model-id" \
    --num_samples 5 \
    --num_inference_steps 1000 \
    --max_new_tokens 100 \
    --temperature 0.7
```

### Inference Process

The inference process consists of the following steps:

1. Generate random noise
2. Use the diffusion model to gradually denoise it into latent representations
3. Reshape the latent representations to match the decoder's expected format
4. Use the latent decoder to generate text from these latent representations

### Key Inference Parameters

- `--diffusion_model_path`: Path to the trained diffusion model directory (required)
- `--decoder_model_id`: HuggingFace model ID for the latent decoder (required)
- `--num_samples`: Number of samples to generate (default: 5)
- `--num_inference_steps`: Number of denoising steps (default: 1000)
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 100)
- `--temperature`: Temperature for sampling (default: 0.7)
- `--seed`: Random seed (default: 42)
- `--use_hf_unet`: Whether the diffusion model uses HuggingFace UNet (flag)
- `--device`: Device to run inference on (default: cuda if available, otherwise cpu)

## Example End-to-End Workflow

1. **Train a latent encoder and decoder**:
   First, train a latent encoder and decoder using the core latent-llm training scripts.

2. **Prepare a dataset**:
   Create or download a dataset of text samples.

3. **Train the diffusion model**:
   ```bash
   python train_latent_diffusion.py \
       --encoder_model_id "your-encoder-model-id" \
       --output_dir "latent_diffusion_model" \
       --dataset_path "text_dataset" \
       --num_epochs 100 \
       --batch_size 16 \
       --use_hf_unet
   ```

4. **Generate text**:
   ```bash
   python generate_with_diffusion.py \
       --diffusion_model_path "latent_diffusion_model/final_model" \
       --decoder_model_id "your-decoder-model-id" \
       --num_samples 5 \
       --num_inference_steps 1000 \
       --max_new_tokens 100
   ```

## References

This implementation is inspired by:

1. [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) by Rombach et al.
2. [HuggingFace Diffusers Library](https://github.com/huggingface/diffusers)
3. [minDALL-E](https://github.com/kakaobrain/minDALL-E) by Kim et al.
4. [minRF](https://github.com/cloneofsimo/minRF) by Simo

## License

This project is licensed under the MIT License. 