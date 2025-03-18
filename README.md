### Latent LLM: Combining Text VAE and Flow Matching Model [WIP Training]

Text VAE is autoregressive model that encodes text into a latent space, trained with CE Loss and KL Divergence.

Flow Matching Model is a Denoising Model that learns to generate completion latent from text and noise.

This novel approach let FM Model to thinks in latent space, it's non auto-regressive


### Installation

```
git clone https://github.com/toilaluan/latent-llm.git
cd latent-llm
pip install uv
uv sync
uv pip install flash-attn --no-build-isolation
```

### Training Text VAE

```
python train.py --model_name Qwen/Qwen2.5-Coder-0.5B-Instruct \
--block-size 32 \
--latent-size 64 \
--learning-rate 1e-4 \
--batch-size 64 \
--device cuda \
--log-interval 5 \
--validating-interval 500 \
--save-interval 1000 \
--kl-weight 1e-4 \
--hub-repo-id your-hub-namespace/latent-lm-vae \
--weight-decay 0
```

### Inference VAE

```
python inference.py --encoder_pretrained_id your-hub-namespace/latent-lm-vae-encoder \
--decoder_pretrained_id your-hub-namespace/latent-lm-vae-decoder \
--text "Hello, how are you?"
```

### Training Flow Matching Model

```
python latent_llm/train_flow_matching.py --model_name unsloth/Qwen2.5-0.5B-Instruct \
--encoder_id your-hub-namespace/latent-lm-vae-encoder \
--decoder_id your-hub-namespace/latent-lm-vae-decoder \
--block_size 32 \
--batch_size 64 \
--num_epochs 10000 \
--learning_rate 0.00001 \
--weight_decay 0.00001 \
--repeat_per_encode_pass 1 --use_wandb --dataset sentence-transformers/simple-wiki --eval_interval 200 --max_steps 1000 --num_samples 1
```




