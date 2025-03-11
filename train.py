from dataclasses import dataclass
import torch
from latent_llm.data.text_dataset import TextDataset
from latent_llm.models.gpt_latent import LatentEncoder, LatentDecoder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import wandb
import logging
from rich.logging import RichHandler
from accelerate import Accelerator

accelerator = Accelerator()


# Update logging configuration to use RichHandler
logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    dataset_id: str = "anothy1/fineweb-edu-cleaned-simplified"
    split: str = "train"
    block_size: int = 1024
    n_gist_tokens: int = 256
    hub_repo_id: str = "toilaluan/smol-lm-2-135m-latent-encoder"
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    max_epochs: int = 10
    batch_size: int = 1
    num_workers: int = 8
    seed: int = 42
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    precision: str = "16-mixed" if device == "cuda" else "32"
    log_interval: int = 10
    validating_interval: int = 100
    save_interval: int = 1000
    max_new_tokens: int = 512
    training_steps: int = 100000
    wandb_project: str = "latent-llm"
    limit: int = 1000


CONFIG = TrainingConfig()
wandb.init(project=CONFIG.wandb_project)

print("--- Training Config ---")
logger.info(CONFIG)
print("---")

ENCODER = LatentEncoder(CONFIG.model_name, CONFIG.n_gist_tokens)
DECODER = LatentDecoder(CONFIG.model_name, CONFIG.n_gist_tokens)

TOKENIZER = AutoTokenizer.from_pretrained(CONFIG.model_name)
TOKENIZER.pad_token = TOKENIZER.eos_token


DATASET = TextDataset(
    dataset_id=CONFIG.dataset_id,
    split=CONFIG.split,
    block_size=CONFIG.block_size,
    model_name=CONFIG.model_name,
    limit=CONFIG.limit,
)


DATALOADER = DataLoader(
    DATASET,
    batch_size=CONFIG.batch_size,
    shuffle=True,
)


def training_step(batch: torch.Tensor) -> torch.Tensor:
    input_ids = batch[:, :-1].to(accelerator.device)
    labels = batch[:, 1:].to(accelerator.device)
    mem_embeds = ENCODER(input_ids, pad_token_id=TOKENIZER.pad_token_id)
    logits, loss = DECODER(input_ids, mem_embeds, labels=labels)
    return loss, mem_embeds, input_ids


current_step = 0

ENCODER.train()
DECODER.train()

for param in DECODER.parameters():
    param.requires_grad = False

OPTIMIZER = torch.optim.AdamW(
    list(ENCODER.parameters()),
    lr=CONFIG.learning_rate,
    weight_decay=CONFIG.weight_decay,
)

ENCODER, DECODER, DATALOADER, OPTIMIZER = accelerator.prepare(
    ENCODER, DECODER, DATALOADER, OPTIMIZER
)
ENCODER.to(accelerator.device)
DECODER.to(accelerator.device)

while True:
    OPTIMIZER.zero_grad()
    batch = next(iter(DATALOADER))
    loss, mem_embeds, input_ids = training_step(batch)
    wandb.log({"train/loss": loss.item()})
    accelerator.backward(loss)
    OPTIMIZER.step()
    current_step += 1
    if current_step % CONFIG.log_interval == 0:
        logger.info(f"[{current_step}/{CONFIG.training_steps}] loss: {loss.item()}")
    if current_step % CONFIG.save_interval == 0:
        logger.info("Saving to hub...")
        ENCODER.push_to_hub(CONFIG.hub_repo_id)

    if current_step % CONFIG.validating_interval == 0:
        logger.info("Generating...")
        ENCODER.eval()
        DECODER.eval()
        with torch.no_grad():
            batch = next(iter(DATALOADER))
            generated_ids = DECODER.generate(
                mem_embeds[:, :1, :],
                input_ids[:, :1],
                max_new_tokens=CONFIG.max_new_tokens,
            )
            completion = TOKENIZER.decode(generated_ids[0])
            label = TOKENIZER.decode(batch[:, 1:][0])
            # Log completion and input_ids
            wandb.log(
                {
                    "train/completion": wandb.Table(
                        columns=["Type", "Text"],
                        data=[["Completion", completion], ["Label", label]],
                    ),
                }
            )
            logger.info(
                f"[{current_step}/{CONFIG.training_steps}] completion: {completion[:32]}..."
            )
            logger.info(
                f"[{current_step}/{CONFIG.training_steps}] label: {label[:32]}..."
            )
        ENCODER.train()
        DECODER.train()

    if current_step >= CONFIG.training_steps:
        break
