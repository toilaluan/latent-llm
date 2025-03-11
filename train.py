from latent_llm.models.latent_encoder import EncoderTransformer, DecoderTransformer
import tiktoken
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

N_LAYERS = 6
N_HEADS = 6
EMBED_DIM = 384
BLOCK_SIZE = 1024
MEM_SIZE = 128


def cycle(loader):
    while True:
        for data in loader:
            yield data


DATASET_ID = "meg/fineweb-bias-man-sentences"

TOKENIZER = tiktoken.get_encoding("gpt2")

DATASET = load_dataset(DATASET_ID, split="train").select(range(10))

VOCAB_SIZE = TOKENIZER.n_vocab
PADDING = TOKENIZER.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

TOTAL_STEPS = 10000
GENERATE_EVERY = 100
LOG_EVERY = 10


def tokenize_function(examples):
    tokenized_ids = TOKENIZER.encode_batch(examples["text"])
    for i in range(len(tokenized_ids)):
        tokenized_ids[i] = tokenized_ids[i][:BLOCK_SIZE] + [PADDING] * max(
            0, BLOCK_SIZE - len(tokenized_ids[i])
        )
    return {
        "input_ids": tokenized_ids,
    }


DATASET = DATASET.map(tokenize_function, batched=True)


def collate_fn(batch):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
    }


ENCODER = EncoderTransformer(
    VOCAB_SIZE, EMBED_DIM, N_HEADS, BLOCK_SIZE, N_LAYERS, MEM_SIZE
)
DECODER = DecoderTransformer(VOCAB_SIZE, EMBED_DIM, N_HEADS, BLOCK_SIZE, N_LAYERS)


OPTIMIZER = torch.optim.AdamW(
    list(ENCODER.parameters()) + list(DECODER.parameters()), lr=1e-4
)
DATALOADER = cycle(
    DataLoader(DATASET, batch_size=2, shuffle=True, collate_fn=collate_fn)
)

for step in range(TOTAL_STEPS):
    batch = next(DATALOADER)
    OPTIMIZER.zero_grad()
    x = batch["input_ids"]

    # Get memory embeddings from the full sequence
    mem_embeds = ENCODER(x)

    # Now prepare inputs and labels for the decoder
    labels = x[:, 1:]  # Shift right for next-token prediction
    decoder_input = x[:, :-1]  # Remove last token for input

    # Pass to decoder
    logits, loss = DECODER(decoder_input, mem_embeds, labels)
    loss.backward()
    OPTIMIZER.step()
    if step % LOG_EVERY == 0:
        print(f"Step {step}, Loss: {loss.item()}")
    if step % GENERATE_EVERY == 0:
        print(
            TOKENIZER.decode(
                DECODER.generate(mem_embeds, x[:, :1], 100, PADDING)[0].tolist()
            )
        )
