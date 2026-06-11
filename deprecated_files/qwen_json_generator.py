import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import json
import ast
import glob

from utils import load_embeddings_from_csv


use_minified_jsons = False  # whether use minified jsons
torch_device = "cpu"  # cuda

rico_elements_token_exceed_size_idxs = []


class UIDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=16384):
        self.embeddings = load_embeddings_from_csv()
        print("Loaded embedding csv.")
        self.tokenizer = tokenizer
        self.max_length = max_length
        #self.prompt = "Generate the UI JSON layout for this embedding:\n"
        self.prompt = ""

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        # load embedding
        embedding = torch.tensor(self.embeddings.iloc[idx]["embedding"], dtype=torch.float32)

        # load ui_data from json file
        ui_data: str = None
        with open(self.embeddings.iloc[idx]["json-path"]) as f:
            if use_minified_jsons:
                raise NotImplementedError("minified json loading not yet implemented")
                ui_data = minify_json(json.load(f))
            else:
                ui_data = f.read()

        assert isinstance(ui_data, str), f"ui_data must be str, found: {type(ui_data)}"

        # tokenization
        prompt_tokens = self.tokenizer(self.prompt, return_tensors="pt").input_ids[0]
        target_tokens = self.tokenizer(ui_data + self.tokenizer.eos_token, return_tensors="pt").input_ids[0]

        # check for bad lengths
        total_len = 1 + len(prompt_tokens) + len(target_tokens)

        faulty_element = {"idx": idx, "json-path": self.embeddings.iloc[idx]["json-path"]}
        # Use a list for order, but consider a set for faster lookups if the list gets huge
        if total_len > self.max_length and faulty_element not in rico_elements_token_exceed_size_idxs:
            print(f"\nWARNING: Sequence length {total_len} exceeds limit {self.max_length}. "
                  f"tThis will be shown only once.")
            print(f"Index: {idx}")
            # Fixed the nested double quotes by using single quotes inside the f-string
            print(f"File: {self.embeddings.iloc[idx]['json-path']}")
            rico_elements_token_exceed_size_idxs.append(faulty_element)
            print()

        return {"embedding": embedding, "prompt_tokens": prompt_tokens, "target_tokens": target_tokens}


def collate_fn(batch):
    embeddings = torch.stack([item["embedding"] for item in batch])
    max_prompt_len = max(len(item["prompt_tokens"]) for item in batch)
    max_target_len = max(len(item["target_tokens"]) for item in batch)

    prompt_tokens = torch.zeros((len(batch), max_prompt_len), dtype=torch.long)
    target_tokens = torch.zeros((len(batch), max_target_len), dtype=torch.long)

    # Labels padding with -100 so they are ignored in CrossEntropy calculation
    labels = torch.full((len(batch), max_prompt_len + max_target_len + 1), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        p_len = len(item["prompt_tokens"])
        t_len = len(item["target_tokens"])

        prompt_tokens[i, -p_len:] = item["prompt_tokens"]
        target_tokens[i, :t_len] = item["target_tokens"]

        # Only the target JSON tokens are scored by the loss function
        labels[i, 1 + p_len: 1 + p_len + t_len] = item["target_tokens"]

    return embeddings, prompt_tokens, target_tokens, labels


class UIJSONGenerator(nn.Module):
    def __init__(self, llm_path, embedding_dim=4096):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)

        # Load directly to GPU in bfloat16 for the A100
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            device_map=torch_device
        )

        # Freeze the base LLM completely for Phase 1
        for param in self.llm.parameters():
            param.requires_grad = False

        hidden_size = self.llm.config.hidden_size

        # The projection layer mapping 4096D to the LLM's dimension
        self.projector = nn.Linear(embedding_dim, hidden_size, dtype=torch.bfloat16).to(torch_device)

    def forward(self, embeddings, prompt_tokens, target_tokens, labels):
        # 1. Project embedding into native hidden dimension space [Batch, 1, Hidden_Size]
        projected_embed = self.projector(embeddings.to(torch.bfloat16).to(torch_device)).unsqueeze(1)

        # 2. Extract standard token embeddings for prompt and target text
        prompt_embeds = self.llm.model.embed_tokens(prompt_tokens.to(torch_device))
        target_embeds = self.llm.model.embed_tokens(target_tokens.to(torch_device))

        # 3. Concatenate all embeddings horizontally
        inputs_embeds = torch.cat([projected_embed, prompt_embeds, target_embeds], dim=1)

        # 4. Standard autoregressive forward pass
        outputs = self.llm(inputs_embeds=inputs_embeds, labels=labels.to(torch_device))
        return outputs.loss


def train():
    print("Using device:", torch_device)

    # Model Initialization
    model = UIJSONGenerator("Qwen/Qwen2.5-Coder-7B", embedding_dim=4096)
    print("Loaded model.")

    # Optimizer focused strictly on the new projection layer
    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=1e-3)
    print("Loaded optimizer.")

    # Load dataset - ensure directories are structured correctly
    dataset = UIDataset("../../rico-data/embeddings/embeddings_qwen.csv", tokenizer=model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

    print("Loaded dataset of size", len(dataset))

    model.train()
    epochs = 5

    print("Training model for", epochs, "epochs on device", torch_device)

    for epoch in range(epochs):
        # Initialize tqdm progress bar for the dataloader
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{epochs - 1}")

        for step, (embeddings, prompt_tokens, target_tokens, labels) in pbar:
            optimizer.zero_grad()
            loss = model(embeddings, prompt_tokens, target_tokens, labels)
            loss.backward()
            optimizer.step()

            # Update the progress bar with the latest loss
            if step % 1 == 0:
                pbar.set_postfix({
                    "step": step,
                    "loss": f"{loss.item():.4f}"
                })

    torch.save(model.state_dict(), "ui_json_generator.pth")
    print("Saved model state to ui_json_generator.pth")

if __name__ == "__main__":
    train()

    print("Finished training.")

    print("Elements exceeding token size limit:", set(rico_elements_token_exceed_size_idxs))
    print("exit.")