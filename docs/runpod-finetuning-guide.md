# Fine-Tuning on RunPod — Step-by-Step Guide

This guide walks through the complete workflow: spinning up a GPU pod on RunPod, generating a dataset with `main.py`, and fine-tuning a model with `finetuning.py`.

---

## Prerequisites

- A [RunPod](https://www.runpod.io) account with credits or a payment method
- A [Firecrawl API key](https://firecrawl.dev)
- An [OpenAI API key](https://platform.openai.com)
- (Optional) A [Hugging Face](https://huggingface.co) account + token if you want to push your model to the Hub

---

## Step 1 — Create a GPU Pod

1. Log in to [RunPod Console](https://www.runpod.io/console/pods)
2. Click **+ Deploy**
3. Select a GPU:

   | GPU | VRAM | Good For |
   |:----|:-----|:---------|
   | RTX A4000 | 16 GB | 1B–3B models |
   | RTX A5000 / A40 | 24–48 GB | 3B–8B models |
   | A100 (40/80 GB) | 40–80 GB | 8B+ models |

   > For `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` (the default), a 16 GB GPU is sufficient.

4. Under **Pod Template**, select the official **RunPod PyTorch** template (comes with PyTorch, CUDA, and Python pre-installed)
5. Configure storage:
   - **Container Disk**: 20 GB (minimum)
   - **Volume Disk**: 50 GB (recommended — persists across pod restarts, stores your model weights)
   - Volume mounts at `/workspace` by default
6. Click **Deploy On-Demand** (or Spot if you want cheaper rates with possible interruptions)
7. Wait 30–120 seconds for the pod to initialize

---

## Step 2 — Connect to Your Pod

You have two options:

### Option A — JupyterLab (easiest)

1. Once the pod is running, click **Connect**
2. Click **Connect to Jupyter Lab** (opens in your browser on port 8888)
3. Open a **Terminal** from the JupyterLab launcher

### Option B — SSH

1. Click **Connect** on your pod
2. Copy the SSH command shown (looks like `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`)
3. Run it in your local terminal

> Both options give you a root shell. All subsequent steps are run in this terminal.

---

## Step 3 — Clone the Repo and Install Dependencies

```bash
cd /workspace
git clone <your-repo-url> firecrawl_finetuning
cd firecrawl_finetuning
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

> `unsloth` will install its optimized kernels for your specific GPU automatically. This may take a few minutes on first install.

---

## Step 4 — Set Up Environment Variables

```bash
cp .env.example .env
```

Edit the `.env` file with your API keys:

```bash
nano .env
```

Add:

```
FIRECRAWL_API_KEY=fc-your-key-here
OPENAI_API_KEY=sk-your-key-here
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

---

## Step 5 — Generate the Training Dataset

Run the dataset generation pipeline. Replace the URL with the website you want to train on:

```bash
python main.py https://docs.example.com
```

For more control:

```bash
python main.py https://docs.example.com \
  --max-pages 100 \
  --qa-per-chunk 3 \
  --chunk-size 2000 \
  --concurrency 15 \
  --multi-turn \
  --output-dir ./output
```

When finished, verify your dataset:

```bash
wc -l ./output/dataset_sharegpt.jsonl
# Should show the number of training examples

head -1 ./output/dataset_sharegpt.jsonl | python -m json.tool
# Should show a properly formatted ShareGPT entry
```

> This step uses the OpenAI API (not the GPU). You can also run it locally and upload the JSONL file to the pod via JupyterLab's file browser.

---

## Step 6 — Verify GPU Access

Before training, confirm PyTorch can see the GPU:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:

```
CUDA available: True
GPU: NVIDIA A40
```

---

## Step 7 — Run Fine-Tuning

### Quick test run (10 steps)

Start with a short run to make sure everything works:

```bash
python finetuning.py \
  --dataset ./output/dataset_sharegpt.jsonl \
  --max-steps 10
```

### Full training run

Once the test passes, run a full training:

```bash
python finetuning.py \
  --dataset ./output/dataset_sharegpt.jsonl \
  --model unsloth/Llama-3.2-1B-Instruct-bnb-4bit \
  --num-epochs 3 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --learning-rate 2e-4 \
  --lora-rank 16 \
  --lora-alpha 16 \
  --warmup-steps 10 \
  --output-dir ./finetuned_model
```

### All training options

| Flag | Default | Description |
|:-----|:--------|:------------|
| `--dataset` | `./output/dataset_sharegpt.jsonl` | Path to ShareGPT JSONL file |
| `--model` | `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` | Base model (HF or Unsloth ID) |
| `--max-seq-length` | `2048` | Max sequence length |
| `--chat-template` | `chatml` | Chat template (chatml, llama, mistral, zephyr) |
| `--lora-rank` | `16` | LoRA rank |
| `--lora-alpha` | `16` | LoRA alpha |
| `--batch-size` | `2` | Per-device batch size |
| `--gradient-accumulation` | `4` | Gradient accumulation steps |
| `--learning-rate` | `2e-4` | Learning rate |
| `--num-epochs` | `1` | Training epochs |
| `--max-steps` | `-1` | Max steps (-1 = use epochs) |
| `--warmup-steps` | `5` | LR warmup steps |
| `--logging-steps` | `1` | Log every N steps |
| `--output-dir` | `./finetuned_model` | Where to save LoRA adapters |
| `--push-to-hub` | None | HF Hub repo (e.g. `username/my-model`) |
| `--gguf` | None | GGUF export (q4_k_m, q8_0, f16) |

---

## Step 8 — Verify the Output

After training completes, check the saved model:

```bash
ls ./finetuned_model/
```

You should see LoRA adapter files:

```
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

---

## Step 9 — Export and Download Your Model

### Option A — Push to Hugging Face Hub

Re-run finetuning with the `--push-to-hub` flag (or do it after training):

```bash
# Login first
huggingface-cli login

# Run with push
python finetuning.py \
  --dataset ./output/dataset_sharegpt.jsonl \
  --push-to-hub your-username/my-finetuned-model \
  --num-epochs 3
```

### Option B — Export as GGUF (for Ollama / llama.cpp)

```bash
python finetuning.py \
  --dataset ./output/dataset_sharegpt.jsonl \
  --gguf q4_k_m \
  --num-epochs 3
```

The GGUF file will be saved to `./finetuned_model_gguf/`.

### Option C — Download via JupyterLab

1. Open JupyterLab in your browser
2. Navigate to the `finetuned_model/` directory in the file browser
3. Right-click files and select **Download**

### Option D — Download via SCP

From your **local machine**:

```bash
scp -P <port> root@<pod-ip>:/workspace/firecrawl_finetuning/finetuned_model/* ./my-local-model/
```

---

## Step 10 — Stop Your Pod

Once you have downloaded or pushed your model, **stop the pod** to avoid charges:

1. Go to [RunPod Console > Pods](https://www.runpod.io/console/pods)
2. Click the **Stop** button on your pod
3. If you no longer need the volume storage, click **Terminate** to fully delete the pod

> Stopped pods still incur volume storage charges ($0.20/GB/month). Terminated pods have no ongoing charges.

---

## Troubleshooting

### `CUDA out of memory`

- Reduce `--batch-size` to `1`
- Reduce `--max-seq-length` to `1024`
- Increase `--gradient-accumulation` to compensate for smaller batch size
- Use a smaller model (e.g. `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`)

### `unsloth` installation fails

```bash
# Try installing with the CUDA-specific extras
pip install unsloth
# If that fails, install from source:
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
```

### Training is very slow

- Confirm you're on a GPU pod (not CPU): `nvidia-smi`
- Check that 4-bit quantization is enabled (default with the Unsloth bnb-4bit models)
- Use `--gradient-accumulation` instead of larger `--batch-size` to stay within VRAM

### Dataset file not found

- Verify the dataset path: `ls ./output/dataset_sharegpt.jsonl`
- If you generated the dataset locally, upload it via JupyterLab's file browser to `/workspace/firecrawl_finetuning/output/`

---

## Recommended Configurations

### Small model, small dataset (< 500 examples)

```bash
python finetuning.py \
  --model unsloth/Llama-3.2-1B-Instruct-bnb-4bit \
  --num-epochs 3 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --lora-rank 16
```

GPU: Any 16 GB+ GPU. Estimated time: 5–15 minutes.

### Medium model, medium dataset (500–5000 examples)

```bash
python finetuning.py \
  --model unsloth/Llama-3.2-3B-Instruct-bnb-4bit \
  --num-epochs 2 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --lora-rank 32 \
  --lora-alpha 32
```

GPU: 24 GB+ (A5000, A40). Estimated time: 30–90 minutes.

### Large model, large dataset (5000+ examples)

```bash
python finetuning.py \
  --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
  --num-epochs 1 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --max-seq-length 4096
```

GPU: 40 GB+ (A100). Estimated time: 1–4 hours.
