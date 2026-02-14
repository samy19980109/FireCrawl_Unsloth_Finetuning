# Fine-Tuning on RunPod — Step-by-Step Guide

Run LoRA fine-tuning on a RunPod GPU pod using a dataset you generated locally with `main.py`.

---

## Prerequisites

- A `dataset_sharegpt.jsonl` file generated locally by `main.py`
- A [RunPod](https://www.runpod.io) account with credits or a payment method
- (Optional) A [Hugging Face](https://huggingface.co) account + token to push your model to the Hub

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

   > For `unsloth/gpt-oss-20b-unsloth-bnb-4bit` (the default), a 16 GB GPU is sufficient (14 GB VRAM with QLoRA).

4. Under **Pod Template**, select the official **RunPod PyTorch** template (comes with PyTorch, CUDA, and Python pre-installed)
5. Configure storage:
   - **Container Disk**: 20 GB (minimum)
   - **Volume Disk**: 50 GB (recommended — persists across pod restarts, stores your model weights)
   - Volume mounts at `/workspace` by default
6. Click **Deploy On-Demand** (or Spot if you want cheaper rates with possible interruptions)
7. Wait 30–120 seconds for the pod to initialize

---

## Step 2 — Connect to Your Pod

### Option A — JupyterLab (easiest)

1. Once the pod is running, click **Connect**
2. Click **Connect to Jupyter Lab** (opens in your browser on port 8888)
3. Open a **Terminal** from the JupyterLab launcher

### Option B — SSH

1. Click **Connect** on your pod
2. Copy the SSH command shown (looks like `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`)
3. Run it in your local terminal

---

## Step 3 — Clone the Repo and Install Dependencies

```bash
cd /workspace
git clone <your-repo-url> firecrawl_finetuning
cd firecrawl_finetuning
pip install -r requirements.txt
```

> `unsloth` will install its optimized kernels for your specific GPU automatically. This may take a few minutes on first install.

---

## Step 4 — Upload Your Dataset

Upload the `dataset_sharegpt.jsonl` you generated locally on your machine.

### Option A — JupyterLab file browser

1. In JupyterLab, navigate to `/workspace/firecrawl_finetuning/output/`
2. Create the `output` folder if it doesn't exist
3. Drag and drop your `dataset_sharegpt.jsonl` file into it

### Option B — SCP from your local machine

```bash
# Run this on your LOCAL machine (not the pod)
scp -P <port> ./output/dataset_sharegpt.jsonl root@<pod-ip>:/workspace/firecrawl_finetuning/output/
```

### Option C — runpodctl

```bash
# Install runpodctl locally, then:
runpodctl send ./output/dataset_sharegpt.jsonl
# Copy the receive command it prints, then run it on the pod
```

Verify the upload on the pod:

```bash
wc -l ./output/dataset_sharegpt.jsonl
head -1 ./output/dataset_sharegpt.jsonl | python -m json.tool
```

---

## Step 5 — Verify GPU Access

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:

```
CUDA available: True
GPU: NVIDIA A40
```

---

## Step 6 — Run Fine-Tuning

### Quick test (10 steps)

Start with a short run to make sure everything works:

```bash
python finetuning.py \
  --dataset ./output/dataset_sharegpt.jsonl \
  --max-steps 10
```

### Full training

Once the test passes, run a full training:

```bash
python finetuning.py \
  --dataset ./output/dataset_sharegpt.jsonl \
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
| `--model` | `unsloth/gpt-oss-20b-unsloth-bnb-4bit` | Base model (HF or Unsloth ID) |
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

## Step 7 — Verify the Output

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

## Step 8 — Download Your Model

### Option A — Push to Hugging Face Hub

```bash
huggingface-cli login

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

1. Navigate to `finetuned_model/` in the JupyterLab file browser
2. Right-click files and select **Download**

### Option D — SCP to your local machine

```bash
# Run on your LOCAL machine
scp -P <port> root@<pod-ip>:/workspace/firecrawl_finetuning/finetuned_model/* ./my-local-model/
```

---

## Step 9 — Stop Your Pod

Once you have your model, **stop the pod** to avoid charges:

1. Go to [RunPod Console > Pods](https://www.runpod.io/console/pods)
2. Click **Stop** on your pod
3. If you no longer need the volume, click **Terminate** to fully delete it

> Stopped pods still incur volume storage charges ($0.20/GB/month). Terminated pods have no ongoing charges.

---

## Troubleshooting

### `CUDA out of memory`

- Reduce `--batch-size` to `1`
- Reduce `--max-seq-length` to `1024`
- Increase `--gradient-accumulation` to compensate for smaller batch size
- Use a smaller model (e.g. `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` at ~4 GB VRAM)

### `unsloth` installation fails

```bash
pip install unsloth
# If that fails, install from source:
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
```

### Training is very slow

- Confirm you're on a GPU pod (not CPU): `nvidia-smi`
- Check that 4-bit quantization is enabled (default with the Unsloth bnb-4bit models)
- Use `--gradient-accumulation` instead of larger `--batch-size` to stay within VRAM

---

## Recommended Configurations

### Default — gpt-oss-20b (< 5000 examples)

```bash
python finetuning.py \
  --num-epochs 3 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --lora-rank 16
```

GPU: Any 16 GB+ GPU (14 GB VRAM with QLoRA). Training time: ~10–30 minutes.

### Larger — gpt-oss-120b (5000+ examples)

```bash
python finetuning.py \
  --model unsloth/gpt-oss-120b-unsloth-bnb-4bit \
  --num-epochs 1 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --lora-rank 32 \
  --lora-alpha 32
```

GPU: 80 GB (H100/A100 80GB). Training time: ~1–4 hours.

### Alternative — Llama 4 Scout (long context, 5000+ examples)

```bash
python finetuning.py \
  --model unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit \
  --num-epochs 1 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --max-seq-length 4096
```

GPU: 80 GB (H100/A100 80GB, ~71 GB VRAM). Training time: ~1–4 hours.
