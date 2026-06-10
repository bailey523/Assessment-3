# AI Text City — Process Summary
### Bailey Cooper

---

## Overview

A fine-tuned AI language model trained on personal writing, integrated into a 3D isometric city where building surfaces are covered in generated prose.

---

## Tools Used

- **Ollama** — run AI models locally on your Mac
- **Open WebUI** — chat interface for local models (runs via Docker)
- **Docker Desktop** — container system that runs Open WebUI
- **Google Colab** — free cloud GPU for training
- **Unsloth** — efficient fine-tuning library
- **Hugging Face** — cloud storage for the trained model
- **GitHub** — hosting the final HTML file

---

## Process

### 1. Local Setup
- Installed Ollama from ollama.com
- Installed Docker Desktop
- Ran Open WebUI via Docker on port 3000:8080
- Accessed chat interface at `http://localhost:3000`

### 2. Preparing Training Data
- Exported personal writing from Pages as `.txt` files
- Collected into a folder — 10 files, ~90,000 words
- Covers fiction, non-fiction, poetry, and critical writing

### 3. Training the Model (Google Colab)
- Opened Google Colab and set runtime to T4 GPU
- Installed Unsloth
- Loaded Meta LLaMA 3.1 8B as the base model
- Applied LoRA fine-tuning (trains ~0.5% of parameters efficiently)
- Mounted Google Drive and loaded writing files
- Formatted text into training chunks
- Trained for 3 epochs — loss reduced from ~3.5 to ~2.3
- Saved and converted model to GGUF format
- Uploaded to Hugging Face

### 4. Running the Model Locally
- Pulled model from Hugging Face into Ollama
- Created a Modelfile with stop tokens and generation settings
- Registered as `ai-writing-model` in Ollama

### 5. AI Text City
- Built an isometric 3D city in HTML/Canvas/JavaScript
- Generated 8 prose passages using the fine-tuned model
- Passages embedded into the HTML file
- Building surfaces covered in the generated text
- Deployed to GitHub as a standalone file — no dependencies

---

## Training Code

### Step 1 — Install Unsloth
```python
!pip install unsloth
```

### Step 2 — Load base model
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = 2048,
    load_in_4bit = True,
)
```

### Step 3 — Apply LoRA
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

### Step 4 — Mount Google Drive and load writing
```python
from google.colab import drive
drive.mount('/content/drive')

import os

texts = []
folder = '/content/drive/MyDrive/AI Writing'

for filename in os.listdir(folder):
    if filename.endswith('.txt'):
        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
            texts.append(f.read())

print(f"Loaded {len(texts)} files")
```

### Step 5 — Format training data
```python
from datasets import Dataset

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

all_chunks = []
for text in texts:
    all_chunks.extend(chunk_text(text))

def formatting_func(examples):
    return {"text": [f"<|im_start|>user\nWrite in my style.<|im_end|>\n<|im_start|>assistant\n{chunk}<|im_end|>\n"
                     for chunk in examples["text"]]}

dataset = Dataset.from_dict({"text": all_chunks})
dataset = dataset.map(formatting_func, batched=True)

print(f"Total training examples: {len(dataset)}")
```

### Step 6 — Set up trainer
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = True,
        logging_steps = 10,
        output_dir = "outputs",
        save_strategy = "no",
    ),
)
print("Trainer ready!")
```

### Step 7 — Train
```python
trainer.train()
```

### Step 8 — Log in to Hugging Face and upload
```python
from huggingface_hub import login
login(token="hf_YOUR_TOKEN_HERE")

model.push_to_hub_gguf(
    "YOUR_USERNAME/ai-writing-model",
    tokenizer,
    quantization_method = "q2_k"
)
print("Done!")
```

### Step 9 — Pull into Ollama (Terminal on Mac)
```bash
ollama pull hf.co/YOUR_USERNAME/ai-writing-model
```

### Step 10 — Create Modelfile and register
```bash
python3 -c "open('Modelfile', 'w').write('FROM hf.co/YOUR_USERNAME/ai-writing-model\nPARAMETER stop \"<|im_start|>\"\nPARAMETER num_predict 400\nPARAMETER temperature 0.7\nPARAMETER top_p 0.9\nPARAMETER repeat_penalty 1.1\n')"

ollama create ai-writing-model -f Modelfile
```

### Step 11 — Run with CORS enabled (for browser access)
```bash
OLLAMA_ORIGINS="*" ollama serve
```
```bash
ollama run ai-writing-model
```

---

## Model Details

- **Base model:** Meta LLaMA 3.1 8B
- **Training method:** LoRA (Low-Rank Adaptation) via Unsloth
- **Trainable parameters:** ~42 million of 8 billion (0.52%)
- **Training data:** ~90,000 words of personal writing
- **Training time:** ~12 minutes on Colab T4 GPU
- **Final loss:** ~2.38
- **Model hosted at:** huggingface.co/Bailey1523/ai-writing-model-v3
