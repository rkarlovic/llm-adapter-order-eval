import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer

# 1. Učitavanje i priprema podataka
# ---------------------------------------------------------
filename = "Retail_Dataset_synthetic.csv"
df = pd.read_csv(filename)

# (Opcionalno) Filtriranje loših podataka - ovdje samo uzimamo zadnjih 5000 redaka 
# jer smo vidjeli da je početak dataseta neispravan, a kraj ispravan.
# U praksi bi trebao napraviti bolju validaciju.
df = df.tail(5000).reset_index(drop=True)

# Formatiranje podataka u instrukcijski format
# Model učimo da na temelju unosa korisnika generira JSON odgovor
def format_instruction(row):
    return f"""### Instruction:
Analyze the user request and extract action, product, and quantity.

### Input:
{row['user_input']}

### Response:
{{
    "action": "{row['action']}",
    "product": "{row['product']}",
    "quantity": {row['quantity']}
}}"""

df['text'] = df.apply(format_instruction, axis=1)

# Pretvaranje u Hugging Face Dataset objekt
dataset = Dataset.from_pandas(df[['text']])

# 2. Konfiguracija Modela (Quantization & Base Model)
# ---------------------------------------------------------
# Koristimo npr. TinyLlama ili Mistral-7B kao bazni model. 
# Zamijeni s "meta-llama/Llama-2-7b-hf" ili "mistralai/Mistral-7B-v0.1" za jače rezultate.
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

# QLoRA konfiguracija (4-bitno učitavanje radi uštede memorije)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Fix za padding

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Priprema modela za k-bit trening
model = prepare_model_for_kbit_training(model)

# 3. LoRA Konfiguracija
# ---------------------------------------------------------
peft_config = LoraConfig(
    r=16,       # Rank - dimenzija matrice adaptera (8, 16, 32, 64)
    lora_alpha=32, # Skaliranje (obično 2x rank)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"] # Moduli na koje lijepimo adaptere
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # Ispis koliko parametara zapravo treniramo

# 4. Trening (SFTTrainer)
# ---------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./lora-retail-adapter",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    fp16=True,             # Koristi mixed precision
    logging_steps=50,
    save_strategy="epoch",
    optim="paged_adamw_32bit" # Optimizator za manju potrošnju memorije
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

print("Počinjem trening...")
trainer.train()

# 5. Spremanje Adaptera
# ---------------------------------------------------------
new_model_name = "retail-adapter-v1"
trainer.model.save_pretrained(new_model_name)
print(f"LoRA adapter spremljen u mapu: {new_model_name}")