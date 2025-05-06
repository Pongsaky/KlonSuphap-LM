import unsloth
import yaml
import os
import torch
import numpy as np
import transformers
import trl
import datasets
import gc
import json
import argparse

from utils import llm_utils

# Set environment variable for Unsloth
os.environ["UNSLOTH_IS_PRESENT"] = "1"

# Function to get data from JSON file
def get_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def add_conversation(conversation: list[dict], role: str, content: str):
    conversation.append({
        "role": role,
        "content": content,
    })

def formatting_conversation(examples):
    # print(examples)
    dict_convs = examples["text"]

    conversations = []
    for dict_conv in dict_convs:
        conversation = []
        add_conversation(conversation, "system", dict_conv["context"])
        add_conversation(conversation, "user", dict_conv["user"])
        add_conversation(conversation, "assistant", dict_conv["answer"])
        conversations.append(conversation)

    return {"conversations": conversations}


def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix(
       '<|begin_of_text|>') for convo in convos]
   return {"text": texts, }

# Function for instruct model inference during callbacks
def generate_response(model, tokenizer, messages, max_new_tokens=100, temperature=0.9, top_p=0.95, top_k=64):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Must add for generation
    )

    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(
            **tokenizer([text], return_tensors="pt").to("cuda"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=tokenizer.special_tokens_map["eos_token"],
            streamer=transformers.TextStreamer(tokenizer, skip_prompt=False),
        )

    # Decode and return the generated text
    return tokenizer.batch_decode(outputs)[0]

# Custom callback for text output
class TextOutputCallback(transformers.TrainerCallback):
    def __init__(self, model, tokenizer, test_inputs, every_n_steps, data_type="standard"):
        self.model = model
        self.tokenizer = tokenizer
        self.test_inputs = test_inputs
        self.every_n_steps = every_n_steps
        self.data_type = data_type
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.current_step = state.global_step
        if self.current_step % self.every_n_steps == 0:
            print(f"\n--- Step {self.current_step} ---")

            rand_idx = int(torch.rand(1) * len(self.test_inputs))
            sample_text = self.test_inputs[rand_idx]
            
            print(f"Input : {sample_text}")
            if self.data_type == "standard":
                inputs = self.tokenizer(sample_text, return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Output : {response}")
            elif self.data_type == "instruct":
                response = generate_response(self.model, self.tokenizer, sample_text[:-1])
                print(f"Output : {response}")

        return control


def add_new_tokens(model, tokenizer, new_tokens, model_type="llama", tag_dict=None):
    if model_type not in ["gemma", "llama"]:
        raise ValueError("Unsupported model type. Supported types are 'gemma' and 'llama'.")

    if tag_dict is None:
        raise ValueError("tag_dict cannot be None.")

    llm_utils.add_new_tokens(model=model, tokenizer=tokenizer, new_tokens=new_tokens, tag_dict=tag_dict, model_type=model_type)
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model based on configuration.")
    parser.add_argument("--no-confirm", action="store_true", help="Skip user confirmation before training.")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    for run_config in config["training_runs"]:
        print("-" * 50)
        print(f"Configuration Summary for Run: {run_config['name']}")
        print("-" * 50)
        
        # Set up WandB
        if run_config["training"].get("report_to") == "wandb":
            wandb_api_key = run_config["wandb"]["api_key_env_var"]
            if wandb_api_key:
                import wandb
                wandb.login(key=wandb_api_key)
                wandb.init(project=run_config["wandb"]
                           ["project"], name=run_config["name"])
            else:
                print(
                    f"Warning: WandB API key environment variable '{run_config['wandb']['api_key_env_var']}' not set. Skipping WandB logging.")

        # Model Information
        print(f"  Model Name: {run_config['model']['name']}")
        print(f"  Model Type: {run_config['model']['type']}")
        print(f"  Quantization: {run_config['model']['quantization']}")

        # LoRA Information
        lora_enabled = run_config["lora"]["enabled"]
        print(f"  LoRA Enabled: {lora_enabled}")
        if lora_enabled:
            print(f"    LoRA r: {run_config['lora']['r']}")
            print(f"    LoRA alpha: {run_config['lora']['alpha']}")
            print(f"    LoRA target_modules: {run_config['lora']['target_modules']}")
            print(f"    LoRA train_embeddings: {run_config['lora'].get('train_embeddings', False)}")

        # Data Information
        data_type = run_config["data"]["type"]
        print(f"  Data Type: {data_type}")
        if data_type == "standard":
            print(f"    Train File: {run_config['data']['train_file']}")
            print(f"    Valid File: {run_config['data']['valid_file']}")
            print(f"    Test Inputs File: {run_config['data']['test_inputs_file']}")
        elif data_type == "instruct":
            print(
                f"    Train Files: {run_config['data']['instruct_train_files']}")
            print(
                f"    Valid Files: {run_config['data']['instruct_valid_files']}")
            print(f"    Test Inputs File: {run_config['data']['test_inputs_file']}")
        print(f"    Phonetic Tokens File: {run_config['data']['phonetic_token_file']}")

        # Training Hyperparameters
        print("  Training Hyperparameters:")
        print(f"    Epochs: {run_config['training']['epochs']}")
        print(f"    Batch Size: {run_config['training']['batch_size']}")
        print(f"    Accumulation Steps: {run_config['training']['accumulation_steps']}")
        print(f"    Learning Rate: {run_config['training']['learning_rate']}")

        # Load phonetic tokens
        phonetic_tokens = get_data_from_json(run_config["data"]["phonetic_token_file"])

        # Load dataset(s)
        data_type = run_config["data"]["type"]
        if data_type == "standard":
            train_data = get_data_from_json(run_config["data"]["train_file"])
            valid_data = get_data_from_json(run_config["data"]["valid_file"])
            train_dataset = datasets.Dataset.from_dict({"text": train_data})
            valid_dataset = datasets.Dataset.from_dict({"text": valid_data})
            test_inputs = get_data_from_json(run_config["data"]["test_inputs_file"])
        elif data_type == "instruct":
            # Training file used for generating new tokens for instruct case
            train_data = get_data_from_json(run_config["data"]["train_file"])
            
            # Load instruct datasets
            instruct_train_files = run_config["data"]["instruct_train_files"]
            instruct_valid_files = run_config["data"]["instruct_valid_files"]

            train_datasets = [datasets.Dataset.from_dict({"text": get_data_from_json(f)}) for f in instruct_train_files]
            valid_datasets = [datasets.Dataset.from_dict({"text": get_data_from_json(f)}) for f in instruct_valid_files]
            train_dataset = datasets.concatenate_datasets(train_datasets)
            valid_dataset = datasets.concatenate_datasets(valid_datasets)
            # test_inputs_raw = get_data_from_json(run_config["data"]["test_inputs_file"])
            # # Assuming test_inputs_file for instruct is a list of conversation structures
            # # We need to format them into prompts for the callback
            # test_inputs = [add_conversation({"messages": item})["text"] for item in test_inputs_raw]

            # Prepare instruct datasets
            train_dataset = train_dataset.map(formatting_conversation, batched=True)
            valid_dataset = valid_dataset.map(formatting_conversation, batched=True)

        # Load model and tokenizer
        model_type = run_config["model"]["type"]
        quantization = run_config["model"]["quantization"]
        load_in_4bit = quantization == "4bit"
        load_in_8bit = quantization == "8bit"
        full_finetuning = quantization == "full"

        model, tokenizer = unsloth.FastModel.from_pretrained(
            model_name=run_config["model"]["name"],
            max_seq_length=run_config["model"]["max_seq_length"],
            dtype=None, # Auto detects
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            full_finetuning=full_finetuning,
            token=run_config["saving"]["hf_token_env_var"], # Use environment variable
        )

        
        if run_config["data"]["type"] == "instruct":
            tokenizer = unsloth.chat_templates.get_chat_template(
                tokenizer,
                chat_template=run_config["data"]["chat_template"],
            )

        if run_config["data"]["add_new_tokens"]:
            # Add new tokens
            tag_dict = llm_utils.extract_phonetic_combinations(train_data, tokenizer, model_type=model_type)
            model, tokenizer = add_new_tokens(model=model, tokenizer=tokenizer, new_tokens=phonetic_tokens, model_type=model_type, tag_dict=tag_dict)

        if run_config["data"]["type"] == "instruct":
            # Format the dataset for instruct
            train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
            valid_dataset = valid_dataset.map(formatting_prompts_func, batched=True)

        # Shuffle datasets
        train_dataset = train_dataset.shuffle(seed=run_config["training"]["seed"])
        valid_dataset = valid_dataset.shuffle(seed=run_config["training"]["seed"])

        # Apply LoRA
        if run_config["lora"]["enabled"]:
            lora_target_modules = run_config["lora"]["target_modules"]
            if run_config["lora"].get("train_embeddings", False):
                 # Ensure embed_tokens and lm_head are included if training embeddings
                 if "embed_tokens" not in lora_target_modules:
                     lora_target_modules.append("embed_tokens")
                 if "lm_head" not in lora_target_modules:
                     lora_target_modules.append("lm_head")

            model = unsloth.FastModel.get_peft_model(
                model,
                r=run_config["lora"]["r"],
                lora_alpha=run_config["lora"]["alpha"],
                lora_dropout=0, # Default dropout
                bias="none", # Default bias
                use_grad_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )

        # Calculate steps
        total_train_samples = len(train_dataset)
        batch_size = run_config["training"]["batch_size"]
        accumulation_steps = run_config["training"]["accumulation_steps"]
        epochs = run_config["training"]["epochs"]

        total_steps = (total_train_samples // (batch_size * accumulation_steps)) * epochs
        every_n_steps = total_steps // (epochs * run_config["callbacks"]["text_output"]["every_n_steps_per_epoch"])
        if every_n_steps == 0:
            every_n_steps = 1 # Ensure at least one output per epoch if dataset is small
        warmup_steps = int(0.1 * total_steps) # 10% warmup
        # Evaluate and save based on save_steps_per_epoch
        eval_steps = total_steps // (epochs * run_config["training"]["eval_times_per_epoch"])
        save_steps = int(np.ceil((total_steps / (run_config["saving"]["save_steps_per_epoch"] * epochs)) / eval_steps) * eval_steps)

        # Calculated Steps Summary
        print("  Calculated Steps:")
        print(f"    Total Steps: {total_steps}")
        print(f"    Warmup Steps: {warmup_steps}")
        print(f"    Eval Steps: {eval_steps}")
        print(f"    Save Steps: {save_steps}")
        print("-" * 50)

        # Initialize callbacks
        callbacks = []
        if run_config["callbacks"]["text_output"]["enabled"]:
            callbacks.append(TextOutputCallback(model, tokenizer, test_inputs, every_n_steps, data_type=data_type))
        if run_config["callbacks"]["early_stopping"]["enabled"]:
            callbacks.append(transformers.EarlyStoppingCallback(
                early_stopping_patience=run_config["callbacks"]["early_stopping"]["patience"],
                early_stopping_threshold=0.0, # Default threshold
            ))

        # Training arguments
        training_args = trl.SFTConfig(
            output_dir=f"./results/{run_config['name']}",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            learning_rate=float(run_config["training"]["learning_rate"]),
            optim=run_config["training"]["optim"],
            seed=run_config["training"]["seed"],
            report_to=run_config["training"].get("report_to"),
            logging_steps=run_config["training"]["logging_steps"],
            eval_steps=eval_steps,
            save_steps=save_steps,
            load_best_model_at_end=run_config["training"]["load_best_model_at_end"],
            metric_for_best_model="eval_loss", # Metric to monitor for early stopping and best model
            run_name=run_config["name"],
            weight_decay=run_config["training"]["weight_decay"],
            lr_scheduler_type=run_config["training"]["lr_scheduler_type"],
            eval_strategy="steps",
            dataset_num_proc=2,
            fp16=not unsloth.is_bf16_supported(),
            bf16=unsloth.is_bf16_supported(),
        )

        # Trainer
        if data_type == "standard":
            trainer = trl.SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                dataset_text_field="text",
                max_seq_length=run_config["model"]["max_seq_length"],
                args=training_args,
                callbacks=callbacks,
            )
        elif data_type == "instruct":
            trainer = trl.SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                dataset_text_field="text",
                max_seq_length=run_config["model"]["max_seq_length"],
                args=training_args,
                callbacks=callbacks,
            )
            
            trainer = unsloth.chat_templates.train_on_responses_only(
                trainer,
                instruction_part="<|start_header_id|>user<|end_header_id|>",
                response_part="<|start_header_id|>assistant<|end_header_id|>"
            )
            
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        # User Confirmation
        if not args.no_confirm:
            while True:
                user_input = input("Do you want to train with these settings? (yes/no): ").lower()
                if user_input in ["yes", "y"]:
                    print("Proceeding with training...")
                    break
                elif user_input in ["no", "n"]:
                    print("Skipping training for this run.")
                    break
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")
        else:
            print("Skipping user confirmation due to --no-confirm flag.")
            print("Proceeding with training...")


        # Train if confirmed or --no-confirm is present
        if args.no_confirm or user_input in ["yes", "y"]:
            trainer.train()

            # Save and push to hub
            if run_config["saving"]["saving_local"]:
                model.save_pretrained(run_config["name"])
                tokenizer.save_pretrained(run_config["name"])
            
            if run_config["saving"]["push_to_hub"]: 
                model.push_to_hub(
                    repo_id=f"{run_config['saving']['hub_model_id_prefix']}/{run_config['name']}",
                    token=f"{run_config['saving']['hf_token_env_var']}",
                )
                tokenizer.push_to_hub(
                    repo_id=f"{run_config['saving']['hub_model_id_prefix']}/{run_config['name']}",
                    token=f"{run_config['saving']['hf_token_env_var']}",
                )

            # End WandB run
            if run_config["training"].get("report_to") == "wandb":
                wandb.finish()

            # Memory cleanup
            del trainer
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

            print(f"Finished training run: {run_config['name']}")
        else:
             # Memory cleanup if training is skipped
            del trainer
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Skipped training run: {run_config['name']}")