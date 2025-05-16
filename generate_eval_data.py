import yaml
import os
import torch
import json
import gc
from utils.file_utils import get_data_from_json
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import word_check
from tqdm import tqdm

# --- Helper Functions --
upper_vowel = ["่", "้", "๊", "๋", "็", "ิ", "ี", "ึ", "ื", "ุ", "ู", "์", "ั"]

def print_poem_formatted(text):
    # Find the maximum width of the lines
    lines = text.split('\n')

    max_width = -1
    for line in lines:
        first_tab = line.split('\t')[0]
        tab_length = len(first_tab) - \
            sum(1 for char in first_tab if char in upper_vowel)
        max_width = max(max_width, tab_length)
    # Add buffer for some space
    max_width += 4
    for line in lines:
        tabs = line.split('\t')
        for tab in tabs:
            vowels_count = sum(1 for char in tab if char in upper_vowel)
            char_length = len(tab) - vowels_count

            print(tab, end='')
            for _ in range(char_length, max_width):
                print(' ', end='')
        print()

def generate_message(user_prompt: str, system_prompt: str):
    """Creates a message list for instruct models."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def generate_response(model, tokenizer, messages: list, model_type: str, inference_params: dict):
    """Generates a response from the model."""
    if model_type == "instruct_model":
        # For instruct models, messages are expected to be a list of conversation dicts
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (model_inputs != tokenizer.pad_token_id).long().to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                model_inputs,
                attention_mask=attention_mask,
                max_new_tokens=inference_params.get("max_new_tokens", 512),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=inference_params.get("temperature", 0.9),
                top_p=inference_params.get("top_p", 0.9),
                top_k=inference_params.get("top_k", 64)
            )
        
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Extract only the assistant's response
        normalized_outputs = []
        for output in decoded_outputs:
            assistant_marker = "assistant\n\n" # Default marker
            # Check for common variations if the default isn't found
            if assistant_marker not in output:
                if "assistant\n" in output: # Llama-3.1 style
                    assistant_marker = "assistant\n"
                elif "<|start_header_id|>assistant<|end_header_id|>\n\n" in output: # Llama-3 style
                     assistant_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"

            if assistant_marker in output:
                normalized_outputs.append(output[output.find(assistant_marker) + len(assistant_marker):])
            else:
                # If no clear assistant marker, try to find the last part after user prompt
                # This is a fallback and might need adjustment based on actual model outputs
                last_user_message_content = ""
                for msg_dict_list in messages: # messages is a list of lists of dicts for batch
                    for msg_dict in msg_dict_list:
                        if msg_dict["role"] == "user":
                            last_user_message_content = msg_dict["content"]
                
                if last_user_message_content and last_user_message_content in output:
                     normalized_outputs.append(output[output.rfind(last_user_message_content) + len(last_user_message_content):].strip())
                else:
                    normalized_outputs.append(output) # Fallback to full output if extraction fails

        return normalized_outputs

    elif model_type == "base_model":
        # For base models, messages are expected to be a list of prompt strings
        model_inputs = tokenizer(
            messages,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,  # Explicitly pass attention mask
                max_new_tokens=inference_params.get("max_new_tokens", 512),
                temperature=inference_params.get("temperature", 0.9),
                top_p=inference_params.get("top_p", 0.9),
                top_k=inference_params.get("top_k", 64),
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # For base models, the output is the continuation of the prompt
        # We need to remove the prompt part from the generated text
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_outputs
    else:
        raise ValueError("Invalid model type. Choose 'instruct_model' or 'base_model'.")

# --- Main Execution ---
if __name__ == "__main__":
    # Default system prompt for instruct models
    DEFAULT_SYSTEM_PROMPT = "You are an expert Thai poet specializing in `กลอนแปด` (Eight-syllable verse) poetry. When a user provides a prompt, respond ONLY with a Thai poem (2-4 stanzas) that addresses their request, without any explanations or commentary. Your poem must strictly follow traditional กลอนแปด structure and rhyming patterns. Include phonetic rhyming tags for all rhyming words using the format: `<r>[vowel][ending consonant]word</r>` (examples: `<r>[a][w]เขา</r>`, `<r>[o][k]นก</r>`, `<r>[a]ผา</r>`, `<r>[i]ศรี</r>`). These tags should mark all external and internal rhymes according to proper กลอนแปด structure. Create vivid, culturally appropriate poetry that demonstrates mastery of Thai prosody while faithfully addressing the user's requested theme or scenario."

    # Check eval_results directory
    if not os.path.exists("eval_results"):
        os.makedirs("eval_results")

    with open("eval_intern_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for run_config in tqdm(config.get("evaluation_runs", []), desc="Overall Evaluation Progress"):
        run_name = run_config.get("name", "unnamed_eval_run")
        print(f"\n--- Starting Evaluation Run: {run_name} ---")
        model_config = run_config.get("model", {})
        model_id: str = model_config.get("id")
        base_model_id = model_config.get("base_model_id")
        model_type = model_config.get("type")

        data_config = run_config.get("data", {})
        test_inputs_file = data_config.get("test_inputs_file")
        phonetic_token_file = data_config.get("phonetic_token_file")

        inference_config = run_config.get("inference", {})
        batch_size = inference_config.get("batch_size", 1)

        output_config = run_config.get("output", {})
        output_file_prefix = output_config.get("file_prefix", "eval_output")

        if not model_id or not model_type or not test_inputs_file:
            print(f"Skipping run '{run_name}' due to missing critical configuration (model_id, model_type, or test_inputs_file).")
            continue

        try:
            print(f"Loading tokenizer for: {model_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Properly handle padding tokens
            if tokenizer.pad_token is None:
                # Try to set a different token as pad_token instead of using eos_token
                if tokenizer.unk_token is not None:
                    tokenizer.pad_token = tokenizer.unk_token
                    print("Set pad_token to unk_token as pad_token was None.")
                else:
                    # Fallback to eos_token if no better alternative
                    tokenizer.pad_token = tokenizer.eos_token
                    print("Set pad_token to eos_token as it was None and no unk_token available.")

            print(f"Loading model: {model_id}")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                # torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto"
            )
            
            model.eval() # Set model to evaluation mode

            print(f"Loading test data from: {test_inputs_file}")
            test_data = get_data_from_json(test_inputs_file)[:10]
            if not isinstance(test_data, list):
                print(f"Error: Test data in {test_inputs_file} is not a list. Skipping run.")
                continue
            
            print(f"Found {len(test_data)} test inputs.")

            outputs_evaluation = []
            # new_batch_messages_for_model = []
            new_batch_prompts_raw = []
            current_index = 0
            pbar_items = tqdm(total=len(test_data), desc=f"Processing items for {run_name}")
            
            current_index_cnt_duplicate = 0
            prev_current_index = -1

            while current_index - len(new_batch_prompts_raw) < len(test_data):
                print(f"\nCurrent index: {current_index}")
                if prev_current_index == current_index:
                    current_index_cnt_duplicate += 1
                    if current_index_cnt_duplicate >= 5:
                        break
                    
                end_index = min(current_index + min(batch_size, len(test_data)) - len(new_batch_prompts_raw), len(test_data))
                batch_prompts_raw = test_data[current_index:end_index]

                if new_batch_prompts_raw:
                    batch_prompts_raw.extend(new_batch_prompts_raw)
                new_batch_prompts_raw = []

                if len(batch_prompts_raw) > batch_size:
                    raise ValueError(f"Batch size exceeded: {len(batch_prompts_raw)} > {batch_size}")
                
                batch_messages_for_model = []
                if model_type == "instruct_model":
                    system_prompt_for_run = model_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
                    for item in batch_prompts_raw:
                        if isinstance(item, dict) and "user" in item: # Expecting list of dicts like {"user": "prompt"}
                            batch_messages_for_model.append(generate_message(item["user"], system_prompt_for_run))
                        elif isinstance(item, str): # Fallback if it's just a list of strings
                             batch_messages_for_model.append(generate_message(item, system_prompt_for_run))
                        else:
                            print(f"Warning: Skipping invalid instruct data item: {item}")
                elif model_type == "base_model":
                    for item in batch_prompts_raw:
                        if isinstance(item, str):
                            batch_messages_for_model.append(item)
                        else:
                             print(f"Warning: Skipping invalid base model data item: {item}")
                             
                if not batch_messages_for_model:
                    # Ensure progress even if all items in batch are invalid
                    current_index += len(batch_prompts_raw)
                    # If the raw batch was empty (e.g. end of list)
                    if not batch_prompts_raw:
                        break
                    continue


                print(f"\nProcessing batch {current_index // batch_size + 1} (Size: {len(batch_messages_for_model)})...")
                
                generated_texts = generate_response(model, tokenizer, batch_messages_for_model, model_type, inference_config)

                for i, gen_text in enumerate(generated_texts):
                    original_prompt_data = batch_prompts_raw[i]
                    user_prompt_display = ""
                    if model_type == "instruct_model":
                        if isinstance(original_prompt_data, dict) and "user" in original_prompt_data:
                            user_prompt_display = original_prompt_data["user"]
                        elif isinstance(original_prompt_data, str): # Fallback
                            user_prompt_display = original_prompt_data
                    elif model_type == "base_model":
                         user_prompt_display = original_prompt_data if isinstance(original_prompt_data, str) else str(original_prompt_data)
                    
                    try:
                        gen_text_waks = word_check.format_str_waks(gen_text)
                        is_non_thai_content = word_check.detect_non_thai_content(gen_text)
                        # klon_vow_mat, klon_th, is_word_fail = word_check.format_waks_syl(gen_text_waks)

                        if len(gen_text_waks) < 8:
                            raise ValueError("WakNumberFail")

                        if is_non_thai_content:
                            raise ValueError("NonThaiContent")

                        # if is_word_fail:
                        #     raise ValueError("WordFail")
                        
                        # for wak in klon_th:
                        #     if len(wak) < 5 or len(wak) > 10:
                        #         raise ValueError("LengthFail")
                        
                        outputs_evaluation.append({
                            "input_prompt": user_prompt_display,
                            "generated_output": gen_text,
                            "model_id": model_id,
                            "run_name": run_name
                        })
                        pbar_items.update(1)
                    except Exception as e:
                        print(f"Output validation failed: {e}. Re-trying this prompt.")
                        print(f"\nInput Prompt: {user_prompt_display}")
                        print("Generated Output:")
                        print_poem_formatted(gen_text)
                        print(f"Raw conversation prompt: {batch_messages_for_model[i]}")
                        print(f"Gen Text: {gen_text}")
                        new_batch_prompts_raw.append(batch_prompts_raw[i])
                        continue
                
                prev_current_index = current_index
                current_index = end_index

            pbar_items.close()
            output_filename = f"eval_results/{output_file_prefix}_{run_name.replace(' ', '_')}.json"
            with open(output_filename, "w", encoding="utf-8") as f_out:
                json.dump(outputs_evaluation, f_out, ensure_ascii=False, indent=4)
            print(f"\nResults for run '{run_name}' saved to {output_filename}")

        finally:
            # Clean up model and clear cache
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Cleaned up resources for run '{run_name}'.")

    print("\n--- All evaluation runs completed. ---")