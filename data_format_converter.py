import os
from utils.data_processing import load_dataset, convert_to_pretrain_format, convert_to_sft_format, save_data_to_json

# Define paths
non_instruct_base_path = os.path.join(os.getcwd(), "dataset")
instruct_base_path = os.path.join(os.getcwd(), "Instruct_Dataset")

# Define dataset file mappings
non_instruct_files = {
    "tagged_train": "training_data.json",
    "tagged_valid": "valid_data.json",
    "tagged_test_input": "test_inputs_phonetic.json",
    "untagged_test_input": "test_inputs.json"
}

instruct_files = {
    "tagged_train": "klon_tagged_train.json",
    "tagged_valid": "klon_tagged_valid.json",
    "phonetic_nmt_train": "phonetic_nmt_train.json",
    "phonetic_nmt_valid": "phonetic_nmt_valid.json",
    "rhyme_syllables_train": "rhyme_syllables_train.json",
    "rhyme_syllables_valid": "rhyme_syllables_valid.json",
    "untagged_train": "klon_untagged_train.json",
    "untagged_valid": "klon_untagged_valid.json"
}

# Load datasets
print("Loading non-instruct datasets...")
non_instruct_data = load_dataset(non_instruct_base_path, non_instruct_files)

print("Loading instruct datasets...")
instruct_data = load_dataset(instruct_base_path, instruct_files)

# Process datasets for pretraining
print("Processing pretraining datasets...")
pretrain_datasets = {
    "non_instruct_tagged_train": convert_to_pretrain_format(non_instruct_data["tagged_train"]),
    "non_instruct_tagged_valid": convert_to_pretrain_format(non_instruct_data["tagged_valid"])
}

# Process datasets for fine-tuning with ShareGPT format
print("Processing SFT datasets...")
sft_datasets = {
    "instruct_untagged_train": convert_to_sft_format(
        instruct_data["untagged_train"], format_type="sharegpt"
    ),
    "instruct_untagged_valid": convert_to_sft_format(
        instruct_data["untagged_valid"], format_type="sharegpt"
    ),
    "instruct_tagged_train": convert_to_sft_format(
        instruct_data["tagged_train"], format_type="sharegpt"
    ),
    "instruct_tagged_valid": convert_to_sft_format(
        instruct_data["tagged_valid"], format_type="sharegpt"
    ),
    "instruct_phonetic_nmt_train": convert_to_sft_format(
        instruct_data["phonetic_nmt_train"], format_type="sharegpt"
    ),
    "instruct_phonetic_nmt_valid": convert_to_sft_format(
        instruct_data["phonetic_nmt_valid"], format_type="sharegpt"
    ),
    "instruct_rhyme_syllables_train": convert_to_sft_format(
        instruct_data["rhyme_syllables_train"], format_type="sharegpt"
    ),
    "instruct_rhyme_syllables_valid": convert_to_sft_format(
        instruct_data["rhyme_syllables_valid"], format_type="sharegpt"
    ),
}

# Save processed datasets (optional)
output_dir = "processed_datasets"
os.makedirs(output_dir, exist_ok=True)

print("Saving processed datasets...")
# Save pretrain datasets
for name, data in pretrain_datasets.items():
    save_data_to_json(data, os.path.join(output_dir, f"{name}_pretrain.json"))

# Save SFT dataset
for name, data in sft_datasets.items():
    save_data_to_json(data, os.path.join(output_dir, f"{name}_sft.json"))

print("Dataset processing complete!")

# Print some statistics
print("\nDataset Statistics:")
print("-" * 40)
print("Non-instruct datasets:")
for key, value in non_instruct_data.items():
    print(f"  {key}: {len(value)} items")

print("\nInstruct datasets:")
for key, value in instruct_data.items():
    print(f"  {key}: {len(value)} items")

print("\nProcessed datasets:")
for key, value in pretrain_datasets.items():
    print(f"  {key}: {len(value)} items")
for key, value in sft_datasets.items():
    print(f"  {key}: {len(value)} items") 