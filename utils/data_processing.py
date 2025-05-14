import os
import json

def get_data_from_json(file_path):
    """Load data from a JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_data_to_json(data, file_path):
    """Save data to a JSON file"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_dataset(base_path, file_names):
    """Load multiple datasets from a base path"""
    results = {}
    for key, file_name in file_names.items():
        file_path = os.path.join(base_path, file_name)
        results[key] = get_data_from_json(file_path)
    return results

def convert_to_pretrain_format(data, format_type="alpaca"):
    """Convert data to pretrain format"""
    if format_type == "alpaca":
        return [{"text": item} for item in data]
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

def convert_to_sft_format(data, format_type="sharegpt"):
    """Convert data to SFT format"""
    formatted_data = []
    
    if format_type == "sharegpt":
        for item in data:
            conversation = [
                {"from": "human", "content": item["user"]},
                {"from": "model", "content": item["answer"]}
            ]
            formatted_data.append({
                "conversations": conversation,
                "system": item["context"]
            })
            
    elif format_type == "alpaca":
        for item in data:
            formatted_data.append({
                "instruction": item["context"],
                "input": item["user"],
                "output": item["answer"],
                "system": item["context"]
            })
            
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
        
    return formatted_data 