import json

def get_data_from_json(file_path):
  with open(file_path, "r", encoding="utf-8") as f:
      data = json.load(f)
  return data
