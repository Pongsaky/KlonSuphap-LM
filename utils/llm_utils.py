import torch
import gc
from collections import defaultdict
import re

def extract_phonetic_combinations(training_data, tokenizer, model_type="gemma|llama"):
    assert(model_type in ["gemma", "llama"])
    # Define regex pattern to match <r>[X][Y]text</r> patterns
    pattern = r'<r>(.+?)</r>'

    tag_dict = defaultdict(set)

    # Process each text in the training data
    for text in training_data:
        # Find all matches of the pattern in the text
        matches = re.findall(pattern, text)

        for item in matches:
            # Extract all tags (e.g., [a], [w])
            tags = re.findall(r'\[.*?\]', item)
            # Extract the word by removing tags
            word = re.sub(r'\[.*?\]', '', item).strip()
            # Add the word to each tag's set
            for tag in tags:
                if "<r>" in word:
                    print(
                        f"Warning: <r> tag found in word '{word}'. Skipping this entry.")
                    continue
                tag_dict[tag].add(word)

    new_tag_dict = defaultdict(set)
    for key, value in tag_dict.items():
        for word in value:
            if model_type == "gemma":
                tokenized_words = tokenizer.tokenizer.encode(word, add_special_tokens=False)
            else:
                tokenized_words = tokenizer.encode(word, add_special_tokens=False)
            # tokenized_words = tokenizer.encode(word, add_special_tokens=False)
            if "<r>" in tokenized_words:
                print(f"Value: {value}")
                print(f"Tokenized word: {word}")
                print(f"Tokenized word: {tokenized_words}")
            for tokenized_word in tokenized_words:
                new_tag_dict[key].add(tokenized_word)

    return new_tag_dict

# Custom function for Gemma-3 token adding
def mean_surround_new_tag_token(model, tag_dict):
    # Calculate the mean of surrounding token embeddings
    embedding_matrix = model.get_input_embeddings().weight.clone()
    lm_head_matrix = model.get_output_embeddings().weight.clone()

    tag_embedding_dict = {}
    tag_lm_head_dict = {}

    for tag, ids_values in tag_dict.items():
        # properly accumulate all the embeddings
        tag_embedding = torch.zeros_like(embedding_matrix[0])
        tag_lm_head = torch.zeros_like(lm_head_matrix[0])
        for idx in ids_values:
            tag_embedding += embedding_matrix[idx]
            tag_lm_head += lm_head_matrix[idx]
        tag_embedding_dict[tag] = tag_embedding / len(ids_values)
        tag_lm_head_dict[tag] = tag_lm_head / len(ids_values)

    return tag_embedding_dict, tag_lm_head_dict


def add_new_tokens(
    model,
    tokenizer,
    tag_dict,
    new_tokens=[],
    model_type="gemma|llama"
):
    """
    Smartly resizes the tokenizer and adds new tokens to the model.
    We also disregard untrained tokens by removing them from the mean calculation.
    """
    # All Unsloth Zoo code licensed under LGPLv3
    assert (isinstance(new_tokens, (list, tuple)))
    assert (len(new_tokens) > 0)
    assert (len(tag_dict) > 0)
    assert (isinstance(tag_dict, dict))
    assert(model_type in ["gemma", "llama"])

    # Check if tokens already exist
    if model_type == "gemma":
        overlapping_tokens = set(new_tokens) & set(tokenizer.tokenizer.vocab.keys())
    else:
        overlapping_tokens = set(new_tokens) & set(tokenizer.vocab.keys())
        
    if len(overlapping_tokens) != 0:
        print(
            f"Unsloth: You're adding new_tokens = {new_tokens}\n"
            f"There are tokens which are overlapping = {list(overlapping_tokens)}\n"
            f"We shall safely ignore these overlapping tokens."
        )
        new_tokens = [x for x in new_tokens if x not in overlapping_tokens]


    # Weirdly be careful reserved tokens can pop out
    tag_embedding_dict, tag_lm_head_dict = mean_surround_new_tag_token(
        model, tag_dict)

    # Get old lengths
    old_input_embedding = model.get_input_embeddings().weight
    old_output_embedding = model.get_output_embeddings().weight
    old_input_length = old_input_embedding.shape[0]
    old_output_length = old_output_embedding.shape[0]
    if model_type == "gemma":
        old_config_size = model.config.text_config.vocab_size
    else:
        old_config_size = model.config.vocab_size

    # Check for tied weights as well
    is_tied = (old_input_embedding.data_ptr() == old_output_embedding.data_ptr()) \
        or (model.config.tie_word_embeddings)

    # Add tokens!
    if model_type == "gemma":
        old_length = len(tokenizer.tokenizer)
        tokenizer.tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer.tokenizer))
    else:
        old_length = len(tokenizer)
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
    # Also resizes lm_head as well!

    # the Word2Vec sum of the other vectors
    embedding_matrix = model.get_input_embeddings().weight
    lm_head_matrix = model.get_output_embeddings().weight

    # Confirm sizes are correct
    if embedding_matrix.shape[0] > (old_input_length + len(new_tokens)):
        raise RuntimeError(
            "Unsloth: Embedding matrix size did not get resized properly. Please file a bug report!"
        )
    if lm_head_matrix.shape[0] > (old_output_length + len(new_tokens)):
        raise RuntimeError(
            "Unsloth: LM Head matrix size did not get resized properly. Please file a bug report!"
        )
    if model_type == "gemma":
        if model.config.text_config.vocab_size > (old_config_size + len(new_tokens)):
            raise RuntimeError(
                "Unsloth: Model's config vocab_size did not get resized properly. Please file a bug report!"
            )
    else:
        if model.config.vocab_size > (old_config_size + len(new_tokens)):
            raise RuntimeError(
                "Unsloth: Model's config vocab_size did not get resized properly. Please file a bug report!"
            )

    key_list = list(tag_dict.keys())
    if model_type == "gemma":
        key_ids_list = [tokenizer.tokenizer.encode(word, add_special_tokens=False)[0] for word in key_list]
    else:
        key_ids_list = [tokenizer.encode(word, add_special_tokens=False)[0] for word in key_list]
    with torch.no_grad():
        for key, ids in zip(key_list, key_ids_list):
            tag_embedding = tag_embedding_dict[key]
            tag_lm_head = tag_lm_head_dict[key]
            embedding_matrix[ids] = tag_embedding
            lm_head_matrix[ids] = tag_lm_head
            pass

    # We set a flag to say we need to train embeddings
    internal_model = model
    while hasattr(internal_model, "model"):
        internal_model._need_to_train_embeddings = True
        internal_model = internal_model.model
    pass
    internal_model._need_to_train_embeddings = True
    
    # Fix up all vocab sizes
    current_model = model
    if hasattr(current_model, "model") and hasattr(current_model, "config"):
        if model_type == "gemma":
            if hasattr(current_model.config.text_config, "vocab_size"):
                current_model.config.text_config.update({"vocab_size": len(tokenizer.tokenizer)})
        else:
            if hasattr(current_model.config, "vocab_size"):
                current_model.config.update({"vocab_size": len(tokenizer)})
        current_model = current_model.model

    # Must tie lm_head and embed_tokens if they are tied!
    # Otherwise error will occur on saving models ie use save_model
    if is_tied:
        model.tie_weights()

    # Clear deleted GPU items
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
    return