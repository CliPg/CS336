import requests

tiny_stories_train_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
tiny_stories_valid_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

tiny_stories_train_path = "TinyStoriesV2-GPT4-train.txt"
tiny_stories_valid_path = "TinyStoriesV2-GPT4-valid.txt"

response = requests.get(tiny_stories_valid_url, stream=True)

with open(tiny_stories_valid_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

