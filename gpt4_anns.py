import base64
import requests
import os
from tqdm import tqdm
import time
import json
# OpenAI API Key
api_key = "#YOUR_API_KEY#"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}",
  "OpenAI-Organization": "YOUR_ORG_NAME"
}

# Path to your image
dir_path = '#YOUR_PATH_TO_IMAGES#'
desc_json = {}
incontext_images_0 = encode_image('#/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg')
incontext_images_1 = encode_image('#/images/009.Brewer_Blackbird/Brewer_Blackbird_0026_2625.jpg')
incontext_images_2 = encode_image('#/images/073.Blue_Jay/Blue_Jay_0006_63504.jpg')
incontext_images_3 = encode_image('#/images/200.Common_Yellowthroat/Common_Yellowthroat_0011_190401.jpg')

for CLASS in tqdm(os.listdir(dir_path), desc='Querying the API...',total=len(os.listdir(dir_path))):
    cls_path = os.path.join(dir_path, CLASS)
    idx = 0
    cls_files = os.listdir(cls_path)
    while idx < len(cls_files):
        IMG = cls_files[idx]
        image_path = os.path.join(cls_path, IMG)
        base64_image = encode_image(image_path)

        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
           {"role": "system", "content": "You are a helpful assistant"},
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "What are some visual features of this bird that would help identify it?"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{incontext_images_0}"
                }
                }
            ]
            },
            {"role": "assistant", "content": "-long wingspan\n-long, hooked beak\n-black and white head\n-medium-sized bird\n"},

            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "What are some visual features of this bird that would help identify it?"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{incontext_images_1}"
                }
                }
            ]
            },
            {"role": "assistant", "content": "-yellow eyes\n-gray pointed beak\n-black glossy plumage\n-iridescent feathers on head\n-iridescent feathers on neck\n-black tail\n"},
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "What are some visual features of this bird that would help identify it?"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{incontext_images_2}"
                }
                }
            ]
            },
            {"role": "assistant", "content": "-blue crest on head\n-black collar around neck\n-white face and throat\n-black and white bars on wings and tail\n-stout, pointed beak\n-black crest on head\n"},
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "What are some visual features of this bird that would help identify it?"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{incontext_images_3}"
                }
                }
            ]
            },
            {"role": "assistant", "content": "-bright yellow throat\n-bright yellow breast\n-small bird\n-pointed beak\n-black mask around eyes\n-black and white head\n"},
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "What are some visual features of this bird that would help identify it?"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            },
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            desc_json[image_path] = response.json()['choices'][0]['message']['content']
            idx += 1
            #store the descriptions in a json file
            with open('#/descriptions_all.json', 'w') as f:
                json.dump(desc_json, f)
        else:
            print(f"Failed to query the API for {image_path}")
            time.sleep(1)

with open('/#/descriptions_all.json', 'w') as f:
    json.dump(desc_json, f)

        