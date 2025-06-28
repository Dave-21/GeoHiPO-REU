"""By Parth Kulkarni"""
# import requests
# import copy
# import torch
import json
import time
import os
from tqdm import tqdm
import argparse
# import numpy as np
# from PIL import Image
import warnings
from glob import glob
from accelerate import Accelerator
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference example for Qwen-2.5-VL")
    parser.add_argument("--video_dir", type=str, default="/home/al209167/datasets/im2gps3ktest", help="Path to the directory containing the images")
    parser.add_argument("--model_base", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Name of the pretrained model")
    parser.add_argument("--outpath", type=str, default="im2gps3k_city_country_predictions.json", help="Path for the output JSON file")
    parser.add_argument("--device", type=str, default="cuda", help="Cuda or CPU")
    parser.add_argument("--pretrained", type=str, default=None)
    return parser.parse_args()

args = parse_args()
pretrained = args.pretrained
device = args.device
model_base=args.model_base

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_base)
model.eval()


def fixed_prompt():
    return 'You are an expert in geography and tourism. You possess extensive knowledge of geography, terrain, landscapes, flora, fauna, infrastructure, and other natural or man-made features that help determine a location from images or descriptions. Additionally, you are well-versed in tourism-related information, including amenities such as hotels, restaurants, attractions, and services available in various locations.'

# Function to process a single video and question
def process_video_question(image_path, question):
    prompt = fixed_prompt()
    user_prompt = prompt + f"\nQuestion: {question}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    predicted_answer = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return predicted_answer

# Function to load and process the JSON file
def process_json_file(video_dir, full_output_path):

    results = []
    start_time = time.time()

    if "yfcc25600" in video_dir:
        files = glob(video_dir + "/**/**/*.jpg")
    else:
        files = glob(video_dir + "/*.jpg")


    for i, file in enumerate(tqdm(files, desc="Processing videos", unit="video")):

        question = "<image>\nWhere is this image taken? Respond with only the city and country."

        if "yfcc25600" in video_dir:
            video_filename = "/".join(files[0].split("/")[-3:])
        else:
            video_filename = file.split("/")[-1]
        # video_path = os.path.join(video_dir, video_filename)

        if not os.path.exists(file):
            model_answer = f"Error: Video file not found at {file}"
        else:
            model_answer = process_video_question(file, question)

        results.append({
            'filename': video_filename,
            'question': question,
            'prediction': model_answer

        })

        if i % 100 == 0:
            with open(full_output_path, 'w') as file:
                json.dump(results, file, indent=4)


    end_time = time.time()
    total_time = end_time - start_time

    return results, total_time



# Main execution
if __name__ == "__main__":
    video_dir = args.video_dir
    model_name = "Qwen2.5-VL-7B-Instruct" #args.pretrained.split('/')[-1]
    outpath = args.outpath

    print("Starting video processing...")

    root = "city_predictions"
    # root = "img2gps_city_prediction"

    save_dir = f"{root}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    full_output_path = os.path.join(save_dir,outpath)

    results, total_time = process_json_file(video_dir, full_output_path)

    # Save results to JSON file
    with open(full_output_path, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"\nProcessing completed in {total_time:.2f} seconds.")
    print(f"Results saved to {full_output_path}")

    # Print summary
    print(f"\nProcessed {len(results)} Images.")
    print(f"Average time per Image: {total_time / len(results):.2f} seconds.")
