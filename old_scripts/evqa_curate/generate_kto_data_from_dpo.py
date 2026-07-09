import json
import os
from pathlib import Path

def convert_dpo_to_kto(input_file):
    """
    Convert DPO format data to KTO format.
    For each DPO example, creates two KTO examples:
    1. One with the chosen response (kto_tag=true)
    2. One with the rejected response (kto_tag=false)
    """
    # Read input DPO data
    with open(input_file, 'r') as f:
        dpo_data = json.load(f)

    kto_data = []
    
    for item in dpo_data:
        # Get common fields
        human_msg = item['conversations'][0]
        img_path = item['images'][0]
        
        # Create KTO example with chosen response (positive)
        chosen_example = {
            "conversations": [
                human_msg,
                {
                    "from": "gpt",
                    "value": item['chosen']['value']
                }
            ],
            "images": [img_path],
            "kto_tag": True
        }
        
        # Create KTO example with rejected response (negative)
        rejected_example = {
            "conversations": [
                human_msg,
                {
                    "from": "gpt", 
                    "value": item['rejected']['value']
                }
            ],
            "images": [img_path],
            "kto_tag": False
        }
        
        kto_data.extend([chosen_example, rejected_example])

    return kto_data

def main():
    # Input/output paths
    input_file = "third_party/LLaMAFactory/data/jinghong_chen/evqa/7B-rag5-answer-dpo_max=4096/train_sharegpt.json"
    output_dir = "third_party/LLaMAFactory/data/jinghong_chen/evqa/7B-rag5-answer-kto_max=4096"
    output_file = os.path.join(output_dir, "train_sharegpt.json")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convert data
    kto_data = convert_dpo_to_kto(input_file)
    
    # Save output
    with open(output_file, 'w') as f:
        json.dump(kto_data, f, indent=2)
    
    print(f"Converted {len(kto_data)} KTO examples")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()
