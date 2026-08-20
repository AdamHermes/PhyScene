import json
import argparse
from pathlib import Path

def convert_3dssg_to_physcene(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scene_ids = []
    class_labels = []
    translations = []
    sizes = []
    angles = []
    objectness = []

    for scene_id, scene_data in data.items():
        scene_ids.append(scene_id)
        if "class_labels" in scene_data:
            class_labels.append(scene_data["class_labels"])
        translations.append(scene_data["translations"])
        sizes.append(scene_data["sizes"])
        angles.append(scene_data["angles"])
        if "objectness" in scene_data:
            objectness.append(scene_data["objectness"])

    output_data = {
        "scene_ids": scene_ids,
        "translations": translations,
        "sizes": sizes,
        "angles": angles
    }

    if class_labels:
        output_data["class_labels"] = class_labels
    if objectness:
        output_data["objectness"] = objectness

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f)
    
    print(f"Successfully converted {input_path} to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 3DSSG collision input to PhyScene format")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSON file (e.g. 3dssg_collision_input.json)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSON file")
    args = parser.parse_args()

    convert_3dssg_to_physcene(args.input, args.output)
