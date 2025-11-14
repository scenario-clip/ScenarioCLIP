import json
import os
from suffixes import suffixes
from tqdm import tqdm

base_path = "/mnt/MIG_store/Datasets"

coco_path = os.path.join(base_path, 'coco/train2017')

data_path = os.path.join(base_path, 'ActionGenome/results_29_9')

data = {'data': []}

with open("indices3.txt") as f:
    ids = [x.removesuffix('\n') for x in f.readlines()]

for index in tqdm(ids, desc="Combining data"):
    filename = f"{index}{suffixes['img']}"
    with open(f"{data_path}/gemma_jsons/{index}{suffixes['relations_json']}") as f:
        file = json.load(f)
    action = file['response']['action']
    dense_caption = file['response']['dense caption']
    focused_regions = file['focused_regions']
    new_fr = []
    for key, value in focused_regions.items():
        new_fr.append(["sam_results/"+key.split("sam_results/")[1], value['relation']])
    with open(f"{data_path}/dino_results/{index}{suffixes['bbox_json']}") as f:
        file = json.load(f)
    object_names = [x.lower() for x in file['labels']]
    bboxes = file['boxes']
    data['data'].append({
        "id": index,
        "image_path": filename,
        "action": action.lower(),
        "dense_caption": dense_caption,
        "objects": list(zip(object_names, bboxes)),
        "relations": new_fr,
    })

with open('combined.json', 'w') as f:
    json.dump(data, f, indent=4)

actions = set()
objects = set()
relations = set()

for obj in tqdm(data['data'], desc="Collecting classes"):
    actions.add(obj['action'])
    for object in obj['objects']:
        objects.add(object[0].lower())
    for relation in obj['relations']:
        relations.add(' '.join(relation[1]).lower())

with open('classes.json', 'w') as f:
    json.dump({
        "actions": sorted(list(actions)),
        "objects": sorted(list(objects)),
        "relations": sorted(list(relations)),
    }, f, indent=4)
