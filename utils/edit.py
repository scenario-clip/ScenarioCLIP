import json
from suffixes import suffixes
from tqdm import tqdm

original_json = "/Datasets/action-genome/results_9_10/new_metadata_63k.json"
base_path = "/Datasets/coco/train2017"
gen_path = "/Datasets/action-genome/results_9_10"

with open(original_json) as f:
    listofstuff = json.load(f)['data']

for i, piece in tqdm(enumerate(listofstuff)):
    index = piece["id"]
    file = f"{gen_path}/gemma_jsons/{index}{suffixes['relations_json']}"
    piece["image_path"] = f'{base_path}/{piece["image_path"]}'
    with open(file) as f:
        src = json.load(f)["focused_regions"]
    piece["relations"] = []
    for p,value in src.items():
        piece["relations"].append([f"{gen_path}/sam_results/"+p.split("sam_results/")[1], value['relation'], value['neg_samples']])
    listofstuff[i] = piece

data = {'data': listofstuff}

with open('final_metadata_openpsg.json', 'w') as f:
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

with open('final_classes_openpsg.json', 'w') as f:
    json.dump({
        "actions": sorted(list(actions)),
        "objects": sorted(list(objects)),
        "relations": sorted(list(relations)),
    }, f, indent=4)