import glob

out_file = 'indices3.txt'

prefixes = [
    "/mnt/MIG_store/Datasets/ActionGenome/results_29_9/gemma_jsons/",
    "/mnt/MIG_store/Datasets/ActionGenome/results_29_9/dino_results/",
    "/mnt/MIG_store/Datasets/ActionGenome/results_29_9/sam_results/",
]

suffixes = [
    "_gemma.json",
    "_grounding_dino.json",
    ""
]

x = set([s.removesuffix(suffixes[0]).removeprefix(prefixes[0]) for s in glob.glob(f"{prefixes[0]}*{suffixes[0]}")])

for i in range(1,len(suffixes)):
    su = suffixes[i]
    pr = prefixes[i]
    y = set([s.removesuffix(su).removeprefix(pr) for s in glob.glob(f"{pr}*{su}")])
    x = x.intersection(y)

x = sorted(list(x))

with open(out_file, 'w') as f:
    f.write("\n".join(x))
