import matplotlib.pyplot as plt
import json
rouge = {
    "rouge-1":{
        "f":[],
        "p":[],
        "r":[]
        },
    "rouge-2":{
        "f":[],
        "p":[],
        "r":[]
        },
    "rouge-l":{
        "f":[],
        "p":[],
        "r":[]
        }
}
for i in range(35):
    with open(f"./statistic/rl/valid_{i}.json", "r") as fp:
        tmp = json.load(fp)
        for key, value in rouge.items():
            for kk, vv in value.items():
                rouge[key][kk].append(tmp[key][kk])
ax = {}
fig, (ax["rouge-1"], ax["rouge-2"], ax["rouge-l"]) = plt.subplots(1, 3, figsize=(15,6))
x = list(range(35))
for k, v in rouge.items():
    for kk, vv in v.items():
        ax[k].plot(x, vv, label = kk)
    ax[k].legend()
    ax[k].set_title(k)
            
plt.savefig("rouge.png")
