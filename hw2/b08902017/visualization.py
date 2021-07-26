import os
import sys
import json
import matplotlib.pyplot as plt

# plot loss graph
'''
losses = []
step_num = 0
i = 0
while os.path.exists("./statistic/my_bert_256/loss_" + str(i)):
    with open("./statistic/my_bert_256/loss_" + str(i), "r") as fp: 
        lines = fp.readlines()
        losses.extend(list(map(lambda x:float(x.strip()),lines)))
        step_num += len(lines)
        i += 1
loss = []
for i in range(0, len(losses), 500):
    loss.append(sum(losses[i:i + 500])/500)
plt.plot(list(range(0, step_num, 500)), loss, label="smaller non-pretrained bert")
losses = []
step_num = 0
i = 0
while os.path.exists("./statistic/my_bert_768/loss_" + str(i)):
    with open("./statistic/my_bert_768/loss_" + str(i), "r") as fp: 
        lines = fp.readlines()
        losses.extend(list(map(lambda x:float(x.strip()),lines)))
        step_num += len(lines)
        i += 1
loss = []
for i in range(0, len(losses), 500):
    loss.append(sum(losses[i:i + 500])/500)
plt.plot(list(range(0, step_num, 500)), loss, label="larger non-pretrained bert")
plt.xlabel("steps")
plt.ylabel("loss")
plt.legend()
plt.savefig("./graph/non-pretrained_loss.png", format="png")
'''
# plot loss curve of robertas
losses = []
step_num = 0
i = 0
while os.path.exists("./statistic/roberta/loss_" + str(i)):
    with open("./statistic/roberta/loss_" + str(i), "r") as fp: 
        lines = fp.readlines()
        losses.extend(list(map(lambda x:float(x.strip()),lines)))
        step_num += len(lines)
        i += 1
loss = []
for i in range(0, len(losses), 500):
    loss.append(sum(losses[i:i + 500])/500)
plt.plot(list(range(0, step_num, 500)), loss, label="roberta")
losses = []
i = 0
while os.path.exists("./statistic/roberta_large/loss_" + str(i)):
    with open("./statistic/roberta_large/loss_" + str(i), "r") as fp: 
        lines = fp.readlines()
        losses.extend(list(map(lambda x:float(x.strip()),lines)))
        i += 1
loss = []
for i in range(0, len(losses), 500):
    loss.append(sum(losses[i:i + 500])/500)
plt.plot(list(range(0, step_num, 500)), loss[:step_num // 500 + 1], label="roberta_large")
plt.xlabel("steps")
plt.ylabel("loss")
plt.legend()
plt.savefig("./graph/roberta_loss.png", format="png")

for model in ["roberta_large", "roberta", "pretrained_bert", "my_bert_768", "my_bert_256"]:
    statistic_folder = "./statistic/" + model
    i = 0
    ems = []
    f1s = []
    p = plt.figure()
    while os.path.exists(statistic_folder + "/valid_" + str(i) + ".json"):
        with open(statistic_folder + "/valid_" + str(i) + ".json", "r") as fp: 
            data = json.load(fp)
            f1s.append(float(data["f1"]))
            ems.append(float(data["em"]))
            i += 1
    plt.plot(list(range(i)), ems, label="em")
    plt.plot(list(range(i)), f1s, label="f1")
    plt.xlabel("epochs")
    plt.ylabel("score")
    plt.legend()
    plt.savefig("./graph/" + model + "_score.png", format="png")
