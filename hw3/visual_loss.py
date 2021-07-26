import matplotlib.pyplot as plt
import json
loss = []
y = 0
for i in range(35):
    try:
        with open(f"./statistic/rl/loss_{i}", "r") as fp:
            loss.extend(fp.readlines())
    except:
        pass
plt.plot(list(range(len(loss))), list(map(lambda x:float(x.strip()),loss)))
plt.savefig("loss.png")
