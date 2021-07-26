import json
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--file", type=str)
args = parser.parse_args()
with open("converted.json", "w", encoding="utf8") as out_f:
    with open(args.file, "r") as fp:
        lines = fp.readlines()
        tmp = ""
        for l in lines:
            flag = 1
            if "}" in l:
                tmp += "}"
                tmp = tmp.replace("\n","")
                #tmp = tmp.replace("\\","")
                print(tmp)
                tmp = json.loads(tmp)
                json.dump(tmp, out_f, ensure_ascii=False)
                out_f.write("\n")
                flag = 0
            if "{" in l:
                tmp = "{"
            elif flag:
                tmp += l

