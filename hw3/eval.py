import json
import argparse
from tw_rouge import get_rouge


def main(args):
    refs, preds = {}, {}

    with open(args.reference) as file:
        for line in file:
            line = json.loads(line)
            refs[line['id']] = line['title']

    with open(args.submission) as file:
        for idx,line in enumerate(file):
            line = json.loads(line)
            print(line["id"])
            preds[line['id']] = line['title']

    keys =  refs.keys()
    refs = [refs[key] for key in keys]
    preds = [preds[key] for key in keys]

    print(json.dumps(get_rouge(preds, refs), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference')
    parser.add_argument('-s', '--submission')
    args = parser.parse_args()
    main(args)
