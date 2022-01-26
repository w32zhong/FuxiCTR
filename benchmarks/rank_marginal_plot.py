import os
import re
import fire
import glob
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def rank_marginal_plot(rank_marginal_result_file, draw=True, title=None, max_samples=1000):
    data = defaultdict(list)
    with open(rank_marginal_result_file, 'r') as fh:
        for i, line in enumerate(fh):
            if i == 0: continue
            line = line.rstrip()
            fields = line.split(',')
            index, qid, docid, rank, score = fields
            rank = float(rank)
            score = float(score)
            data[(qid, docid)].append((rank, score))

    all_Y = []
    for i, key in enumerate(data):
        X = [x[0] for x in data[key]]
        Y = [x[1] for x in data[key]]
        if draw: plt.plot(X, Y)
        all_Y.append(Y)
        if i > max_samples:
            break
    all_Y = np.array(all_Y)
    if draw:
        if title is not None: plt.title(title)
        plt.show()
    return X, all_Y.mean(0)


def plot_all_under_dir(root_dir, title=None):
    for file in glob.glob(f'{root_dir}/rank_marginal-*.txt'): 
        m = re.search(r"rank_marginal-(\w+)_base.txt", file)
        model_name = m.group(1)
        print(model_name)
        x, y = rank_marginal_plot(file, draw=False)
        plt.plot(x, y, label=model_name)
    if title is not None: plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        "rank_marginal_plot": rank_marginal_plot,
        "plot_all_under_dir": plot_all_under_dir,
    })
