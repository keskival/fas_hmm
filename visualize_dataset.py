#!/usr/bin/python3

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn


NUMBER_OF_RUNS = 100000
NUMBER_OF_EVENT_IDS = 47

def get_all_sequences_histograms(correct=True):
    histogram = [0 for _ in range(NUMBER_OF_EVENT_IDS)]
    for run_index in range(NUMBER_OF_RUNS):
        sequence_source = "correct_runs" if correct else "runs_with_errors"
        run = np.load(f"numpy_data/{sequence_source}/{run_index}.npy")
        for event in run:
            histogram[event] += 1
    histogram = np.asarray(histogram)
    histogram = histogram / np.sum(histogram)
    return histogram

if os.path.isfile("healthy_histogram.npy"):
    healthy_histogram = np.load("healthy_histogram.npy")
else:
    healthy_histogram = get_all_sequences_histograms(True)
    np.save("healthy_histogram.npy", healthy_histogram)
if os.path.isfile("degraded_histogram.npy"):
    degraded_histogram = np.load("degraded_histogram.npy")
else:
    degraded_histogram = get_all_sequences_histograms(False)
    np.save("degraded_histogram.npy", degraded_histogram)

print(f"healthy_histogram={healthy_histogram}")
print(f"degraded_histogram={degraded_histogram}")

plt.figure(figsize=(15,8))
g = seaborn.barplot(x=np.arange(0, NUMBER_OF_EVENT_IDS), y=healthy_histogram)
g.set_xlabel("Event id")
g.set_ylabel("Frequency")
g.set_title("Healthy runs")
g.get_figure().savefig("results/healthy_runs_data_histogram.eps")

plt.show()

plt.figure(figsize=(15,8))
g = seaborn.barplot(x=np.arange(0, NUMBER_OF_EVENT_IDS), y=degraded_histogram)

g.set_xlabel("Event id")
g.set_ylabel("Frequency")
g.set_title("Degraded runs")
g.get_figure().savefig("results/degraded_runs_data_histogram.eps")
plt.show()
