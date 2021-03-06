# Multinomial Hidden Markov Model Evaluation

This repository implements an evaluation of a multinomial Hidden Markov Model (HMM) for simulated interleaved process traces.

The simulation dataset is generated by this project: https://github.com/keskival/FAS-Simulator

Get the data by following the instructions there to get the directory named `numpy_data`.

Run the evaluation: `./train_hmm.py`

The script will save trained models for further evaluation. It takes some days to run on a reasonable computer.

This produces outputs to the standard output giving minimums, means and maximums for some discriminative metrics for 1,000 faultless runs in the validation set, and 1,000 runs with simulated errors, for each number of HMM hidden states from 1 to 64.

KL divergences are computed for the distributions of hidden states for faultless runs and the anomalous runs against training set distribution.

In the end ROC values are graphed for each number of hidden states based on the discriminative power of the KL divergences against the training set.

## Results

See the results in: [Results](https://github.com/keskival/fas_hmm/blob/main/results/results.pdf).
