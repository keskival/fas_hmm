#!/usr/bin/python3

import os
import numpy as np
from hmmlearn import hmm
import pickle


NUMBER_OF_RUNS = 100000
TRAINING_SET = 90000
NUMBER_OF_HIDDEN_STATES = 64
NUMBER_OF_TRAINING_SEQUENCES = 100
NUMBER_OF_VALIDATION_SEQUENCES = 100

def transform(sequences):
    lengths = [r.shape[0] for r in sequences]
    return np.concatenate(sequences, axis=0), lengths

def get_n_sequences(n, correct=True, validation=False):
    runs = []
    for i in range(n):
        if correct:
            min_index = TRAINING_SET + 1 if validation else 0
            max_index = NUMBER_OF_RUNS if validation else TRAINING_SET
        else:
            min_index = 0
            max_index = NUMBER_OF_RUNS
        run_index = np.random.randint(min_index, max_index)
        sequence_source = "correct_runs" if correct else "runs_with_errors"
        run = np.load(f"numpy_data/{sequence_source}/{run_index}.npy").reshape([-1, 1])
        runs.append(run)
    return runs

for number_of_hidden_states in range(1, NUMBER_OF_HIDDEN_STATES):
    model = hmm.MultinomialHMM(number_of_hidden_states)
    correct_runs = get_n_sequences(NUMBER_OF_TRAINING_SEQUENCES)

    X, lengths = transform(correct_runs)

    model_filename = f"hmm_{number_of_hidden_states}.pkl"

    if os.path.isfile(model_filename):
        with open(model_filename, "rb") as file:
            model = pickle.load(file)
    else:
        print("Fitting...")
        model.fit(X, lengths)
        print("Fitted.")

        with open(model_filename, "wb") as file:
            pickle.dump(model, file)

    X_correct = get_n_sequences(NUMBER_OF_VALIDATION_SEQUENCES, validation=True)

    correct_log_probs = []
    for X in X_correct:
        log_prob = model.score(X, [len(X)])
        correct_log_probs.append(log_prob)

    X_error = get_n_sequences(NUMBER_OF_VALIDATION_SEQUENCES, False, validation=True)

    faulty_log_probs = []
    for X in X_error:
        log_prob = model.score(X, [len(X)])
        faulty_log_probs.append(log_prob)

    correct_log_probs = np.stack(correct_log_probs)
    faulty_log_probs = np.stack(faulty_log_probs)
    correct_log_prob = [np.min(correct_log_probs), np.mean(correct_log_probs), np.max(correct_log_probs)]
    faulty_log_prob = [np.min(faulty_log_probs), np.mean(faulty_log_probs), np.max(faulty_log_probs)]
    print(f"Number of hidden states: {number_of_hidden_states}, Mean correct log prob: {correct_log_prob}, mean faulty log prob: {faulty_log_prob}")
