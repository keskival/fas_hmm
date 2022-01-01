#!/usr/bin/python3

import os
import numpy as np
from hmmlearn import hmm
import pickle
import sklearn
import functools
import matplotlib.pyplot as plt


NUMBER_OF_RUNS = 100000
TRAINING_SET = 90000
NUMBER_OF_HIDDEN_STATES = 32 #64
NUMBER_OF_TRAINING_SEQUENCES = 1000
NUMBER_OF_VALIDATION_SEQUENCES = 1000
SEQUENCE_LENGTH = 100

def transform(sequences):
    lengths = [[SEQUENCE_LENGTH] * (r.shape[0] // SEQUENCE_LENGTH) + [r.shape[0] - (r.shape[0] // SEQUENCE_LENGTH) * SEQUENCE_LENGTH] for r in sequences]
    lengths = functools.reduce(lambda a, b: a + b, lengths, [])
    lengths = [l for l in lengths if l > 0]
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

scores = []
direct_scores = []
hidden_states_sequence = []

for number_of_hidden_states in range(1, NUMBER_OF_HIDDEN_STATES):
    model = hmm.MultinomialHMM(number_of_hidden_states)
    correct_runs = get_n_sequences(NUMBER_OF_TRAINING_SEQUENCES)

    X, lengths = transform(correct_runs)

    model_filename = f"hmm_{number_of_hidden_states}_{SEQUENCE_LENGTH}.pkl"

    if os.path.isfile(model_filename):
        with open(model_filename, "rb") as file:
            model = pickle.load(file)
    else:
        print("Fitting...")
        model.fit(X, lengths)
        print("Fitted.")
        with open(model_filename, "wb") as file:
            pickle.dump(model, file)

    hidden_states = model.predict(X, lengths)
    histogram_of_training_run_hidden_states = [0 for _ in range(number_of_hidden_states)]
    for s in hidden_states:
        histogram_of_training_run_hidden_states[s] += 1
    histogram_of_training_run_hidden_states = [c/len(X) for c in histogram_of_training_run_hidden_states]

    X_correct = get_n_sequences(NUMBER_OF_VALIDATION_SEQUENCES, validation=True)

    correct_log_probs = []
    correct_distribution_differences = []
    histogram_of_correct_run_hidden_states = [0 for _ in range(number_of_hidden_states)]
    epsilon = 1e-8
    for X in X_correct:
        offset = np.random.randint(0, len(X) - SEQUENCE_LENGTH)
        X = X[offset:offset + SEQUENCE_LENGTH]
        log_prob = model.score(X, [len(X)])
        hidden_states = model.predict(X, [len(X)])
        correct_log_probs.append(log_prob)
        local_histogram_of_correct_run_hidden_states = [0 for _ in range(number_of_hidden_states)]
        for s in hidden_states:
            local_histogram_of_correct_run_hidden_states[s] += 1
            histogram_of_correct_run_hidden_states[s] += 1
        local_histogram_of_correct_run_hidden_states = [c/len(X) for c in local_histogram_of_correct_run_hidden_states]
        # D_kl(correct || training)
        p = np.array(local_histogram_of_correct_run_hidden_states)
        q = np.array(histogram_of_training_run_hidden_states)
        non_zero = (q > 0) * (p > 0)
        p = p[non_zero]
        q = q[non_zero]
        d_kl = np.sum(p * np.log(p/q))
        correct_distribution_differences.append(d_kl)
    histogram_of_correct_run_hidden_states = [c/(len(X_correct) * SEQUENCE_LENGTH) for c in histogram_of_correct_run_hidden_states]

    X_error = get_n_sequences(NUMBER_OF_VALIDATION_SEQUENCES, False, validation=True)

    faulty_log_probs = []
    erroneous_distribution_differences = []
    histogram_of_faulty_run_hidden_states = [0 for _ in range(number_of_hidden_states)]
    for X in X_error:
        offset = np.random.randint(0, len(X) - SEQUENCE_LENGTH)
        X = X[offset:offset + SEQUENCE_LENGTH]
        log_prob = model.score(X, [len(X)])
        hidden_states = model.predict(X, [len(X)])
        faulty_log_probs.append(log_prob)
        local_histogram_of_faulty_run_hidden_states = [0 for _ in range(number_of_hidden_states)]
        for s in hidden_states:
            local_histogram_of_faulty_run_hidden_states[s] += 1
            histogram_of_faulty_run_hidden_states[s] += 1
        local_histogram_of_faulty_run_hidden_states = [c/len(X) for c in local_histogram_of_faulty_run_hidden_states]
        # D_kl(faulty || training)
        p = np.array(local_histogram_of_faulty_run_hidden_states)
        q = np.array(histogram_of_training_run_hidden_states)
        non_zero = (q > 0) * (p > 0)
        p = p[non_zero]
        q = q[non_zero]
        d_kl = np.sum(p * np.log(p/q))
        erroneous_distribution_differences.append(d_kl)

    histogram_of_faulty_run_hidden_states = [c/(len(X_error) * SEQUENCE_LENGTH) for c in histogram_of_faulty_run_hidden_states]

    correct_log_probs = np.stack(correct_log_probs)
    faulty_log_probs = np.stack(faulty_log_probs)
    correct_distribution_differences = np.stack(correct_distribution_differences)
    erroneous_distribution_differences = np.stack(erroneous_distribution_differences)
    correct_log_prob = [np.min(correct_log_probs), np.mean(correct_log_probs), np.max(correct_log_probs)]
    faulty_log_prob = [np.min(faulty_log_probs), np.mean(faulty_log_probs), np.max(faulty_log_probs)]
    correct_distribution_diff = [np.min(correct_distribution_differences), np.mean(correct_distribution_differences), np.max(correct_distribution_differences)]
    faulty_distribution_diff = [np.min(erroneous_distribution_differences), np.mean(erroneous_distribution_differences), np.max(erroneous_distribution_differences)]

    log_probabilities_positive_for_positive = correct_log_probs
    log_probabilities_positive_for_negative = faulty_log_probs

    y_true = np.concatenate([np.ones_like(log_probabilities_positive_for_positive), np.zeros_like(log_probabilities_positive_for_negative)])
    y_pred = np.concatenate([log_probabilities_positive_for_positive, log_probabilities_positive_for_negative])
    roc_auc_score_direct = sklearn.metrics.roc_auc_score(y_true, y_pred)

    y_true = np.concatenate([np.zeros_like(correct_distribution_differences), np.ones_like(erroneous_distribution_differences)])
    y_pred = np.concatenate([correct_distribution_differences, erroneous_distribution_differences])
    roc_auc_score = sklearn.metrics.roc_auc_score(y_true, y_pred)

    print(f"Number of hidden states: {number_of_hidden_states}, Correct log prob min,mean,max: {correct_log_prob}, faulty log prob min,mean,max: {faulty_log_prob}, correct hidden state distribution diffs: {correct_distribution_diff}, faulty hidden state distribution diffs: {faulty_distribution_diff}")
    print(f"ROC AUC score direct: {roc_auc_score_direct}, ROC AUC score: {roc_auc_score}")
    hidden_states_sequence.append(number_of_hidden_states)
    direct_scores.append(roc_auc_score_direct)
    scores.append(roc_auc_score)
print(f"hidden_states={hidden_states_sequence}")
print(f"direct_scores={direct_scores}")
print(f"scores={scores}")

plt.plot(hidden_states_sequence, scores)
plt.show()
