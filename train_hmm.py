#!/usr/bin/python3

import os
import numpy as np
from hmmlearn import hmm
import pickle
import sklearn
import functools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


NUMBER_OF_RUNS = 100000
TRAINING_SET = 90000
NUMBER_OF_HIDDEN_STATES = 43 # 64
NUMBER_OF_TRAINING_SEQUENCES = 1000
NUMBER_OF_VALIDATION_SEQUENCES = 1000
SEQUENCE_LENGTH = int(os.environ.get("SEQUENCE_LENGTH", 100))
FULL_SEQUENCES = SEQUENCE_LENGTH == -1


def transform(sequences):
    if FULL_SEQUENCES:
        lengths = [len(r) for r in sequences]
    else:
        target_lengths = [(r.shape[0] // SEQUENCE_LENGTH) * SEQUENCE_LENGTH for r in sequences]
        offsets = [np.random.randint(0, r.shape[0] - target_length) for r, target_length in zip(sequences, target_lengths)]
        clipped_runs = [r[offset:offset + target_length] for r, offset, target_length in zip(sequences, offsets, target_lengths)]
        lengths = [[SEQUENCE_LENGTH] * (len(r) // SEQUENCE_LENGTH) for r in clipped_runs]
    return np.concatenate(clipped_runs, axis=0), lengths

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

if FULL_SEQUENCES:
    evaluation_results_filename = f"evaluation_results.pkl"
else:
    evaluation_results_filename = f"evaluation_results_{SEQUENCE_LENGTH}.pkl"

if os.path.isfile(evaluation_results_filename):
    with open(evaluation_results_filename, "rb") as file:
        [hidden_states_sequence, direct_scores, scores,
            all_log_likelihoods_hmm_likelihood_healthy, all_log_likelihoods_hmm_likelihood_degraded,
            all_likelihoods_hmm_hidden_kl_healthy, all_likelihoods_hmm_hidden_kl_degraded] = pickle.load(file)
else:
    scores = []
    direct_scores = []
    hidden_states_sequence = []

    all_log_likelihoods_hmm_likelihood_healthy = []
    all_log_likelihoods_hmm_likelihood_degraded = []
    all_likelihoods_hmm_hidden_kl_healthy = []
    all_likelihoods_hmm_hidden_kl_degraded = []

    for number_of_hidden_states in range(1, NUMBER_OF_HIDDEN_STATES):
        model = hmm.MultinomialHMM(number_of_hidden_states)
        correct_runs = get_n_sequences(NUMBER_OF_TRAINING_SEQUENCES)

        X, lengths = transform(correct_runs)

        if FULL_SEQUENCES:
            model_filename = f"hmm_{number_of_hidden_states}.pkl"
        else:
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
        for X in X_correct:
            if not FULL_SEQUENCES:
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
        total_count = np.sum(np.asarray(histogram_of_correct_run_hidden_states))
        histogram_of_correct_run_hidden_states = [c/total_count for c in histogram_of_correct_run_hidden_states]

        X_error = get_n_sequences(NUMBER_OF_VALIDATION_SEQUENCES, False, validation=True)

        faulty_log_probs = []
        erroneous_distribution_differences = []
        histogram_of_faulty_run_hidden_states = [0 for _ in range(number_of_hidden_states)]
        for X in X_error:
            if not FULL_SEQUENCES:
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

        total_count = np.sum(np.asarray(histogram_of_correct_run_hidden_states))
        histogram_of_faulty_run_hidden_states = [c/total_count for c in histogram_of_faulty_run_hidden_states]

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

        correct_log_probs = np.asarray(correct_log_probs).flatten()
        faulty_log_probs = np.asarray(faulty_log_probs).flatten()
        correct_distribution_differences = np.asarray(correct_distribution_differences).flatten()
        erroneous_distribution_differences = np.asarray(erroneous_distribution_differences).flatten()

        all_log_likelihoods_hmm_likelihood_healthy.append(correct_log_probs)
        all_log_likelihoods_hmm_likelihood_degraded.append(faulty_log_probs)
        all_likelihoods_hmm_hidden_kl_healthy.append(correct_distribution_differences)
        all_likelihoods_hmm_hidden_kl_degraded.append(erroneous_distribution_differences)

        worst_score_for_a_healthy_validation_run = np.max(correct_distribution_differences)
        number_of_degraded_samples_correctly_detected = np.sum(erroneous_distribution_differences > worst_score_for_a_healthy_validation_run)
        fraction_of_degraded_samples_correctly_detected = number_of_degraded_samples_correctly_detected / erroneous_distribution_differences.shape[0]

        print(f"The worst score for a healthy validation sample: {worst_score_for_a_healthy_validation_run}")
        print(f"Number of degraded validation samples correctly detected: {number_of_degraded_samples_correctly_detected}")
        print(f"Fraction of degraded validation samples correctly detected: {fraction_of_degraded_samples_correctly_detected}")

    with open(evaluation_results_filename, "wb") as file:
        pickle.dump([hidden_states_sequence, direct_scores, scores,
            all_log_likelihoods_hmm_likelihood_healthy, all_log_likelihoods_hmm_likelihood_degraded,
            all_likelihoods_hmm_hidden_kl_healthy, all_likelihoods_hmm_hidden_kl_degraded], file)

print(f"hidden_states={hidden_states_sequence}")
print(f"direct_scores={direct_scores}")
print(f"scores={scores}")

SEQUENCE_LABEL = f"sequence length: {SEQUENCE_LENGTH}" if not FULL_SEQUENCES else "full sequence"

plt.figure(figsize=(15,8))
g = sns.barplot(x=hidden_states_sequence, y=scores)
g.set_xlabel("Number of hidden states")
g.set_ylabel("Receiver Operating Characteristic")
g.set_title(f"ROC metric for hidden state distribution KL divergence, {SEQUENCE_LABEL}")
g.get_figure().savefig(f"results/roc_kl_score_{SEQUENCE_LENGTH}.eps")
plt.show()

plt.figure(figsize=(15,8))
g = sns.barplot(x=hidden_states_sequence, y=direct_scores)
g.set_xlabel("Number of hidden states")
g.set_ylabel("Receiver Operating Characteristic")
g.set_title(f"ROC metric for HMM likelihood, {SEQUENCE_LABEL}")
g.get_figure().savefig(f"results/roc_hmm_score_{SEQUENCE_LENGTH}.eps")
plt.show()

all_log_likelihoods_hmm_likelihood_healthy = np.asarray(all_log_likelihoods_hmm_likelihood_healthy[5]).flatten()
all_log_likelihoods_hmm_likelihood_degraded = np.asarray(all_log_likelihoods_hmm_likelihood_degraded[5]).flatten()
all_likelihoods_hmm_hidden_kl_healthy = np.asarray(all_likelihoods_hmm_hidden_kl_healthy[5]).flatten()
all_likelihoods_hmm_hidden_kl_degraded = np.asarray(all_likelihoods_hmm_hidden_kl_degraded[5]).flatten()

df = pd.concat(axis=0, ignore_index=True, objs=[
    pd.DataFrame.from_dict({'value': all_log_likelihoods_hmm_likelihood_healthy, 'name': 'Healthy'}),
    pd.DataFrame.from_dict({'value': all_log_likelihoods_hmm_likelihood_degraded, 'name': 'Degraded'})
])
plt.figure(figsize=(15,8))
g = sns.histplot(
    data=df, x='value', hue='name', multiple='dodge', bins=100
)

g.set_xlabel("HMM Likelihood")
g.set_ylabel("Count")
g.set_title(f"Likelihoods of sequences, {SEQUENCE_LABEL}")
g.get_figure().savefig(f"results/hmm_histograms_{SEQUENCE_LENGTH}.eps")
plt.show()

df = pd.concat(axis=0, ignore_index=True, objs=[
    pd.DataFrame.from_dict({'value': all_likelihoods_hmm_hidden_kl_healthy, 'name': 'Healthy'}),
    pd.DataFrame.from_dict({'value': all_likelihoods_hmm_hidden_kl_degraded, 'name': 'Degraded'})
])
plt.figure(figsize=(15,8))
g = sns.histplot(
    data=df, x='value', hue='name', multiple='dodge', bins=100
)

g.set_xlabel("Hidden states KL divergence")
g.set_ylabel("Count")
g.set_title(f"KL divergences of hidden states, {SEQUENCE_LABEL}")
g.get_figure().savefig(f"results/kl_histograms_{SEQUENCE_LENGTH}.eps")
plt.show()
