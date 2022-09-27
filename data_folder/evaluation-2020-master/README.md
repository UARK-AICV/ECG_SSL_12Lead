# PhysioNet/CinC Challenge 2020 Evaluation Metrics

This repository contains the Python and MATLAB evaluation code for the PhysioNet/Computing in Cardiology Challenge 2020. The `evaluate_12ECG_score` script evaluates the output of your algorithm using the evaluation metric that is described on the [webpage](https://physionetchallenges.github.io/2020/) for the PhysioNet/CinC Challenge 2020. While this script reports multiple evaluation metric, we use the last score (`Challenge Metric`) to evaluate your algorithm.

## Python

You can run the Python evaluation code by installing the NumPy Python package and running

    python evaluate_12ECG_score.py labels outputs scores.csv class_scores.csv

where `labels` is a directory containing files with one or more labels for each 12-lead ECG recording, such as the training database on the PhysioNet webpage; `outputs` is a directory containing files with outputs produced by your algorithm for those recordings; `scores.csv` (optional) is a collection of scores for your algorithm; and `class_scores.csv` (optional) is a collection of per-class scores for your algorithm.

## MATLAB

You can run the MATLAB evaluation code by installing Python and the NumPy Python package and running

    evaluate_12ECG_score(labels, outputs, scores.csv, class_scores.csv)

where `labels` is a directory containing files with one or more labels for each 12-lead ECG recording, such as the training database on the PhysioNet webpage; `outputs` is a directory containing files with outputs produced by your algorithm for those recordings; `scores.csv` (optional) is a collection of scores for your algorithm; and `class_scores.csv` (optional) is a collection of per-class scores for your algorithm.

## Troubleshooting

Unable to run this code with your code? Try one of the [baseline classifiers](https://physionetchallenges.github.io/2020/#submissions) on the [training data](https://physionetchallenges.github.io/2020/#data). Unable to install or run Python? Try  [Python](https://www.python.org/downloads/), [Anaconda](https://www.anaconda.com/products/individual), or your package manager.
