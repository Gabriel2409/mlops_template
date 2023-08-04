# Pipeline dummy_classifier

> _Note:_ This is a `README.md` boilerplate generated using `Kedro 0.18.11`.

## Overview

This pipeline is designed to perform the task of combining 'title' and 'description' fields from raw project data into a single 'text' field.
It takes raw project data as input and produces processed project data with the combined 'text' field as output.

## Pipeline Inputs

- **projects_train_raw**: Raw training data containing 'title' and 'description' fields for projects.
- **projects_test_raw**: Raw test data containing 'title' and 'description' fields for projects.

## Pipeline Outputs

- **projects_train_text**: Processed training data with 'text' field that combines 'title' and 'description'.
- **projects_test_text**: Processed test data with 'text' field that combines 'title' and 'description'.
