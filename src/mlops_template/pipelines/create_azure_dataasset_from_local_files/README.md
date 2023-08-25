# Pipeline create_azure_dataasset_from_local_files

> *Note:* This is a `README.md` boilerplate generated using `Kedro 0.18.12`.

## Overview

Creates a new dataasset in azure from the local files train and test.
This pipeline is supposed to be run only when the dataset changes and uploads it to
azure. Note that you need to run the command with `kedro azureml run` as a local run
will only save it locally


## Pipeline inputs

`projects_train_raw_local`: the csv train data
`projects_test_raw_local`: the csv test data

## Pipeline outputs

`projects_train_raw`: the same data on azure
`projects_test_raw`: the same data on azure
