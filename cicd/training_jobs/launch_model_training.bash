#!/bin/bash

AZURE_RG="$1"
AZUREML_WORKSPACE="$2"

script_dir="$(cd "$(dirname "$0")" && pwd)"

az ml job create \
--resource-group $AZURE_RG \
--workspace-name $AZUREML_WORKSPACE \
--file $script_dir/train_job.yml