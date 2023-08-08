#!/bin/bash
AZURE_RG="$1"
AZUREML_WORKSPACE="$2"

az ml environment create \
--name mlopstplt-cpu-env \
--resource-group $AZURE_RG \
--workspace-name $AZUREML_WORKSPACE \
--image mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest \
--conda-file conda-dependencies-cpu.yml

