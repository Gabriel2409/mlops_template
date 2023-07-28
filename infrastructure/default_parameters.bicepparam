// to make deploying with cli easier.
// parameters will be overriden by pipeline variables
using 'main.bicep'

param baseName = 'mlops_sample'
param location = 'westeurope'
param saName = 'mlopsstorage124fxgc'
param saContainerName = 'mlopscontainer'
