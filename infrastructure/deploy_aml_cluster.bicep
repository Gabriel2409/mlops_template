
targetScope = 'resourceGroup'

param location string = 'westeurope'
param prefix string
param postfix string
param env string // azure devops environment
param resource_group string // added because scope is resourceGroup, not subscription

param tags object = {
  Owner: 'mlopstemplate'
  Project: 'mlopstemplate'
  Environment: env
  Toolkit: 'bicep'
  Name: prefix
}


module mlw './modules/aml_computecluster.bicep' = {
  name: 'mlw'
  scope: resourceGroup(resource_group)
  params: {
    location: location
    workspaceName: mlw.outputs.mlwName
  }
}
