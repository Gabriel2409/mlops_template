// in the initial template, target scope is subscription.this allows to creates resource group.
// However you need authorization at the subscription level to do so.
// In this version, we suppose resource groups are created beforehand. That allows
// to limit the service principal authorization to the resource group.
// Here, we also use only a signle environement. In reality, we may want to separate
// prod and dev env

// compute clusters and environments can be deployed as arm templates here as well.
// However, as ml environments are bound to change more often, they are not deployed here

targetScope = 'resourceGroup'
param location string
param namespace string
param environement string
param resource_group string
param aml_workspace string
param application_insights string
param key_vault string
param container_registry string
param storage_account string

param tags object = {
  Owner: 'mlopstemplate'
  Project: 'mlopstemplate'
  Environment: environement
  Toolkit: 'bicep'
  Name: namespace
}

// resource rg 'Microsoft.Resources/resourceGroups@2020-06-01' = {
//   name: resource_group
//   location: location

//   tags: tags
// }

// Storage Account
module st './modules/storage_account.bicep' = {
  name: 'st'
  scope: resourceGroup(resource_group)
  params: {
    name: storage_account
    location: location
    tags: tags
  }
}

// Key Vault
module kv './modules/key_vault.bicep' = {
  name: 'kv'
  scope: resourceGroup(resource_group)
  params: {
    name: key_vault
    location: location
    tags: tags
  }
}

// App Insights
module appi './modules/application_insights.bicep' = {
  name: 'appi'
  scope: resourceGroup(resource_group)
  params: {
    name: application_insights
    location: location
    tags: tags
  }
}

// Container Registry
module cr './modules/container_registry.bicep' = {
  name: 'cr'
  scope: resourceGroup(resource_group)
  params: {
    name: container_registry
    location: location
    tags: tags
  }
}

// AML workspace
module mlw './modules/aml_workspace.bicep' = {
  name: 'mlw'
  scope: resourceGroup(resource_group)
  params: {
    name: aml_workspace
    location: location
    stoacctid: st.outputs.stOut
    kvid: kv.outputs.kvOut
    appinsightid: appi.outputs.appiOut
    crid: cr.outputs.crOut
    tags: tags
  }
}

module mlwcc './modules/aml_cpu_computecluster.bicep' = {
  name: 'mlwcc'
  dependsOn: [ mlw ]
  scope: resourceGroup(resource_group)
  params: {
    location: location
    name: 'cpu-cluster'
    aml_workspace: aml_workspace
  }
}
