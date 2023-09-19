// in the initial template, target scope is subscription.this allows to creates resource group.
// However you need authorization at the subscription level to do so.
// In this version, we suppose resource groups are created beforehand. That allows
// to limit the service principal authorization to the resource group.
// Here, we also use only a signle environement. In reality, we may want to separate
// prod and dev env

// compute clusters and environments can be deployed as arm templates here as well.
// However, I prefer to deploy them independently as you don't need resource group
// permission to deploy them, only machine learning workspace level permissions.

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

var baseName  = '${prefix}-${postfix}${env}'
// var resource_group = 'rg-${baseName}'

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
    baseName: '${uniqueString(resourceGroup().id)}${env}'
    location: location
    tags: tags
  }
}

// Key Vault
module kv './modules/key_vault.bicep' = {
  name: 'kv'
  scope: resourceGroup(resource_group)
  params: {
    baseName: baseName
    location: location
    tags: tags
  }
}

// App Insights
module appi './modules/application_insights.bicep' = {
  name: 'appi'
  scope: resourceGroup(resource_group)
  params: {
    baseName: baseName
    location: location
    tags: tags
  }
}

// Container Registry
module cr './modules/container_registry.bicep' = {
  name: 'cr'
  scope: resourceGroup(resource_group)
  params: {
    baseName: '${uniqueString(resourceGroup().id)}${env}'
    location: location
    tags: tags
  }
}

// AML workspace
module mlw './modules/aml_workspace.bicep' = {
  name: 'mlw'
  scope: resourceGroup(resource_group)
  params: {
    baseName: baseName
    location: location
    stoacctid: st.outputs.stOut
    kvid: kv.outputs.kvOut
    appinsightid: appi.outputs.appiOut
    crid: cr.outputs.crOut
    tags: tags
  }
}


// AML compute cluster
// module mlwcc './modules/aml_computecluster.bicep' = {
//   name: 'mlwcc'
//   scope: resourceGroup(resource_group)
//   params: {
//     location: location
//     workspaceName: mlw.outputs.mlwName
//   }
// }