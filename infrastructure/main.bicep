@minLength(3)
@maxLength(12)
@description('basename to add to all of our resources')
param baseName string

@description('where to deploy the storage account, ex westeurope')
@allowed([
  'eastus'
  'eastus2'
  'southcentralus'
  'southeastasia'
  'westcentralus'
  'westeurope'
  'westus2'
  'centralus'
])
param location string

@description('name of the storage account (must be globally unique)')
param saName string = '${toLower(baseName)}sa'

@description('name of the blob storage container')
param saContainerName string = '${toLower(baseName)}sac'

resource storageAccount 'Microsoft.Storage/storageAccounts@2022-09-01' = {
  name: saName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
  }
}

resource blobServices 'Microsoft.Storage/storageAccounts/blobServices@2022-09-01' = {
  name: 'default'
  parent: storageAccount
}

resource storageContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-09-01' = {
  name: saContainerName
  parent: blobServices
}
