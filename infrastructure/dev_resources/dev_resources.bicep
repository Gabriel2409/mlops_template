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
param saName string = '${toLower(baseName)}amlsafxgc0045'

@description('name of the blob storage container')
param saContainerName string = '${toLower(baseName)}amlsac'

@description('name of the key vault')
param keyVaultName string = '${toLower(baseName)}amlkv'

@description('name of the application insights')
param applicationInsightsName string = '${toLower(baseName)}amlai'

@description('name of the container registry')
param containerRegistryName string = '${toLower(baseName)}amlcr'

@description('name of the Azure ML workspace')
param amlWorkspaceName string = '${toLower(baseName)}amlws'

resource storageAccount 'Microsoft.Storage/storageAccounts@2022-09-01' = {
  name: saName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    supportsHttpsTrafficOnly: true
  }
}

resource blobServices 'Microsoft.Storage/storageAccounts/blobServices@2022-09-01' = {
  name: 'default'
  parent: storageAccount
  properties: {
    deleteRetentionPolicy: {
      enabled: false
    }
  }
}

resource storageContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-09-01' = {
  name: saContainerName
  parent: blobServices
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: keyVaultName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      name: 'standard'
      family: 'A'
    }
    accessPolicies: []

  }
}

resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: applicationInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
  }
}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2022-12-01' = {
  name: containerRegistryName
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: true
  }
}

resource amlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-04-01' = {
  name: amlWorkspaceName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    tier: 'Standard'
    name: 'Standard'
  }
  properties: {
    friendlyName: amlWorkspaceName
    keyVault: keyVault.id
    applicationInsights: applicationInsights.id
    storageAccount: storageAccount.id
    containerRegistry: containerRegistry.id
  }

}
