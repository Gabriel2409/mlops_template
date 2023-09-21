param name string
param location string
param stoacctid string
param kvid string
param appinsightid string
param crid string
param tags object

// AML workspace
resource mlw 'Microsoft.MachineLearningServices/workspaces@2023-04-01' = {
  name: name
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    tier: 'Standard'
    name: 'Standard'
  }
  properties: {
    storageAccount: stoacctid
    keyVault: kvid
    applicationInsights: appinsightid
    containerRegistry: crid
    // encryption: {
    //   status: 'Disabled'
    //   keyVaultProperties: {
    //     keyIdentifier: ''
    //     keyVaultArmId: ''
    //   }
    // }
  }

  tags: tags
}

output mlwOut string = mlw.id
