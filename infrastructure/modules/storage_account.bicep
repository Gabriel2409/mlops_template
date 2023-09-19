param baseName string
param location string
param tags object

// storage account and container
resource st 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'st${baseName}'
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    encryption: {
      services: {
        blob: {
          enabled: true
        }
        file: {
          enabled: true
        }
      }
      keySource: 'Microsoft.Storage'
    }
    supportsHttpsTrafficOnly: true
  }

  tags: tags
}


resource blobServices 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  name: 'default'
  parent: st
  properties: {
    deleteRetentionPolicy: {
      enabled: false
    }
  }
}

resource storageContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-09-01' = {
  name: 'stc${baseName}'
  parent: blobServices
}

output stOut string = st.id
