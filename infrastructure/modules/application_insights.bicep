param baseName string
param location string
param tags object

resource appi 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${baseName}'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
  }

  tags: tags
}

output appiOut string = appi.id
