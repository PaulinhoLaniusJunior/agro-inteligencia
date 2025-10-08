# populate_fnn.ps1
# Popula a API FNN a partir do arquivo fnn_log_payloads.jsonl
# Aceita:
#  A) já no padrão: chuva_mm, ph_solo, temp_c, nitrogenio, densidade, rendimento_alto
#  B) alternativa: ph, nitrogen, phosphorus, potassium, organic_matter, label

$uri = "http://127.0.0.1:5001/log/soil_data"
# como estamos executando a partir de ARQUIVO, $PSScriptRoot funciona
$jfile = Join-Path $PSScriptRoot "fnn_log_payloads.jsonl"

if (-not (Test-Path $jfile)) { Write-Error "Arquivo não encontrado: $jfile"; exit 1 }

Get-Content $jfile | ForEach-Object {
  if ([string]::IsNullOrWhiteSpace($_)) { return }
  try {
    $o = $_ | ConvertFrom-Json

    if ($o.PSObject.Properties.Name -contains "chuva_mm") {
      $body = $_
    }
    else {
      $chuva     = [Math]::Round(70 + (Get-Random -Minimum 0 -Maximum 100), 1)
      $temp      = [Math]::Round(18 + (Get-Random -Minimum 0 -Maximum 12), 1)
      $densidade = [Math]::Round(1.10 + (Get-Random -Minimum 0 -Maximum 0.40), 2)
      $r_alto    = if ($o.label -eq "fertil") { 1 } else { 0 }

      $mapped = [ordered]@{
        chuva_mm        = $chuva
        ph_solo         = [double]$o.ph
        temp_c          = $temp
        nitrogenio      = [double]$o.nitrogen
        densidade       = $densidade
        rendimento_alto = $r_alto
      }
      $body = ($mapped | ConvertTo-Json -Depth 3)
    }

    $res = Invoke-RestMethod -Uri $uri -Method Post -ContentType "application/json" -Body $body
    Write-Host "OK:" ($res | ConvertTo-Json -Compress)
  }
  catch {
    Write-Warning "Falhou: $($_.Exception.Message)"
  }
}
