param(
    [datetime]$StartDate = [datetime]::new(2021, 11, 18, 0, 0, 0, [DateTimeKind]::Utc),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "real_data\binance_20_assets"),
    [double]$NotionalPerAsset = 10000
)

$ErrorActionPreference = "Stop"
$symbols = @(
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "SOLUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT",
    "BCHUSDT", "TRXUSDT", "AVAXUSDT", "UNIUSDT", "ATOMUSDT",
    "ETCUSDT", "XLMUSDT", "FILUSDT", "AAVEUSDT", "NEARUSDT"
)
$culture = [Globalization.CultureInfo]::InvariantCulture
$startUtc = [DateTime]::SpecifyKind($StartDate.Date, [DateTimeKind]::Utc)
$startTime = [DateTimeOffset]::new($startUtc.AddDays(-1)).ToUnixTimeMilliseconds()
$todayUtc = [DateTime]::SpecifyKind([DateTime]::UtcNow.Date, [DateTimeKind]::Utc)
$endTime = [DateTimeOffset]::new($todayUtc).ToUnixTimeMilliseconds() - 1
$pricesBySymbol = @{}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

foreach ($symbol in $symbols) {
    $prices = [System.Collections.Generic.Dictionary[string, double]]::new([StringComparer]::Ordinal)
    $cursor = $startTime
    while ($cursor -le $endTime) {
        $uri = "https://api.binance.com/api/v3/klines?symbol=$symbol&interval=1d&startTime=$cursor&endTime=$endTime&limit=1000"
        $bars = Invoke-RestMethod -Uri $uri -Method Get -TimeoutSec 60
        if ($null -eq $bars -or $bars.Count -eq 0) {
            break
        }

        foreach ($bar in $bars) {
            $openTime = [int64]$bar[0]
            $date = [DateTimeOffset]::FromUnixTimeMilliseconds($openTime).UtcDateTime.ToString("yyyy-MM-dd", $culture)
            $close = [double]::Parse([string]$bar[4], $culture)
            if ($prices.ContainsKey($date)) {
                throw "Duplicate daily candle for $symbol on $date"
            }
            $prices.Add($date, $close)
        }

        $lastOpenTime = [int64]$bars[$bars.Count - 1][0]
        if ($lastOpenTime -lt $cursor) {
            throw "Binance returned non-advancing candles for $symbol"
        }
        $cursor = $lastOpenTime + 86400000
        if ($bars.Count -lt 1000) {
            break
        }
        Start-Sleep -Milliseconds 100
    }

    if ($prices.Count -lt 3) {
        throw "Insufficient daily candles for $symbol"
    }
    $pricesBySymbol[$symbol] = $prices
    Write-Output "$symbol`: downloaded $($prices.Count) daily closes"
}

$commonDates = [System.Collections.Generic.List[string]]::new()
foreach ($date in $pricesBySymbol[$symbols[0]].Keys) {
    $presentInAll = $true
    foreach ($symbol in $symbols) {
        if (-not $pricesBySymbol[$symbol].ContainsKey($date)) {
            $presentInAll = $false
            break
        }
    }
    if ($presentInAll) {
        $commonDates.Add($date)
    }
}
$commonDates.Sort([StringComparer]::Ordinal)

$returnLines = [System.Collections.Generic.List[string]]::new()
$returnLines.Add((@("date") + $symbols) -join ",")
for ($index = 1; $index -lt $commonDates.Count; $index++) {
    $date = [DateTime]::ParseExact($commonDates[$index], "yyyy-MM-dd", $culture)
    $priorDate = [DateTime]::ParseExact($commonDates[$index - 1], "yyyy-MM-dd", $culture)
    if ($date -ne $priorDate.AddDays(1)) {
        continue
    }

    $fields = [string[]]::new($symbols.Count + 1)
    $fields[0] = $commonDates[$index]
    for ($asset = 0; $asset -lt $symbols.Count; $asset++) {
        $symbol = $symbols[$asset]
        $previousClose = $pricesBySymbol[$symbol][$commonDates[$index - 1]]
        $currentClose = $pricesBySymbol[$symbol][$commonDates[$index]]
        $logReturn = [Math]::Log($currentClose / $previousClose)
        if ([double]::IsNaN($logReturn) -or [double]::IsInfinity($logReturn)) {
            throw "Non-finite log return for $symbol on $($commonDates[$index])"
        }
        $fields[$asset + 1] = $logReturn.ToString("R", $culture)
    }
    $returnLines.Add(($fields -join ","))
}

if ($returnLines.Count -lt 3) {
    throw "Fewer than two aligned return observations were produced"
}

$encoding = [Text.UTF8Encoding]::new($false)
$returnsCsv = Join-Path $OutputDirectory "returns.csv"
$positionsCsv = Join-Path $OutputDirectory "positions.csv"
[IO.File]::WriteAllLines($returnsCsv, $returnLines, $encoding)
$positionLines = [System.Collections.Generic.List[string]]::new()
$positionLines.Add("asset,notional")
foreach ($symbol in $symbols) {
    $positionLines.Add("$symbol,$($NotionalPerAsset.ToString('R', $culture))")
}
[IO.File]::WriteAllLines($positionsCsv, $positionLines, $encoding)

Write-Output "Aligned return observations: $($returnLines.Count - 1)"
Write-Output "Date range: $($returnLines[1].Split(',')[0]) to $($returnLines[$returnLines.Count - 1].Split(',')[0])"
Write-Output "Wrote $returnsCsv"
Write-Output "Wrote $positionsCsv"
