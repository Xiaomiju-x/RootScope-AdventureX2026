[CmdletBinding()]
param(
    [string]$Output = "",
    [string]$ExistingDataset = "",
    [Alias('LicensePolicy')]
    [string]$LicensePolicyPath = "",
    [ValidateRange(1, 300)]
    [int]$TargetGrass = 30,
    [ValidateRange(1, 300)]
    [int]$TargetShrub = 30,
    [ValidateRange(1, 300)]
    [int]$TargetTree = 30,
    [ValidateRange(1, 300)]
    [int]$TargetUnknown = 40,
    [ValidateRange(1, 50)]
    [int]$HoldoutDHashDistance = 6,
    [ValidateRange(0, 20)]
    [int]$CandidateDHashDistance = 2,
    [ValidateRange(1, 10)]
    [int]$MaxApiBatchesPerQuery = 4,
    [ValidateRange(1, 100)]
    [int]$MaxPerSourcePlan = 40,
    [ValidateRange(1, 100)]
    [int]$MaxPerCreatorGroup = 12,
    [ValidateRange(1, 100)]
    [int]$MaxPerUnknownHint = 40,
    [ValidateRange(1, 100)]
    [int]$MinPerUnknownHint = 40,
    [switch]$FinalizeOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ScriptPath = $MyInvocation.MyCommand.Path
$AdventureXRoot = (Resolve-Path (Join-Path (Split-Path $ScriptPath -Parent) "../..")).Path
if (-not $Output) {
    $Output = Join-Path $AdventureXRoot "datasets/desert_plants_wikimedia_staging_e0"
}
if (-not $ExistingDataset) {
    $ExistingDataset = Join-Path $AdventureXRoot "datasets/desert_plants_v1"
}
if (-not $LicensePolicyPath) {
    $LicensePolicyPath = Join-Path (Split-Path $ScriptPath -Parent) 'wikimedia_license_policy_v1.json'
}
$Output = [IO.Path]::GetFullPath($Output)
$ExistingDataset = [IO.Path]::GetFullPath($ExistingDataset)
$LicensePolicyPath = [IO.Path]::GetFullPath($LicensePolicyPath)

if (-not (Test-Path -LiteralPath $LicensePolicyPath -PathType Leaf)) {
    throw "Shared license policy not found: $LicensePolicyPath"
}
$script:LicensePolicySha256 = (Get-FileHash -LiteralPath $LicensePolicyPath -Algorithm SHA256).Hash.ToLowerInvariant()
$script:LicensePolicy = Get-Content -LiteralPath $LicensePolicyPath -Raw -Encoding utf8 | ConvertFrom-Json
if ([string]$script:LicensePolicy.schema_version -cne 'rootscope.wikimedia_license_policy.v1') {
    throw "Unsupported shared license policy schema: $($script:LicensePolicy.schema_version)"
}
if ([string]$script:LicensePolicy.matching.comparison -cne 'ordinal_case_sensitive' -or
    [bool]$script:LicensePolicy.matching.trim_whitespace -or
    [bool]$script:LicensePolicy.matching.normalize_trailing_slash -or
    [string]$script:LicensePolicy.matching.unknown_binding_action -cne 'REJECT') {
    throw 'Shared license policy must require exact ordinal pair matching with REJECT fallback'
}

$ApiUrl = "https://commons.wikimedia.org/w/api.php"
$UserAgent = "RootScopeAdventureX/2.0 (educational dataset provenance audit; xiaomiju-x@users.noreply.github.com)"
$AllowedMime = @($script:LicensePolicy.image_constraints.allowed_mime | ForEach-Object { [string]$_ })
$MinimumOriginalSide = [int]$script:LicensePolicy.image_constraints.minimum_original_side
$MinimumDownloadedSide = [int]$script:LicensePolicy.image_constraints.minimum_downloaded_side
$DHashAlgorithm = [string]$script:LicensePolicy.image_constraints.dhash_algorithm
if ($MinimumOriginalSide -ne 720 -or $MinimumDownloadedSide -ne 448 -or
    $DHashAlgorithm -cne 'rootscope_rgb_center_sample_9x8_v1') {
    throw 'Unexpected image constraints in shared license policy'
}
if ($MinPerUnknownHint -gt $MaxPerUnknownHint) { throw 'MinPerUnknownHint cannot exceed MaxPerUnknownHint' }
$RequiredUnknownHints = @(
    'negative:bare_sand',
    'negative:rocks',
    'negative:blank_card',
    'negative:hand',
    'negative:irrelevant_object',
    'negative:other_plant_form'
)

$SourcePlan = @(
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Stipagrostis'; SpeciesHint='Stipagrostis spp.' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Stipagrostis plumosa'; SpeciesHint='Stipagrostis plumosa' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Cenchrus ciliaris'; SpeciesHint='Cenchrus ciliaris' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Cenchrus divisus'; SpeciesHint='Cenchrus divisus' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Pennisetum setaceum'; SpeciesHint='Cenchrus setaceus (syn. Pennisetum setaceum)' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Muhlenbergia porteri'; SpeciesHint='Muhlenbergia porteri' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Sporobolus airoides'; SpeciesHint='Sporobolus airoides' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Achnatherum hymenoides'; SpeciesHint='Achnatherum hymenoides' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Bouteloua eriopoda'; SpeciesHint='Bouteloua eriopoda' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Panicum turgidum'; SpeciesHint='Panicum turgidum' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Ammophila arenaria'; SpeciesHint='Ammophila arenaria' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Leymus arenarius'; SpeciesHint='Leymus arenarius' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Spinifex sericeus'; SpeciesHint='Spinifex sericeus' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Aristida adscensionis'; SpeciesHint='Aristida adscensionis' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Aristida pungens'; SpeciesHint='Stipagrostis pungens (syn. Aristida pungens)' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Pleuraphis rigida'; SpeciesHint='Pleuraphis rigida' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Hilaria rigida'; SpeciesHint='Pleuraphis rigida (syn. Hilaria rigida)' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Eragrostis lehmanniana'; SpeciesHint='Eragrostis lehmanniana' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Poa secunda'; SpeciesHint='Poa secunda' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Nassella pulchra'; SpeciesHint='Nassella pulchra' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Aristida purpurea'; SpeciesHint='Aristida purpurea' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Pleuraphis mutica'; SpeciesHint='Pleuraphis mutica' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Bouteloua gracilis'; SpeciesHint='Bouteloua gracilis' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Sporobolus cryptandrus'; SpeciesHint='Sporobolus cryptandrus' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Sporobolus flexuosus'; SpeciesHint='Sporobolus flexuosus' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Hesperostipa comata'; SpeciesHint='Hesperostipa comata' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Elymus elymoides'; SpeciesHint='Elymus elymoides' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='category'; Query='Dasyochloa pulchella'; SpeciesHint='Dasyochloa pulchella' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='search'; Query='desert grass tussock whole plant'; SpeciesHint='desert grass morphology' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='search'; Query='desert bunchgrass whole plant'; SpeciesHint='desert bunchgrass morphology' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='search'; Query='arid bunch grass tussock'; SpeciesHint='arid bunchgrass morphology' },
    [pscustomobject]@{ ClassId='grass_clump'; Mode='search'; Query='sand dune grass clump whole plant'; SpeciesHint='dune grass morphology' },

    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Larrea tridentata'; SpeciesHint='Larrea tridentata' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Artemisia tridentata'; SpeciesHint='Artemisia tridentata' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Haloxylon ammodendron'; SpeciesHint='Haloxylon ammodendron' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Atriplex canescens'; SpeciesHint='Atriplex canescens' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Haloxylon persicum'; SpeciesHint='Haloxylon persicum' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Calligonum comosum'; SpeciesHint='Calligonum comosum' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Atriplex halimus'; SpeciesHint='Atriplex halimus' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Artemisia herba-alba'; SpeciesHint='Artemisia herba-alba' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Leptadenia pyrotechnica'; SpeciesHint='Leptadenia pyrotechnica' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Ephedra alata'; SpeciesHint='Ephedra alata' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Rhanterium epapposum'; SpeciesHint='Rhanterium epapposum' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Zygophyllum dumosum'; SpeciesHint='Zygophyllum dumosum' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Ambrosia dumosa'; SpeciesHint='Ambrosia dumosa' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Encelia farinosa'; SpeciesHint='Encelia farinosa' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Atriplex confertifolia'; SpeciesHint='Atriplex confertifolia' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Ephedra californica'; SpeciesHint='Ephedra californica' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Krascheninnikovia lanata'; SpeciesHint='Krascheninnikovia lanata' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='category'; Query='Coleogyne ramosissima'; SpeciesHint='Coleogyne ramosissima' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='search'; Query='desert low shrub whole plant'; SpeciesHint='desert shrub morphology' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='search'; Query='sand desert bush isolated whole plant'; SpeciesHint='desert shrub morphology' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='search'; Query='arid shrub whole plant'; SpeciesHint='arid shrub morphology' },
    [pscustomobject]@{ ClassId='low_shrub'; Mode='search'; Query='xerophytic shrub whole plant'; SpeciesHint='xerophytic shrub morphology' },

    [pscustomobject]@{ ClassId='young_tree'; Mode='category'; Query='Vachellia tortilis'; SpeciesHint='Vachellia tortilis' },
    [pscustomobject]@{ ClassId='young_tree'; Mode='category'; Query='Acacia tortilis'; SpeciesHint='Vachellia tortilis (syn. Acacia tortilis)' },
    [pscustomobject]@{ ClassId='young_tree'; Mode='category'; Query='Prosopis cineraria'; SpeciesHint='Prosopis cineraria' },
    [pscustomobject]@{ ClassId='young_tree'; Mode='category'; Query='Populus euphratica'; SpeciesHint='Populus euphratica' },
    [pscustomobject]@{ ClassId='young_tree'; Mode='category'; Query='Tamarix aphylla'; SpeciesHint='Tamarix aphylla' },
    [pscustomobject]@{ ClassId='young_tree'; Mode='category'; Query='Vachellia erioloba'; SpeciesHint='Vachellia erioloba' },
    [pscustomobject]@{ ClassId='young_tree'; Mode='category'; Query='Vachellia nilotica'; SpeciesHint='Vachellia nilotica' },
    [pscustomobject]@{ ClassId='young_tree'; Mode='category'; Query='Senegalia senegal'; SpeciesHint='Senegalia senegal (syn. Acacia senegal)' },
    [pscustomobject]@{ ClassId='young_tree'; Mode='category'; Query='Prosopis juliflora'; SpeciesHint='Prosopis juliflora' },
    [pscustomobject]@{ ClassId='young_tree'; Mode='category'; Query='Parkinsonia aculeata'; SpeciesHint='Parkinsonia aculeata' },
    [pscustomobject]@{ ClassId='young_tree'; Mode='category'; Query='Boscia albitrunca'; SpeciesHint='Boscia albitrunca' },
    [pscustomobject]@{ ClassId='young_tree'; Mode='category'; Query='Balanites aegyptiaca'; SpeciesHint='Balanites aegyptiaca' },
    [pscustomobject]@{ ClassId='young_tree'; Mode='search'; Query='young desert tree whole plant'; SpeciesHint='young desert tree morphology' },
    [pscustomobject]@{ ClassId='young_tree'; Mode='search'; Query='desert tree sapling whole plant'; SpeciesHint='desert tree sapling morphology' },
    [pscustomobject]@{ ClassId='young_tree'; Mode='search'; Query='young acacia tree desert'; SpeciesHint='young acacia morphology' },
    [pscustomobject]@{ ClassId='young_tree'; Mode='search'; Query='tree seedling growing in sand'; SpeciesHint='tree seedling morphology' },

    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='bare sand desert surface close up'; SpeciesHint='negative:bare_sand' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='desert gravel rock surface close up'; SpeciesHint='negative:rocks' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='blank white card paper isolated'; SpeciesHint='negative:blank_card' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='blank sheet of white paper isolated'; SpeciesHint='negative:blank_card' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='blank index card isolated'; SpeciesHint='negative:blank_card' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='white card blank paper'; SpeciesHint='negative:blank_card' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Blank paper'; SpeciesHint='negative:blank_card' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Index cards'; SpeciesHint='negative:blank_card' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Sheets of paper'; SpeciesHint='negative:blank_card' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='human hand on sand'; SpeciesHint='negative:hand' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='shadow on desert sand'; SpeciesHint='negative:shadow' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='desert animal on sand'; SpeciesHint='negative:animal' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='plastic object on sand'; SpeciesHint='negative:irrelevant_object' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Cacti in habitat'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Sand'; SpeciesHint='negative:bare_sand' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Sand dunes'; SpeciesHint='negative:bare_sand' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Dunes'; SpeciesHint='negative:bare_sand' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Sand ripples'; SpeciesHint='negative:bare_sand' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Ergs'; SpeciesHint='negative:bare_sand' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='desert sand ripples no vegetation'; SpeciesHint='negative:bare_sand' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='empty sand dune surface texture'; SpeciesHint='negative:bare_sand' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Gravel'; SpeciesHint='negative:rocks' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Desert pavement'; SpeciesHint='negative:rocks' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Shadows'; SpeciesHint='negative:shadow' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Hands'; SpeciesHint='negative:hand' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Plastic bottles'; SpeciesHint='negative:irrelevant_object' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Cacti'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Opuntia in habitat'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Agave in habitat'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Yucca in habitat'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Aloes in habitat'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Desert wildflowers'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Carnegiea gigantea'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Opuntia ficus-indica'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Ferocactus wislizeni'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Agave americana'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Yucca brevifolia'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Aloe vera'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Echinocactus grusonii'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='category'; Query='Cylindropuntia bigelovii'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='desert cactus whole plant habitat'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='desert succulent whole plant habitat'; SpeciesHint='negative:other_plant_form' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='empty sand tray close up'; SpeciesHint='negative:bare_sand' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='stone object on sand close up'; SpeciesHint='negative:rocks' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='gloved hand on sand'; SpeciesHint='negative:hand' },
    [pscustomobject]@{ ClassId='unknown'; Mode='search'; Query='plastic cup bottle on sand'; SpeciesHint='negative:irrelevant_object' }
)

$TargetByClass = @{
    grass_clump = $TargetGrass
    low_shrub = $TargetShrub
    young_tree = $TargetTree
    unknown = $TargetUnknown
}

function ConvertTo-PlainText {
    param([AllowNull()][object]$Value)
    if ($null -eq $Value) { return "" }
    $text = [string]$Value
    $text = [regex]::Replace($text, '<[^>]+>', ' ')
    $text = [Net.WebUtility]::HtmlDecode($text)
    return ([regex]::Replace($text, '\s+', ' ')).Trim()
}

function Get-ObjectValue {
    param([AllowNull()][object]$Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function Get-MetaValue {
    param([AllowNull()][object]$Metadata, [string]$Name)
    if ($null -eq $Metadata) { return "" }
    $property = $Metadata.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) { return "" }
    $valueProperty = $property.Value.PSObject.Properties['value']
    if ($null -eq $valueProperty) { return "" }
    return ConvertTo-PlainText $valueProperty.Value
}

function Get-Sha256Bytes {
    param([byte[]]$Bytes)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($Bytes))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Get-Sha256Text {
    param([string]$Text)
    return Get-Sha256Bytes ([Text.Encoding]::UTF8.GetBytes($Text))
}

function Get-FileSha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-AtomicUtf8 {
    param([string]$Path, [string]$Content)
    $parent = Split-Path $Path -Parent
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
    $suffix = if ($script:RunId) { $script:RunId } else { [Guid]::NewGuid().ToString('N') }
    $temporary = "$Path.$suffix.tmp"
    [IO.File]::WriteAllText($temporary, $Content, [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Write-Event {
    param([string]$Event, [hashtable]$Fields = @{})
    $entry = [ordered]@{
        at_utc = [DateTime]::UtcNow.ToString('o')
        event = $Event
        run_id = $script:RunId
    }
    foreach ($key in ($Fields.Keys | Sort-Object)) { $entry[$key] = $Fields[$key] }
    Add-Content -LiteralPath $script:RecoveryLog -Value (($entry | ConvertTo-Json -Compress -Depth 10)) -Encoding utf8
}

function Invoke-WithRetry {
    param([scriptblock]$Action, [string]$Operation, [int]$Retries = 5)
    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        try { return & $Action }
        catch {
            Write-Event 'network_attempt_failed' @{ operation=$Operation; attempt=$attempt; error=$_.Exception.Message }
            if ($attempt -eq $Retries) { throw }
            Start-Sleep -Milliseconds ([Math]::Min(12000, 700 * [Math]::Pow(2, $attempt - 1)))
        }
    }
}

function Invoke-CommonsApi {
    param([hashtable]$Parameters)
    $pairs = foreach ($key in ($Parameters.Keys | Sort-Object)) {
        "{0}={1}" -f [Net.WebUtility]::UrlEncode([string]$key), [Net.WebUtility]::UrlEncode([string]$Parameters[$key])
    }
    $uri = "${ApiUrl}?" + ($pairs -join '&')
    return Invoke-WithRetry -Operation "commons_api" -Action {
        Invoke-RestMethod -Uri $uri -Headers @{ 'User-Agent'=$UserAgent; 'Accept'='application/json' } -TimeoutSec 60
    }
}

function Get-CommonsPages {
    param([string]$Mode, [string]$Query)
    $common = @{
        action='query'; format='json'; formatversion='2'; prop='imageinfo'
        iiprop='url|mime|size|sha1|extmetadata'; iiurlwidth='1280'
    }
    if ($Mode -eq 'category') {
        $common.generator='categorymembers'; $common.gcmtitle="Category:$Query"
        $common.gcmtype='file'; $common.gcmlimit='50'
    } else {
        $common.generator='search'; $common.gsrnamespace='6'
        $common.gsrlimit='50'; $common.gsrsearch=$Query
    }
    $allPages = [Collections.Generic.List[object]]::new()
    $continuation = @{}
    for ($batch = 1; $batch -le $MaxApiBatchesPerQuery; $batch++) {
        $request = @{}
        foreach ($key in $common.Keys) { $request[$key] = $common[$key] }
        foreach ($key in $continuation.Keys) { $request[$key] = $continuation[$key] }

        $response = Invoke-CommonsApi $request
        $queryObject = Get-ObjectValue $response 'query'
        $pages = Get-ObjectValue $queryObject 'pages'
        if ($null -ne $pages) {
            foreach ($page in @($pages)) { [void]$allPages.Add($page) }
        }

        $continueObject = Get-ObjectValue $response 'continue'
        if ($null -eq $continueObject) { break }
        $continuation = @{}
        foreach ($property in $continueObject.PSObject.Properties) {
            $continuation[$property.Name] = $property.Value
        }
        if ($continuation.Count -eq 0) { break }
    }
    return @($allPages)
}

function Resolve-LicenseDecision {
    param(
        [string]$RawName,
        [AllowEmptyString()][string]$RawUrl,
        [string]$Copyrighted,
        [string]$Context,
        [string]$SourceProvider,
        [int64]$PageId,
        [string]$SourceGroup
    )
    foreach ($license in @($script:LicensePolicy.licenses)) {
        foreach ($binding in @($license.raw_bindings)) {
            if ([string]$binding.raw_name -cne $RawName -or [string]$binding.raw_url -cne $RawUrl) { continue }
            $falseRaw = Get-ObjectValue $license 'copyrighted_exact_values'
            $falseValues = [Collections.Generic.List[string]]::new()
            if ($null -ne $falseRaw) { foreach ($value in @($falseRaw)) { $falseValues.Add([string]$value) } }
            if ($falseValues.Count -gt 0 -and $falseValues -cnotcontains $Copyrighted) { return $null }
            return [pscustomobject]@{
                BindingId = "policy:$($license.canonical_id):$RawName|$RawUrl"
                CanonicalId = [string]$license.canonical_id
                CanonicalName = [string]$license.canonical_name
                CanonicalUrl = [string]$license.canonical_url
                IsLegacyException = $false
            }
        }
    }
    foreach ($exception in @($script:LicensePolicy.legacy_exceptions)) {
        if ([string]$exception.context -cne $Context -or
            [string]$exception.source_provider -cne $SourceProvider -or
            [int64]$exception.pageid -ne $PageId -or
            [string]$exception.source_group -cne $SourceGroup -or
            [string]$exception.raw_name -cne $RawName -or
            [string]$exception.raw_url -cne $RawUrl) { continue }
        $license = @($script:LicensePolicy.licenses | Where-Object { [string]$_.canonical_id -ceq [string]$exception.canonical_id })
        if ($license.Count -ne 1) { throw "Policy exception references unknown canonical license: $($exception.exception_id)" }
        return [pscustomobject]@{
            BindingId = "exception:$($exception.exception_id)"
            CanonicalId = [string]$license[0].canonical_id
            CanonicalName = [string]$license[0].canonical_name
            CanonicalUrl = [string]$license[0].canonical_url
            IsLegacyException = $true
        }
    }
    return $null
}

function Get-ImageFacts {
    param([string]$Path)
    Add-Type -AssemblyName System.Drawing
    $bitmap = [Drawing.Bitmap]::new($Path)
    try {
        $mime = if ($bitmap.RawFormat.Guid -eq [Drawing.Imaging.ImageFormat]::Jpeg.Guid) {
            'image/jpeg'
        } elseif ($bitmap.RawFormat.Guid -eq [Drawing.Imaging.ImageFormat]::Png.Guid) {
            'image/png'
        } else {
            'unsupported'
        }
        $bits = [Text.StringBuilder]::new(64)
        for ($y = 0; $y -lt 8; $y++) {
            $sourceY = [Math]::Min($bitmap.Height - 1, [int][Math]::Floor(($y + 0.5) * $bitmap.Height / 8.0))
            for ($x = 0; $x -lt 8; $x++) {
                $leftX = [Math]::Min($bitmap.Width - 1, [int][Math]::Floor(($x + 0.5) * $bitmap.Width / 9.0))
                $rightX = [Math]::Min($bitmap.Width - 1, [int][Math]::Floor(($x + 1.5) * $bitmap.Width / 9.0))
                $left = $bitmap.GetPixel($leftX, $sourceY)
                $right = $bitmap.GetPixel($rightX, $sourceY)
                $leftLuma = 299 * $left.R + 587 * $left.G + 114 * $left.B
                $rightLuma = 299 * $right.R + 587 * $right.G + 114 * $right.B
                [void]$bits.Append($(if ($leftLuma -gt $rightLuma) { '1' } else { '0' }))
            }
        }
        $binary = $bits.ToString()
        $hex = [Text.StringBuilder]::new(16)
        for ($offset = 0; $offset -lt 64; $offset += 4) {
            [void]$hex.Append(([Convert]::ToInt32($binary.Substring($offset, 4), 2)).ToString('x'))
        }
        return [pscustomobject]@{
            Width = $bitmap.Width
            Height = $bitmap.Height
            Mime = $mime
            DHash64 = $hex.ToString()
        }
    } finally { $bitmap.Dispose() }
}

function Get-DHash64 {
    param([string]$Path)
    return (Get-ImageFacts $Path).DHash64
}

function Get-HammingDistance {
    param([string]$Left, [string]$Right)
    if ($Left.Length -ne 16 -or $Right.Length -ne 16) { return 64 }
    $distance = 0
    for ($index = 0; $index -lt 16; $index++) {
        $xor = [Convert]::ToInt32($Left.Substring($index, 1), 16) -bxor [Convert]::ToInt32($Right.Substring($index, 1), 16)
        while ($xor -ne 0) { $distance += $xor -band 1; $xor = $xor -shr 1 }
    }
    return $distance
}

function Read-JsonLines {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return @() }
    $records = @()
    foreach ($line in [IO.File]::ReadLines($Path)) {
        if ($line.Trim()) { $records += ($line | ConvertFrom-Json) }
    }
    return @($records)
}

function Set-RecordProperty {
    param([object]$Record, [string]$Name, [AllowNull()][object]$Value)
    if ($null -eq $Record.PSObject.Properties[$Name]) {
        $Record | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
    } else {
        $Record.$Name = $Value
    }
}

function Resolve-DatasetFile {
    param([string]$Root, [string]$RelativePath, [string]$Context)
    if (-not $RelativePath -or $RelativePath.Contains('\') -or [IO.Path]::IsPathRooted($RelativePath)) {
        throw "$Context has unsafe filename: $RelativePath"
    }
    $rootFull = [IO.Path]::GetFullPath($Root).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $candidate = [IO.Path]::GetFullPath((Join-Path $Root $RelativePath))
    if (-not $candidate.StartsWith($rootFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Context filename escapes dataset root: $RelativePath"
    }
    return $candidate
}

function Normalize-CandidateRecord {
    param([object]$Record)
    $pageId = [int64]$Record.pageid
    $identity = "candidate pageid=$pageId"
    $sourceGroup = [string]$Record.source_group
    $expectedCreatorGroup = 'commons-creator:' + (Get-Sha256Text ([string]$Record.artist)).Substring(0,16)
    if ($null -ne $Record.PSObject.Properties['creator_group'] -and [string]$Record.creator_group -cne $expectedCreatorGroup) {
        throw "$identity creator_group does not match the exact normalized artist hash"
    }
    $rawName = if ($null -ne $Record.PSObject.Properties['license_raw_name']) { [string]$Record.license_raw_name } else { [string]$Record.license }
    $rawUrl = if ($null -ne $Record.PSObject.Properties['license_raw_url']) { [string]$Record.license_raw_url } else { [string]$Record.license_url }
    if ($null -ne $Record.PSObject.Properties['license_raw_name'] -and [string]$Record.license -cne $rawName) {
        throw "$identity license and license_raw_name disagree"
    }
    if ($null -ne $Record.PSObject.Properties['license_raw_url'] -and [string]$Record.license_url -cne $rawUrl) {
        throw "$identity license_url and license_raw_url disagree"
    }
    $decision = Resolve-LicenseDecision -RawName $rawName -RawUrl $rawUrl -Copyrighted ([string]$Record.copyrighted) `
        -Context 'staging_manifest' -SourceProvider ([string]$Record.source_provider) -PageId $pageId -SourceGroup $sourceGroup
    if ($null -eq $decision) { throw "$identity raw license pair is not exactly allowlisted" }
    if ($null -ne $Record.PSObject.Properties['license_canonical_name'] -and [string]$Record.license_canonical_name -cne $decision.CanonicalName) {
        throw "$identity canonical license name mismatch"
    }
    if ($null -ne $Record.PSObject.Properties['license_canonical_url'] -and [string]$Record.license_canonical_url -cne $decision.CanonicalUrl) {
        throw "$identity canonical license URL mismatch"
    }

    $path = Resolve-DatasetFile -Root $Output -RelativePath ([string]$Record.filename) -Context $identity
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "$identity references missing file: $path" }
    $actualSha = Get-FileSha256 $path
    if ($actualSha -cne ([string]$Record.download_sha256).ToLowerInvariant()) { throw "$identity SHA-256 mismatch: $path" }
    $facts = Get-ImageFacts $path
    if ($AllowedMime -cnotcontains $facts.Mime) { throw "$identity decoded MIME is not allowlisted: $($facts.Mime)" }
    if ([string]$Record.mime -cne $facts.Mime) { throw "$identity metadata/decoded MIME mismatch" }
    if ([Math]::Min([int]$Record.original_width, [int]$Record.original_height) -lt $MinimumOriginalSide) {
        throw "$identity original metadata side is below $MinimumOriginalSide"
    }
    if ([Math]::Min($facts.Width, $facts.Height) -lt $MinimumDownloadedSide) {
        throw "$identity downloaded side is below $MinimumDownloadedSide"
    }
    if ($null -ne $Record.PSObject.Properties['dhash64_algorithm']) {
        if ([string]$Record.dhash64_algorithm -cne $DHashAlgorithm -or [string]$Record.dhash64 -cne $facts.DHash64) {
            throw "$identity portable dHash mismatch"
        }
    } else {
        Set-RecordProperty $Record 'legacy_dhash64_gdi_v1' ([string]$Record.dhash64)
    }

    Set-RecordProperty $Record 'license_raw_name' $rawName
    Set-RecordProperty $Record 'license_raw_url' $rawUrl
    Set-RecordProperty $Record 'license_canonical_id' $decision.CanonicalId
    Set-RecordProperty $Record 'license_canonical_name' $decision.CanonicalName
    Set-RecordProperty $Record 'license_canonical_url' $decision.CanonicalUrl
    Set-RecordProperty $Record 'license_binding_id' $decision.BindingId
    Set-RecordProperty $Record 'license_policy_sha256' $script:LicensePolicySha256
    Set-RecordProperty $Record 'creator_group' $expectedCreatorGroup
    Set-RecordProperty $Record 'download_width' $facts.Width
    Set-RecordProperty $Record 'download_height' $facts.Height
    Set-RecordProperty $Record 'download_mime' $facts.Mime
    Set-RecordProperty $Record 'dhash64_algorithm' $DHashAlgorithm
    Set-RecordProperty $Record 'dhash64' $facts.DHash64
    return $Record
}

function Get-OverlapCounts {
    param([object[]]$Records)
    $pageIds = [Collections.Generic.HashSet[int64]]::new()
    $groups = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $commons = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $sha256 = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($item in $script:ExistingRecords) {
        [void]$pageIds.Add([int64]$item.pageid)
        [void]$groups.Add([string]$item.source_group)
        if ($item.commons_sha1) { [void]$commons.Add([string]$item.commons_sha1) }
        if ($item.download_sha256) { [void]$sha256.Add([string]$item.download_sha256) }
    }
    return [ordered]@{
        pageid = @($Records | Where-Object { $pageIds.Contains([int64]$_.pageid) }).Count
        source_group = @($Records | Where-Object { $groups.Contains([string]$_.source_group) }).Count
        commons_sha1 = @($Records | Where-Object { $commons.Contains([string]$_.commons_sha1) }).Count
        download_sha256 = @($Records | Where-Object { $sha256.Contains([string]$_.download_sha256) }).Count
    }
}

function Test-AcquisitionTargetMet {
    param([object[]]$Records, [string]$ClassId)
    $classCount = @($Records | Where-Object { [string]$_.class_id -ceq $ClassId }).Count
    if ($classCount -lt [int]$TargetByClass[$ClassId]) { return $false }
    if ($ClassId -cne 'unknown') { return $true }
    foreach ($hint in $RequiredUnknownHints) {
        $hintCount = @($Records | Where-Object { [string]$_.class_id -ceq 'unknown' -and [string]$_.species_hint -ceq $hint }).Count
        if ($hintCount -lt $MinPerUnknownHint) { return $false }
    }
    return $true
}

function Save-Outputs {
    param([object[]]$Records)
    foreach ($record in $Records) {
        if ($null -eq $record.PSObject.Properties['candidate_label_status']) {
            $record | Add-Member -NotePropertyName candidate_label_status -NotePropertyValue 'query_or_category_derived_unverified'
        }
        if ($null -eq $record.PSObject.Properties['species_hint_status']) {
            $record | Add-Member -NotePropertyName species_hint_status -NotePropertyValue 'acquisition_hint_not_a_reviewed_species_or_shape_label'
        }
    }
    $orderedRecords = @($Records | Sort-Object class_id, pageid)
    $manifestLines = foreach ($record in $orderedRecords) { $record | ConvertTo-Json -Compress -Depth 20 }
    Write-AtomicUtf8 (Join-Path $Output 'manifest.jsonl') (($manifestLines -join "`n") + $(if (@($manifestLines).Count) { "`n" } else { "" }))

    $classCounts = [ordered]@{}
    $licenseCounts = [ordered]@{}
    foreach ($classId in @('grass_clump','low_shrub','young_tree','unknown')) {
        $classCounts[$classId] = @($orderedRecords | Where-Object class_id -eq $classId).Count
    }
    foreach ($name in @($orderedRecords | ForEach-Object license_canonical_name | Sort-Object -Unique)) {
        $licenseCounts[$name] = @($orderedRecords | Where-Object license_canonical_name -ceq $name).Count
    }
    $targetsMet = $true
    foreach ($classId in $TargetByClass.Keys) { if (-not (Test-AcquisitionTargetMet $orderedRecords $classId)) { $targetsMet = $false } }
    $manifestPath = Join-Path $Output 'manifest.jsonl'
    $overlaps = Get-OverlapCounts $orderedRecords
    $sourcePlanCounts = @($orderedRecords | Group-Object class_id,acquisition_mode,acquisition_query | ForEach-Object {
        [ordered]@{ class_id=[string]$_.Group[0].class_id; acquisition_mode=[string]$_.Group[0].acquisition_mode; acquisition_query=[string]$_.Group[0].acquisition_query; count=$_.Count }
    } | Sort-Object class_id,acquisition_mode,acquisition_query)
    $hintCounts = @($orderedRecords | Group-Object class_id,species_hint | ForEach-Object {
        [ordered]@{ class_id=[string]$_.Group[0].class_id; species_hint=[string]$_.Group[0].species_hint; count=$_.Count }
    } | Sort-Object class_id,species_hint)
    $creatorCounts = @($orderedRecords | Group-Object creator_group | ForEach-Object {
        [ordered]@{ creator_group=[string]$_.Name; artist=[string]$_.Group[0].artist; count=$_.Count }
    } | Sort-Object @{Expression='count';Descending=$true},creator_group)
    $summary = [ordered]@{
        schema_version='rootscope.wikimedia_candidate_staging.v1'
        status='STAGING_CANDIDATES_MANUAL_VISUAL_AND_LICENSE_REVIEW_REQUIRED_NOT_TRAIN_READY'
        generated_at_utc=[DateTime]::UtcNow.ToString('o')
        provider='Wikimedia Commons'
        total=$orderedRecords.Count
        class_counts=$classCounts
        requested_targets=[ordered]@{ grass_clump=$TargetGrass; low_shrub=$TargetShrub; young_tree=$TargetTree; unknown=$TargetUnknown }
        acquisition_targets_met=$targetsMet
        license_counts=$licenseCounts
        review_status='pending_human_visual_and_license_review'
        split_status='UNASSIGNED_DO_NOT_TRAIN'
        manifest_sha256=(Get-FileSha256 $manifestPath)
        collector_script_sha256=$script:CollectorScriptSha256
        license_policy_sha256=$script:LicensePolicySha256
        license_policy_file=[IO.Path]::GetFileName($LicensePolicyPath)
        image_constraints=[ordered]@{
            allowed_mime=@($AllowedMime)
            minimum_original_side=$MinimumOriginalSide
            minimum_downloaded_side=$MinimumDownloadedSide
            dhash_algorithm=$DHashAlgorithm
        }
        existing_overlap_counts=$overlaps
        source_group_overlap_with_existing=$overlaps.source_group
        exact_content_overlap_with_existing=$overlaps.download_sha256
        permanent_print_holdout_count=$script:PermanentHoldouts.Count
        permanent_print_holdout_pageids=@($script:PermanentHoldouts | ForEach-Object { [int64]$_.pageid } | Sort-Object)
        quarantine_receipt_count=@($script:QuarantineReceipts).Count
        quarantine_receipts=@($script:QuarantineReceipts)
        holdout_dhash_rejection_threshold=$HoldoutDHashDistance
        candidate_dhash_rejection_threshold=$CandidateDHashDistance
        api_batches_per_query_limit=$MaxApiBatchesPerQuery
        diversity_caps=[ordered]@{
            max_per_source_plan=$MaxPerSourcePlan
            max_per_creator_group=$MaxPerCreatorGroup
            max_per_unknown_species_hint=$MaxPerUnknownHint
            existing_over_cap_action='KEEP_UNREVIEWED_AND_REQUIRE_HUMAN_MERGE_OR_REJECT; DO_NOT AUTO_DELETE'
        }
        unknown_hint_requirements=[ordered]@{
            required_hints=@($RequiredUnknownHints)
            minimum_per_hint=$MinPerUnknownHint
            policy='TOTAL_TARGET_AND_ALL_SIX_REQUIRED_HINT_MINIMUMS'
            counts=@($RequiredUnknownHints | ForEach-Object {
                $hint=$_; [ordered]@{ species_hint=$hint; count=@($orderedRecords | Where-Object { [string]$_.class_id -ceq 'unknown' -and [string]$_.species_hint -ceq $hint }).Count; met=(@($orderedRecords | Where-Object { [string]$_.class_id -ceq 'unknown' -and [string]$_.species_hint -ceq $hint }).Count -ge $MinPerUnknownHint) }
            })
            unknown_hint_minimums_met=@($RequiredUnknownHints | Where-Object {
                $hint=$_; @($orderedRecords | Where-Object { [string]$_.class_id -ceq 'unknown' -and [string]$_.species_hint -ceq $hint }).Count -ge $MinPerUnknownHint
            }).Count -eq $RequiredUnknownHints.Count
        }
        diversity_counts=[ordered]@{
            by_source_plan=$sourcePlanCounts
            by_species_hint=$hintCounts
            by_creator_group=$creatorCounts
            source_plans_over_cap=@($sourcePlanCounts | Where-Object { $_.count -gt $MaxPerSourcePlan })
            creators_over_cap=@($creatorCounts | Where-Object { $_.count -gt $MaxPerCreatorGroup })
            unknown_hints_over_cap=@($hintCounts | Where-Object { $_.class_id -ceq 'unknown' -and $_.count -gt $MaxPerUnknownHint })
        }
        collector_run_id=$script:RunId
        legal_note='Commons metadata is machine-screened, not a warranty. A human must re-open each file page and verify attribution, license, personality/privacy and other rights before training or publication.'
    }
    Write-AtomicUtf8 (Join-Path $Output 'summary.json') (($summary | ConvertTo-Json -Depth 20) + "`n")

    $attribution = [Collections.Generic.List[string]]::new()
    $attribution.Add('# RootScope Wikimedia staging attribution ledger')
    $attribution.Add('')
    $attribution.Add('Status: machine-screened candidate metadata; human re-review required before use. This is not a training split.')
    $attribution.Add('')
    foreach ($record in $orderedRecords) {
        $licenseText = $record.license_canonical_name
        if ($record.license_canonical_url) { $licenseText = "[$($record.license_canonical_name)]($($record.license_canonical_url))" }
        $attribution.Add("- ``$($record.filename)`` — $($record.artist) — [$($record.title)]($($record.source_page)) — $licenseText — source group ``$($record.source_group)``")
    }
    Write-AtomicUtf8 (Join-Path $Output 'ATTRIBUTION.md') (($attribution -join "`n") + "`n")

    $readme = @"
# RootScope Wikimedia candidate staging E0

Status: ``STAGING_CANDIDATES_MANUAL_VISUAL_AND_LICENSE_REVIEW_REQUIRED_NOT_TRAIN_READY``

This directory is an acquisition pool, not a dataset split. Every record is ``UNASSIGNED_DO_NOT_TRAIN`` and still needs human visual and rights review. No image here may enter training merely because the acquisition target was reached.

The collector accepts only an exact allowlist of CC BY, CC BY-SA, CC0 and Public Domain labels exposed by the Wikimedia Commons file API. It records the file page, creator/credit, license name and URL, Commons SHA-1, local SHA-256, dHash, access time and ``source_group``. It excludes every page/hash already present in ``desert_plants_v1`` and separately blocks dHash-near matches to the six permanent print holdouts.

Wikimedia Commons explains that individual files have different reuse conditions and that reusers remain responsible for verifying copyright and other applicable rights: [Commons reuse guidance](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia/en) and [Commons licensing policy](https://commons.wikimedia.org/wiki/Commons:Licensing/en).

Resume the same acquisition safely from the AdventureX root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/dataset/collect_wikimedia_candidates.ps1
```

Files:

- ``manifest.jsonl``: one immutable-provenance candidate record per downloaded image;
- ``summary.json``: counts and explicit non-ready status;
- ``ATTRIBUTION.md``: human-readable attribution ledger;
- ``recovery_log.jsonl``: append-only acquisition, rejection and network retry events;
- ``images/<class_id>/``: downloaded 1280-pixel Commons derivatives, pending review.
"@
    Write-AtomicUtf8 (Join-Path $Output 'README.md') ($readme + "`n")
}

New-Item -ItemType Directory -Path $Output -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Output 'images') -Force | Out-Null
$script:RunId = [Guid]::NewGuid().ToString('N')
$script:CollectorScriptSha256 = Get-FileSha256 $ScriptPath
$lockPath = Join-Path $Output '.collector.lock'
try {
    $script:CollectorLock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
} catch {
    throw "Another collector owns the staging lock: $lockPath"
}
try {
$lockReceipt = [ordered]@{ run_id=$script:RunId; pid=$PID; started_at_utc=[DateTime]::UtcNow.ToString('o'); script_sha256=$script:CollectorScriptSha256; policy_sha256=$script:LicensePolicySha256 }
$lockBytes = [Text.Encoding]::UTF8.GetBytes(($lockReceipt | ConvertTo-Json -Compress))
$script:CollectorLock.SetLength(0)
[void]$script:CollectorLock.Write($lockBytes, 0, $lockBytes.Length)
$script:CollectorLock.Flush($true)

$script:RecoveryLog = Join-Path $Output 'recovery_log.jsonl'
$script:QuarantineReceipts = @()
if (-not (Test-Path -LiteralPath $script:RecoveryLog)) {
    [IO.File]::WriteAllText($script:RecoveryLog, '', [Text.UTF8Encoding]::new($false))
}

$existingManifestPath = Join-Path $ExistingDataset 'manifest.jsonl'
if (-not (Test-Path -LiteralPath $existingManifestPath)) {
    throw "Existing dataset manifest not found: $existingManifestPath"
}
$existing = @(Read-JsonLines $existingManifestPath)
$script:ExistingRecords = $existing
$script:PermanentHoldouts = @($existing | Where-Object { [string]$_.split -ceq 'print_demo' -or [string]$_.domain -ceq 'print_demo_source' })
$expectedHoldoutPageIds = @(133271396,75559442,2738023,4424728,5445424,6021614 | Sort-Object)
$actualHoldoutPageIds = @($script:PermanentHoldouts | ForEach-Object { [int64]$_.pageid } | Sort-Object)
if ($actualHoldoutPageIds.Count -ne $expectedHoldoutPageIds.Count -or
    @(Compare-Object -ReferenceObject $expectedHoldoutPageIds -DifferenceObject $actualHoldoutPageIds).Count -ne 0) {
    throw "Permanent holdout identity set mismatch: $($actualHoldoutPageIds -join ',')"
}

$priorSummaryPath = Join-Path $Output 'summary.json'
if ($FinalizeOnly -and -not (Test-Path -LiteralPath $priorSummaryPath -PathType Leaf)) { throw "FinalizeOnly requires existing summary: $priorSummaryPath" }
if (Test-Path -LiteralPath $priorSummaryPath -PathType Leaf) {
    $priorSummary = Get-Content -LiteralPath $priorSummaryPath -Raw -Encoding utf8 | ConvertFrom-Json
    $sealedPrior = $null -ne $priorSummary.PSObject.Properties['manifest_sha256']
    if ($FinalizeOnly -or $sealedPrior) {
        if ($FinalizeOnly -or -not $PSBoundParameters.ContainsKey('TargetGrass')) { $TargetGrass = [int]$priorSummary.requested_targets.grass_clump }
        if ($FinalizeOnly -or -not $PSBoundParameters.ContainsKey('TargetShrub')) { $TargetShrub = [int]$priorSummary.requested_targets.low_shrub }
        if ($FinalizeOnly -or -not $PSBoundParameters.ContainsKey('TargetTree')) { $TargetTree = [int]$priorSummary.requested_targets.young_tree }
        if ($FinalizeOnly -or -not $PSBoundParameters.ContainsKey('TargetUnknown')) { $TargetUnknown = [int]$priorSummary.requested_targets.unknown }
        if ($FinalizeOnly -or -not $PSBoundParameters.ContainsKey('HoldoutDHashDistance')) { $HoldoutDHashDistance = [int]$priorSummary.holdout_dhash_rejection_threshold }
        if ($FinalizeOnly -or -not $PSBoundParameters.ContainsKey('CandidateDHashDistance')) { $CandidateDHashDistance = [int]$priorSummary.candidate_dhash_rejection_threshold }
        if ($FinalizeOnly -or -not $PSBoundParameters.ContainsKey('MaxApiBatchesPerQuery')) { $MaxApiBatchesPerQuery = [int]$priorSummary.api_batches_per_query_limit }
        if ($null -ne $priorSummary.PSObject.Properties['unknown_hint_requirements'] -and ($FinalizeOnly -or -not $PSBoundParameters.ContainsKey('MinPerUnknownHint'))) { $MinPerUnknownHint = [int]$priorSummary.unknown_hint_requirements.minimum_per_hint }
        if ($null -ne $priorSummary.PSObject.Properties['diversity_caps']) {
            if ($FinalizeOnly -or -not $PSBoundParameters.ContainsKey('MaxPerSourcePlan')) { $MaxPerSourcePlan = [int]$priorSummary.diversity_caps.max_per_source_plan }
            if ($FinalizeOnly -or -not $PSBoundParameters.ContainsKey('MaxPerCreatorGroup')) { $MaxPerCreatorGroup = [int]$priorSummary.diversity_caps.max_per_creator_group }
            if ($FinalizeOnly -or -not $PSBoundParameters.ContainsKey('MaxPerUnknownHint')) { $MaxPerUnknownHint = [int]$priorSummary.diversity_caps.max_per_unknown_species_hint }
        }
        if ($MinPerUnknownHint -gt $MaxPerUnknownHint) { throw 'Prior summary has impossible unknown hint limits' }
        $TargetByClass = @{ grass_clump=$TargetGrass; low_shrub=$TargetShrub; young_tree=$TargetTree; unknown=$TargetUnknown }
    }
}

$records = @(Read-JsonLines (Join-Path $Output 'manifest.jsonl'))
foreach ($item in $records) { [void](Normalize-CandidateRecord $item) }

$pendingDir = Join-Path $Output 'pending'
if (Test-Path -LiteralPath $pendingDir -PathType Container) {
    foreach ($pendingFile in @(Get-ChildItem -LiteralPath $pendingDir -Filter '*.json' -File | Sort-Object Name)) {
        $pending = Get-Content -LiteralPath $pendingFile.FullName -Raw -Encoding utf8 | ConvertFrom-Json
        if ([string]$pending.schema_version -cne 'rootscope.wikimedia_candidate_pending.v1') {
            throw "Unsupported pending receipt: $($pendingFile.FullName)"
        }
        $pendingRecord = $pending.record
        $pageId = [int64]$pendingRecord.pageid
        $expectedSha = [string]$pending.download_sha256
        if ([string]$pendingRecord.download_sha256 -cne $expectedSha) { throw "Pending receipt SHA disagreement: $($pendingFile.FullName)" }
        $finalPath = Resolve-DatasetFile -Root $Output -RelativePath ([string]$pending.final_path) -Context "pending pageid=$pageId"
        $temporaryPath = Resolve-DatasetFile -Root $Output -RelativePath ([string]$pending.temporary_path) -Context "pending pageid=$pageId"
        $existingRecord = @($records | Where-Object { [int64]$_.pageid -eq $pageId })
        if ($existingRecord.Count -gt 1) { throw "Pending receipt sees duplicate manifest pageid=$pageId" }
        if ($existingRecord.Count -eq 1) {
            if ([string]$existingRecord[0].download_sha256 -cne $expectedSha -or -not (Test-Path -LiteralPath $finalPath -PathType Leaf) -or
                (Get-FileSha256 $finalPath) -cne $expectedSha) { throw "Committed pending receipt conflicts with manifest pageid=$pageId" }
            if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
                if ((Get-FileSha256 $temporaryPath) -cne $expectedSha) { throw "Stale pending temp SHA mismatch: $temporaryPath" }
                Remove-Item -LiteralPath $temporaryPath
            }
            Remove-Item -LiteralPath $pendingFile.FullName
            Write-Event 'pending_receipt_already_committed' @{ pageid=$pageId; sha256=$expectedSha }
            continue
        }
        if (Test-Path -LiteralPath $finalPath -PathType Leaf) {
            if ((Get-FileSha256 $finalPath) -cne $expectedSha) { throw "Pending final path SHA mismatch: $finalPath" }
            if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
                if ((Get-FileSha256 $temporaryPath) -cne $expectedSha) { throw "Pending temp SHA mismatch: $temporaryPath" }
                Remove-Item -LiteralPath $temporaryPath
            }
        } elseif (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
            if ((Get-FileSha256 $temporaryPath) -cne $expectedSha) { throw "Pending temp SHA mismatch: $temporaryPath" }
            Move-Item -LiteralPath $temporaryPath -Destination $finalPath
        } else {
            throw "Pending receipt has neither matching temp nor final payload: $($pendingFile.FullName)"
        }
        [void](Normalize-CandidateRecord $pendingRecord)
        $records += $pendingRecord
        Save-Outputs $records
        Remove-Item -LiteralPath $pendingFile.FullName
        Write-Event 'pending_orphan_recovered' @{ pageid=$pageId; filename=[string]$pending.final_path; sha256=$expectedSha }
    }
}

$blockedPageIds = [Collections.Generic.HashSet[int64]]::new()
$blockedCommonsSha1 = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$blockedDownloadSha256 = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$usedPageIds = [Collections.Generic.HashSet[int64]]::new()
$usedCommonsSha1 = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$usedDownloadSha256 = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$candidateDHashes = [Collections.Generic.List[string]]::new()
$holdoutDHashes = [Collections.Generic.List[string]]::new()

foreach ($item in $existing) {
    [void]$blockedPageIds.Add([int64]$item.pageid)
    if ($item.commons_sha1) { [void]$blockedCommonsSha1.Add([string]$item.commons_sha1) }
    if ($item.download_sha256) { [void]$blockedDownloadSha256.Add([string]$item.download_sha256) }
    if ([string]$item.split -ceq 'print_demo' -or [string]$item.domain -ceq 'print_demo_source') {
        $holdoutPath = Join-Path $ExistingDataset ([string]$item.filename)
        if (-not (Test-Path -LiteralPath $holdoutPath)) { throw "Permanent holdout missing: $holdoutPath" }
        if ((Get-FileSha256 $holdoutPath) -ne ([string]$item.download_sha256).ToLowerInvariant()) {
            throw "Permanent holdout SHA-256 mismatch: $holdoutPath"
        }
        $holdoutDHashes.Add((Get-DHash64 $holdoutPath))
    }
}

$quarantineRoot = Join-Path $Output 'quarantine'
if (Test-Path -LiteralPath $quarantineRoot -PathType Container) {
    $quarantinedPageIds = [Collections.Generic.HashSet[int64]]::new()
    foreach ($receiptFile in @(Get-ChildItem -LiteralPath $quarantineRoot -Filter 'receipt.json' -File -Recurse | Sort-Object FullName)) {
        $receipt = Get-Content -LiteralPath $receiptFile.FullName -Raw -Encoding utf8 | ConvertFrom-Json
        if ([string]$receipt.schema_version -cne 'rootscope.wikimedia_candidate_quarantine.v1' -or
            [string]$receipt.status -cne 'COMPLETE') {
            throw "Unsupported or incomplete quarantine receipt: $($receiptFile.FullName)"
        }
        $receiptId = [string]$receipt.receipt_id
        if (-not $receiptId -or $receiptFile.Directory.Name -cne $receiptId) {
            throw "Quarantine receipt directory/identity mismatch: $($receiptFile.FullName)"
        }
        $pageId = [int64]$receipt.pageid
        if ($pageId -le 0 -or -not $quarantinedPageIds.Add($pageId)) {
            throw "Duplicate or invalid quarantined pageid=$pageId"
        }
        $expectedPrefix = "quarantine/$receiptId/payload/"
        if (-not ([string]$receipt.destination).StartsWith($expectedPrefix, [StringComparison]::Ordinal)) {
            throw "Quarantine destination is not receipt-bound: $($receiptFile.FullName)"
        }
        $payloadPath = Resolve-DatasetFile -Root $Output -RelativePath ([string]$receipt.destination) -Context "quarantine receipt $receiptId"
        $payloadSha = ([string]$receipt.file_sha256).ToLowerInvariant()
        if (-not (Test-Path -LiteralPath $payloadPath -PathType Leaf) -or (Get-FileSha256 $payloadPath) -cne $payloadSha) {
            throw "Quarantine payload is missing or changed: $($receiptFile.FullName)"
        }
        if (@($records | Where-Object { [int64]$_.pageid -eq $pageId }).Count -ne 0) {
            throw "Quarantined pageid reappeared in manifest: $pageId"
        }
        [void]$blockedPageIds.Add($pageId)
        if ($receipt.record.commons_sha1) { [void]$blockedCommonsSha1.Add([string]$receipt.record.commons_sha1) }
        if ($receipt.record.download_sha256) { [void]$blockedDownloadSha256.Add([string]$receipt.record.download_sha256) }
        $script:QuarantineReceipts += [ordered]@{
            receipt_id=$receiptId
            pageid=$pageId
            receipt_sha256=(Get-FileSha256 $receiptFile.FullName)
            payload_sha256=$payloadSha
            reason=[string]$receipt.reason
        }
    }
}

foreach ($item in $records) {
    [void]$usedPageIds.Add([int64]$item.pageid)
    [void]$usedCommonsSha1.Add([string]$item.commons_sha1)
    [void]$usedDownloadSha256.Add([string]$item.download_sha256)
    $candidateDHashes.Add([string]$item.dhash64)
}

Write-Event 'run_started' @{
    output=$Output; existing_records=$existing.Count; resumed_records=$records.Count
    target_grass=$TargetGrass; target_shrub=$TargetShrub; target_tree=$TargetTree; target_unknown=$TargetUnknown
}

if ($FinalizeOnly) {
    Save-Outputs $records
    Write-Event 'finalize_only_finished' @{ total=$records.Count; manifest_sha256=(Get-FileSha256 (Join-Path $Output 'manifest.jsonl')) }
    Write-Host "FinalizeOnly sealed $($records.Count) records without network access."
    return
}

$rejectTitle = [regex]'(?i)\b(map|distribution|range map|diagram|drawing|illustration|idealized|cross[- ]section|herbarium|specimen|logo|stamp|coin|painting|poster|book page|satellite|modis|landsat|coat of arms|flag|icon|symbol|chart|graph|microscope|fossil)\b'

foreach ($source in $SourcePlan) {
    $classId = [string]$source.ClassId
    $currentCount = @($records | Where-Object class_id -eq $classId).Count
    if (Test-AcquisitionTargetMet $records $classId) { continue }
    $sourcePlanCount = @($records | Where-Object {
        [string]$_.class_id -ceq $classId -and [string]$_.acquisition_mode -ceq [string]$source.Mode -and
        [string]$_.acquisition_query -ceq [string]$source.Query
    }).Count
    if ($sourcePlanCount -ge $MaxPerSourcePlan) {
        Write-Event 'source_plan_cap_reached' @{ class_id=$classId; mode=[string]$source.Mode; query=[string]$source.Query; count=$sourcePlanCount; cap=$MaxPerSourcePlan }
        continue
    }
    if ($classId -ceq 'unknown') {
        $hintCount = @($records | Where-Object { [string]$_.class_id -ceq 'unknown' -and [string]$_.species_hint -ceq [string]$source.SpeciesHint }).Count
        if ($RequiredUnknownHints -cnotcontains [string]$source.SpeciesHint -and $currentCount -ge $TargetUnknown) { continue }
        if ($hintCount -ge $MaxPerUnknownHint) {
            Write-Event 'unknown_hint_cap_reached' @{ species_hint=[string]$source.SpeciesHint; count=$hintCount; cap=$MaxPerUnknownHint }
            continue
        }
    }
    Write-Host "[$classId] $($source.Mode): $($source.Query) ($currentCount/$($TargetByClass[$classId]))"
    try { $pages = @(Get-CommonsPages -Mode $source.Mode -Query $source.Query) }
    catch {
        Write-Event 'source_query_failed' @{ class_id=$classId; mode=$source.Mode; query=$source.Query; error=$_.Exception.Message }
        continue
    }

    foreach ($page in $pages) {
        if (Test-AcquisitionTargetMet $records $classId) { break }
        $sourcePlanCount = @($records | Where-Object {
            [string]$_.class_id -ceq $classId -and [string]$_.acquisition_mode -ceq [string]$source.Mode -and
            [string]$_.acquisition_query -ceq [string]$source.Query
        }).Count
        if ($sourcePlanCount -ge $MaxPerSourcePlan) { break }
        if ($classId -ceq 'unknown') {
            $hintCount = @($records | Where-Object { [string]$_.class_id -ceq 'unknown' -and [string]$_.species_hint -ceq [string]$source.SpeciesHint }).Count
            if ($RequiredUnknownHints -cnotcontains [string]$source.SpeciesHint -and @($records | Where-Object { [string]$_.class_id -ceq 'unknown' }).Count -ge $TargetUnknown) { break }
            if ($hintCount -ge $MaxPerUnknownHint) { break }
        }
        $rawPageId = Get-ObjectValue $page 'pageid'
        if ($null -eq $rawPageId) { continue }
        $pageId = [int64]$rawPageId
        if ($blockedPageIds.Contains($pageId) -or $usedPageIds.Contains($pageId)) { continue }
        $imageInfo = Get-ObjectValue $page 'imageinfo'
        if ($null -eq $imageInfo -or @($imageInfo).Count -eq 0) { continue }
        $info = @($imageInfo)[0]
        $metadata = Get-ObjectValue $info 'extmetadata'
        $mime = [string](Get-ObjectValue $info 'mime')
        $width = [int](Get-ObjectValue $info 'width')
        $height = [int](Get-ObjectValue $info 'height')
        $title = [string](Get-ObjectValue $page 'title')
        $description = Get-MetaValue $metadata 'ImageDescription'
        $licenseName = Get-MetaValue $metadata 'LicenseShortName'
        if (-not $licenseName) { $licenseName = Get-MetaValue $metadata 'UsageTerms' }
        $licenseUrl = Get-MetaValue $metadata 'LicenseUrl'
        $copyrighted = Get-MetaValue $metadata 'Copyrighted'
        $artist = Get-MetaValue $metadata 'Artist'
        if (-not $artist) { $artist = Get-MetaValue $metadata 'Credit' }
        $creatorGroup = if ($artist) { 'commons-creator:' + (Get-Sha256Text $artist).Substring(0,16) } else { '' }
        $credit = Get-MetaValue $metadata 'Credit'
        $attributionRequired = Get-MetaValue $metadata 'AttributionRequired'
        $usageTerms = Get-MetaValue $metadata 'UsageTerms'
        $restrictions = Get-MetaValue $metadata 'Restrictions'
        $licenseDecision = Resolve-LicenseDecision -RawName $licenseName -RawUrl $licenseUrl -Copyrighted $copyrighted `
            -Context 'commons_api' -SourceProvider 'Wikimedia Commons' -PageId $pageId -SourceGroup "commons:$pageId"

        if ($AllowedMime -cnotcontains $mime -or [Math]::Min($width, $height) -lt $MinimumOriginalSide) { continue }
        if ($rejectTitle.IsMatch("$title $description")) { continue }
        if ($null -eq $licenseDecision -or -not $artist) {
            Write-Event 'metadata_rejected' @{ class_id=$classId; pageid=$pageId; title=$title; license=$licenseName; has_artist=[bool]$artist }
            continue
        }
        $creatorCount = @($records | Where-Object { [string]$_.creator_group -ceq $creatorGroup }).Count
        if ($creatorCount -ge $MaxPerCreatorGroup) {
            Write-Event 'creator_group_cap_reached' @{ class_id=$classId; pageid=$pageId; creator_group=$creatorGroup; artist=$artist; count=$creatorCount; cap=$MaxPerCreatorGroup }
            continue
        }
        $commonsSha1 = ([string](Get-ObjectValue $info 'sha1')).ToLowerInvariant()
        if (-not $commonsSha1 -or $blockedCommonsSha1.Contains($commonsSha1) -or $usedCommonsSha1.Contains($commonsSha1)) { continue }
        $sourcePage = [string](Get-ObjectValue $info 'descriptionurl')
        if (-not $sourcePage) { $sourcePage = "https://commons.wikimedia.org/?curid=$pageId" }
        $originalUrl = [string](Get-ObjectValue $info 'url')
        $downloadUrl = [string](Get-ObjectValue $info 'thumburl')
        if (-not $downloadUrl) { $downloadUrl = $originalUrl }
        if (-not $downloadUrl) { continue }

        $classDir = Join-Path (Join-Path $Output 'images') $classId
        New-Item -ItemType Directory -Path $classDir -Force | Out-Null
        $temporaryPath = Join-Path $classDir (".{0}.{1}.download" -f $pageId, $script:RunId)
        $pendingPath = $null
        try {
            Invoke-WithRetry -Operation "download:$pageId" -Action {
                Invoke-WebRequest -Uri $downloadUrl -Headers @{ 'User-Agent'=$UserAgent; 'Referer'=$sourcePage; 'Accept'='image/jpeg,image/png,*/*;q=0.2' } -OutFile $temporaryPath -TimeoutSec 90
            } | Out-Null
            $payload = [IO.File]::ReadAllBytes($temporaryPath)
            $downloadSha256 = Get-Sha256Bytes $payload
            if ($blockedDownloadSha256.Contains($downloadSha256) -or $usedDownloadSha256.Contains($downloadSha256)) {
                Remove-Item -LiteralPath $temporaryPath
                continue
            }
            $facts = Get-ImageFacts $temporaryPath
            if ($facts.Mime -cne $mime) { throw "Downloaded MIME mismatch for pageid=$pageId metadata=$mime decoded=$($facts.Mime)" }
            if ([Math]::Min($facts.Width, $facts.Height) -lt $MinimumDownloadedSide) {
                Write-Event 'downloaded_dimensions_rejected' @{ class_id=$classId; pageid=$pageId; width=$facts.Width; height=$facts.Height; minimum_side=$MinimumDownloadedSide }
                Remove-Item -LiteralPath $temporaryPath
                continue
            }
            $dhash = $facts.DHash64
            $holdoutDistance = 64
            foreach ($known in $holdoutDHashes) { $holdoutDistance = [Math]::Min($holdoutDistance, (Get-HammingDistance $dhash $known)) }
            if ($holdoutDistance -le $HoldoutDHashDistance) {
                Write-Event 'holdout_near_duplicate_rejected' @{ class_id=$classId; pageid=$pageId; title=$title; dhash64=$dhash; minimum_distance=$holdoutDistance }
                Remove-Item -LiteralPath $temporaryPath
                continue
            }
            $candidateDistance = 64
            foreach ($known in $candidateDHashes) { $candidateDistance = [Math]::Min($candidateDistance, (Get-HammingDistance $dhash $known)) }
            if ($candidateDistance -le $CandidateDHashDistance) {
                Write-Event 'candidate_near_duplicate_rejected' @{ class_id=$classId; pageid=$pageId; title=$title; dhash64=$dhash; minimum_distance=$candidateDistance }
                Remove-Item -LiteralPath $temporaryPath
                continue
            }
            $extension = if ($mime -eq 'image/png') { '.png' } else { '.jpg' }
            $filenameOnly = "{0}_{1}_{2}{3}" -f $classId, $pageId, $downloadSha256.Substring(0, 12), $extension
            $finalPath = Join-Path $classDir $filenameOnly
            $relativeFilename = "images/$classId/$filenameOnly"
            $record = [pscustomobject][ordered]@{
                schema_version='rootscope.wikimedia_candidate.v1'
                class_id=$classId
                species_hint=[string]$source.SpeciesHint
                species_hint_status='acquisition_hint_not_a_reviewed_species_or_shape_label'
                candidate_label_status='query_or_category_derived_unverified'
                acquisition_mode=[string]$source.Mode
                acquisition_query=[string]$source.Query
                domain='natural_web_candidate'
                split='UNASSIGNED_DO_NOT_TRAIN'
                review_status='pending_human_visual_and_license_review'
                training_eligible=$false
                print_eligible=$false
                source_provider='Wikimedia Commons'
                source_group="commons:$pageId"
                source_group_basis='one authoritative Commons original file page; crops/augmentations/prints/recaptures must inherit this group'
                creator_group=$creatorGroup
                pageid=$pageId
                title=$title
                source_page=$sourcePage
                original_url=$originalUrl
                download_url=$downloadUrl
                commons_sha1=$commonsSha1
                mime=$mime
                original_width=$width
                original_height=$height
                artist=$artist
                credit=$credit
                license=$licenseName
                license_url=$licenseUrl
                license_raw_name=$licenseName
                license_raw_url=$licenseUrl
                license_canonical_id=[string]$licenseDecision.CanonicalId
                license_canonical_name=[string]$licenseDecision.CanonicalName
                license_canonical_url=[string]$licenseDecision.CanonicalUrl
                license_binding_id=[string]$licenseDecision.BindingId
                license_allowlist_rule=[string]$licenseDecision.BindingId
                license_policy_sha256=$script:LicensePolicySha256
                usage_terms=$usageTerms
                attribution_required=$attributionRequired
                copyrighted=$copyrighted
                restrictions=$restrictions
                description=$description.Substring(0, [Math]::Min(1500, $description.Length))
                license_metadata_source='Wikimedia Commons action=query imageinfo extmetadata and canonical file page'
                rights_review_status='machine_allowlist_pass_human_file_page_and_non_copyright_rights_review_pending'
                accessed_at_utc=[DateTime]::UtcNow.ToString('o')
                filename=$relativeFilename
                download_bytes=$payload.Length
                download_sha256=$downloadSha256
                download_width=$facts.Width
                download_height=$facts.Height
                download_mime=$facts.Mime
                dhash64_algorithm=$DHashAlgorithm
                dhash64=$dhash
                minimum_holdout_dhash_distance=$holdoutDistance
                minimum_prior_candidate_dhash_distance=$candidateDistance
            }
            $pendingDir = Join-Path $Output 'pending'
            New-Item -ItemType Directory -Path $pendingDir -Force | Out-Null
            $pendingPath = Join-Path $pendingDir ("{0}_{1}.json" -f $pageId, $downloadSha256)
            $pendingReceipt = [ordered]@{
                schema_version='rootscope.wikimedia_candidate_pending.v1'
                run_id=$script:RunId
                created_at_utc=[DateTime]::UtcNow.ToString('o')
                temporary_path=("images/$classId/" + [IO.Path]::GetFileName($temporaryPath))
                final_path=$relativeFilename
                download_sha256=$downloadSha256
                record=$record
            }
            Write-AtomicUtf8 $pendingPath (($pendingReceipt | ConvertTo-Json -Depth 30) + "`n")
            if (Test-Path -LiteralPath $finalPath -PathType Leaf) {
                if ((Get-FileSha256 $finalPath) -cne $downloadSha256) {
                    throw "Existing orphan path has a different SHA-256: $finalPath"
                }
                Remove-Item -LiteralPath $temporaryPath
                Write-Event 'same_sha_orphan_adopted' @{ pageid=$pageId; filename=$relativeFilename; sha256=$downloadSha256 }
            } else {
                Move-Item -LiteralPath $temporaryPath -Destination $finalPath
            }
            $records += $record
            [void]$usedPageIds.Add($pageId)
            [void]$usedCommonsSha1.Add($commonsSha1)
            [void]$usedDownloadSha256.Add($downloadSha256)
            $candidateDHashes.Add($dhash)
            Save-Outputs $records
            Remove-Item -LiteralPath $pendingPath
            $pendingPath = $null
            Write-Event 'candidate_saved' @{ class_id=$classId; pageid=$pageId; filename=$relativeFilename; source_group="commons:$pageId"; license=$licenseName; sha256=$downloadSha256 }
            Write-Host "  saved $relativeFilename"
            Start-Sleep -Milliseconds 250
        } catch {
            if (Test-Path -LiteralPath $temporaryPath) { Remove-Item -LiteralPath $temporaryPath }
            Write-Event 'candidate_failed' @{ class_id=$classId; pageid=$pageId; title=$title; error=$_.Exception.Message }
            if ($pendingPath -and (Test-Path -LiteralPath $pendingPath)) { throw }
        }
    }
}

Save-Outputs $records
$finalCounts = [ordered]@{}
foreach ($classId in @('grass_clump','low_shrub','young_tree','unknown')) {
    $finalCounts[$classId] = @($records | Where-Object class_id -eq $classId).Count
}
Write-Event 'run_finished' @{ total=$records.Count; class_counts=$finalCounts }
Write-Host (($finalCounts | ConvertTo-Json -Compress))
Write-Host "Output: $Output"
Write-Host "Status: STAGING_CANDIDATES_MANUAL_VISUAL_AND_LICENSE_REVIEW_REQUIRED_NOT_TRAIN_READY"
} finally {
    if ($null -ne $script:CollectorLock) { $script:CollectorLock.Dispose() }
}
