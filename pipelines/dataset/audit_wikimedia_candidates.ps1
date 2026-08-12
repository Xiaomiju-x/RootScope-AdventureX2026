[CmdletBinding()]
param(
    [string]$Dataset = "",
    [string]$ExistingDataset = "",
    [string]$Out = "",
    [string]$LicensePolicy = "",
    [int]$HoldoutDHashDistance = -1,
    [int]$CandidateDHashDistance = -1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ScriptPath = $MyInvocation.MyCommand.Path
$AdventureXRoot = (Resolve-Path (Join-Path (Split-Path $ScriptPath -Parent) '../..')).Path
if (-not $Dataset) { $Dataset = Join-Path $AdventureXRoot 'datasets/desert_plants_wikimedia_staging_e0' }
if (-not $ExistingDataset) { $ExistingDataset = Join-Path $AdventureXRoot 'datasets/desert_plants_v1' }
if (-not $LicensePolicy) { $LicensePolicy = Join-Path (Split-Path $ScriptPath -Parent) 'wikimedia_license_policy_v1.json' }
$Dataset = [IO.Path]::GetFullPath($Dataset)
$ExistingDataset = [IO.Path]::GetFullPath($ExistingDataset)
$LicensePolicy = [IO.Path]::GetFullPath($LicensePolicy)
if (-not $Out) { $Out = Join-Path $Dataset 'integrity_audit.json' }
$Out = [IO.Path]::GetFullPath($Out)

$ExpectedClasses = @('grass_clump','low_shrub','young_tree','unknown')
$MinimumCounts = @{ grass_clump=30; low_shrub=30; young_tree=30; unknown=40 }
if (-not (Test-Path -LiteralPath $LicensePolicy -PathType Leaf)) { throw "Missing shared license policy: $LicensePolicy" }
$PolicySha256 = (Get-FileHash -LiteralPath $LicensePolicy -Algorithm SHA256).Hash.ToLowerInvariant()
$Policy = Get-Content -LiteralPath $LicensePolicy -Raw -Encoding utf8 | ConvertFrom-Json
if ([string]$Policy.schema_version -cne 'rootscope.wikimedia_license_policy.v1' -or
    [string]$Policy.matching.comparison -cne 'ordinal_case_sensitive' -or
    [bool]$Policy.matching.trim_whitespace -or [bool]$Policy.matching.normalize_trailing_slash -or
    [string]$Policy.matching.unknown_binding_action -cne 'REJECT') { throw 'Invalid shared exact-pair license policy' }
$AllowedMime = @($Policy.image_constraints.allowed_mime | ForEach-Object { [string]$_ })
$MinimumOriginalSide = [int]$Policy.image_constraints.minimum_original_side
$MinimumDownloadedSide = [int]$Policy.image_constraints.minimum_downloaded_side
$DHashAlgorithm = [string]$Policy.image_constraints.dhash_algorithm

function Read-JsonLines {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing JSONL: $Path" }
    $items = @()
    foreach ($line in [IO.File]::ReadLines($Path)) {
        if ($line.Trim()) { $items += ($line | ConvertFrom-Json) }
    }
    return @($items)
}

function Get-FileSha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-Sha256Text {
    param([string]$Text)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($Text)))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Get-ImageFacts {
    param([string]$Path)
    Add-Type -AssemblyName System.Drawing
    $bitmap = [Drawing.Bitmap]::new($Path)
    try {
        $mime = if ($bitmap.RawFormat.Guid -eq [Drawing.Imaging.ImageFormat]::Jpeg.Guid) { 'image/jpeg' }
            elseif ($bitmap.RawFormat.Guid -eq [Drawing.Imaging.ImageFormat]::Png.Guid) { 'image/png' }
            else { 'unsupported' }
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
        return [pscustomobject]@{ Width=$bitmap.Width; Height=$bitmap.Height; Mime=$mime; DHash64=$hex.ToString() }
    } finally { $bitmap.Dispose() }
}

function Get-DHash64 { param([string]$Path); return (Get-ImageFacts $Path).DHash64 }

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

function Get-ObjectValue {
    param([AllowNull()][object]$Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function Resolve-LicenseDecision {
    param([object]$Record, [string]$Context)
    $pageId = [int64]$Record.pageid
    $rawNameValue = Get-ObjectValue $Record 'license_raw_name'
    $rawUrlValue = Get-ObjectValue $Record 'license_raw_url'
    $rawName = if ($null -ne $rawNameValue) { [string]$rawNameValue } else { [string]$Record.license }
    $rawUrl = if ($null -ne $rawUrlValue) { [string]$rawUrlValue } else { [string]$Record.license_url }
    if ($null -ne $rawNameValue -and [string]$Record.license -cne $rawName) { return $null }
    if ($null -ne $rawUrlValue -and [string]$Record.license_url -cne $rawUrl) { return $null }
    foreach ($license in @($Policy.licenses)) {
        foreach ($binding in @($license.raw_bindings)) {
            if ([string]$binding.raw_name -cne $rawName -or [string]$binding.raw_url -cne $rawUrl) { continue }
            $falseRaw = Get-ObjectValue $license 'copyrighted_exact_values'
            $falseValues = [Collections.Generic.List[string]]::new()
            if ($null -ne $falseRaw) { foreach ($value in @($falseRaw)) { $falseValues.Add([string]$value) } }
            if ($falseValues.Count -gt 0 -and $falseValues -cnotcontains [string]$Record.copyrighted) { return $null }
            return [pscustomobject]@{ RawName=$rawName; RawUrl=$rawUrl; CanonicalId=[string]$license.canonical_id; CanonicalName=[string]$license.canonical_name; CanonicalUrl=[string]$license.canonical_url; BindingId="policy:$($license.canonical_id):$rawName|$rawUrl" }
        }
    }
    foreach ($exception in @($Policy.legacy_exceptions)) {
        if ([string]$exception.context -cne $Context -or [string]$exception.source_provider -cne [string]$Record.source_provider -or
            [int64]$exception.pageid -ne $pageId -or [string]$exception.source_group -cne [string]$Record.source_group -or
            [string]$exception.raw_name -cne $rawName -or [string]$exception.raw_url -cne $rawUrl) { continue }
        $license = @($Policy.licenses | Where-Object { [string]$_.canonical_id -ceq [string]$exception.canonical_id })
        if ($license.Count -ne 1) { return $null }
        return [pscustomobject]@{ RawName=$rawName; RawUrl=$rawUrl; CanonicalId=[string]$license[0].canonical_id; CanonicalName=[string]$license[0].canonical_name; CanonicalUrl=[string]$license[0].canonical_url; BindingId="exception:$($exception.exception_id)" }
    }
    return $null
}

function Resolve-DatasetFile {
    param([string]$Root, [string]$RelativePath, [string]$Context)
    if (-not $RelativePath -or $RelativePath.Contains('\') -or [IO.Path]::IsPathRooted($RelativePath)) {
        Add-Failure 'unsafe_filename' "$Context filename=$RelativePath"
        return $null
    }
    $rootFull = [IO.Path]::GetFullPath($Root).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $candidate = [IO.Path]::GetFullPath((Join-Path $Root $RelativePath))
    if (-not $candidate.StartsWith($rootFull, [StringComparison]::OrdinalIgnoreCase)) {
        Add-Failure 'unsafe_filename' "$Context filename=$RelativePath"
        return $null
    }
    return $candidate
}

function Add-Failure {
    param([string]$Code, [string]$Detail)
    $script:Failures.Add([pscustomobject]@{ code=$Code; detail=$Detail })
}

function Assert-Unique {
    param([object[]]$Records, [string]$Field)
    $duplicates = @($Records | Group-Object -Property $Field | Where-Object Count -gt 1)
    foreach ($duplicate in $duplicates) { Add-Failure "duplicate_$Field" "$($duplicate.Name) count=$($duplicate.Count)" }
}

$script:Failures = [Collections.Generic.List[object]]::new()
$Checks = [ordered]@{}
$manifestPath = Join-Path $Dataset 'manifest.jsonl'
$summaryPath = Join-Path $Dataset 'summary.json'
$collectorPath = Join-Path (Split-Path $ScriptPath -Parent) 'collect_wikimedia_candidates.ps1'
$records = @(Read-JsonLines $manifestPath)
$existing = @(Read-JsonLines (Join-Path $ExistingDataset 'manifest.jsonl'))
if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) { throw "Missing summary: $summaryPath" }
$summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding utf8 | ConvertFrom-Json
$manifestSha256 = Get-FileSha256 $manifestPath
$collectorSha256 = Get-FileSha256 $collectorPath

if ([string]$summary.manifest_sha256 -cne $manifestSha256) { Add-Failure 'summary_manifest_sha256' "actual=$manifestSha256 summary=$($summary.manifest_sha256)" }
if ([string]$summary.license_policy_sha256 -cne $PolicySha256) { Add-Failure 'summary_policy_sha256' "actual=$PolicySha256 summary=$($summary.license_policy_sha256)" }
if ([string]$summary.collector_script_sha256 -cne $collectorSha256) { Add-Failure 'summary_collector_sha256' "actual=$collectorSha256 summary=$($summary.collector_script_sha256)" }
$summaryHoldoutThreshold = [int]$summary.holdout_dhash_rejection_threshold
$summaryCandidateThreshold = [int]$summary.candidate_dhash_rejection_threshold
if ($HoldoutDHashDistance -ge 0 -and $HoldoutDHashDistance -ne $summaryHoldoutThreshold) { Add-Failure 'holdout_threshold_argument_mismatch' "argument=$HoldoutDHashDistance summary=$summaryHoldoutThreshold" }
if ($CandidateDHashDistance -ge 0 -and $CandidateDHashDistance -ne $summaryCandidateThreshold) { Add-Failure 'candidate_threshold_argument_mismatch' "argument=$CandidateDHashDistance summary=$summaryCandidateThreshold" }
$HoldoutDHashDistance = $summaryHoldoutThreshold
$CandidateDHashDistance = $summaryCandidateThreshold
if ($HoldoutDHashDistance -lt 1 -or $HoldoutDHashDistance -gt 50) { Add-Failure 'holdout_threshold_range' $HoldoutDHashDistance }
if ($CandidateDHashDistance -lt 0 -or $CandidateDHashDistance -gt 20) { Add-Failure 'candidate_threshold_range' $CandidateDHashDistance }
if ([int]$summary.image_constraints.minimum_original_side -ne $MinimumOriginalSide -or
    [int]$summary.image_constraints.minimum_downloaded_side -ne $MinimumDownloadedSide -or
    [string]$summary.image_constraints.dhash_algorithm -cne $DHashAlgorithm) { Add-Failure 'summary_image_constraints' 'summary does not match shared policy' }

$expectedHoldoutPageIds = @(133271396,75559442,2738023,4424728,5445424,6021614 | Sort-Object)
$holdouts = @($existing | Where-Object { [string]$_.split -ceq 'print_demo' -or [string]$_.domain -ceq 'print_demo_source' })
$actualHoldoutPageIds = @($holdouts | ForEach-Object { [int64]$_.pageid } | Sort-Object)
if ($actualHoldoutPageIds.Count -ne 6 -or @(Compare-Object $expectedHoldoutPageIds $actualHoldoutPageIds).Count -ne 0) { Add-Failure 'holdout_identity_set' ($actualHoldoutPageIds -join ',') }
if ([int]$summary.permanent_print_holdout_count -ne $holdouts.Count -or
    @(Compare-Object @($summary.permanent_print_holdout_pageids | ForEach-Object { [int64]$_ } | Sort-Object) $actualHoldoutPageIds).Count -ne 0) { Add-Failure 'summary_holdouts' 'summary holdout set mismatch' }
$holdoutHashes = [Collections.Generic.List[string]]::new()
foreach ($holdout in $holdouts) {
    $path = Resolve-DatasetFile -Root $ExistingDataset -RelativePath ([string]$holdout.filename) -Context "holdout pageid=$($holdout.pageid)"
    if (-not $path -or -not (Test-Path -LiteralPath $path -PathType Leaf)) { Add-Failure 'holdout_missing' ([string]$path); continue }
    if ((Get-FileSha256 $path) -cne ([string]$holdout.download_sha256).ToLowerInvariant()) { Add-Failure 'holdout_sha256' $path; continue }
    try { $facts = Get-ImageFacts $path } catch { Add-Failure 'holdout_decode' "$path $($_.Exception.Message)"; continue }
    if ($AllowedMime -cnotcontains $facts.Mime -or [Math]::Min($facts.Width,$facts.Height) -lt $MinimumDownloadedSide) { Add-Failure 'holdout_image_constraints' $path; continue }
    $holdoutHashes.Add($facts.DHash64)
}

Assert-Unique $records 'pageid'; Assert-Unique $records 'source_group'; Assert-Unique $records 'commons_sha1'; Assert-Unique $records 'download_sha256'
$existingPageIds = [Collections.Generic.HashSet[int64]]::new()
$existingGroups = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$existingCommonsSha1 = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$existingSha256 = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($item in $existing) {
    [void]$existingPageIds.Add([int64]$item.pageid); [void]$existingGroups.Add([string]$item.source_group)
    if ($item.commons_sha1) { [void]$existingCommonsSha1.Add([string]$item.commons_sha1) }
    if ($item.download_sha256) { [void]$existingSha256.Add([string]$item.download_sha256) }
}

$overlaps = [ordered]@{ pageid=0; source_group=0; commons_sha1=0; download_sha256=0 }
$candidateHashes = [Collections.Generic.List[string]]::new()
$manifestFiles = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$minimumHoldoutDistance = 64; $minimumCandidateDistance = 64
foreach ($record in $records) {
    $identity = "pageid=$($record.pageid)"
    if (($record.pageid -isnot [int]) -and ($record.pageid -isnot [long])) { Add-Failure 'pageid_type' $identity; continue }
    if ($ExpectedClasses -cnotcontains [string]$record.class_id) { Add-Failure 'class_id' "$identity class=$($record.class_id)" }
    if ([string]$record.schema_version -cne 'rootscope.wikimedia_candidate.v1') { Add-Failure 'schema_version' $identity }
    if ([string]$record.source_provider -cne 'Wikimedia Commons') { Add-Failure 'provider' $identity }
    if ([string]$record.source_group -cne "commons:$($record.pageid)") { Add-Failure 'source_group' $identity }
    if ([string]$record.split -cne 'UNASSIGNED_DO_NOT_TRAIN') { Add-Failure 'split' $identity }
    if (($record.training_eligible -isnot [bool]) -or $record.training_eligible) { Add-Failure 'training_eligible' $identity }
    if (($record.print_eligible -isnot [bool]) -or $record.print_eligible) { Add-Failure 'print_eligible' $identity }
    if ([string]$record.review_status -cne 'pending_human_visual_and_license_review' -or
        [string]$record.candidate_label_status -cne 'query_or_category_derived_unverified' -or
        [string]$record.species_hint_status -cne 'acquisition_hint_not_a_reviewed_species_or_shape_label' -or
        [string]$record.rights_review_status -cne 'machine_allowlist_pass_human_file_page_and_non_copyright_rights_review_pending') { Add-Failure 'unreviewed_status_contract' $identity }
    if (-not ([string]$record.artist)) { Add-Failure 'artist_missing' $identity }
    $expectedCreatorGroup = 'commons-creator:' + (Get-Sha256Text ([string]$record.artist)).Substring(0,16)
    if ([string]$record.creator_group -cne $expectedCreatorGroup) { Add-Failure 'creator_group' $identity }
    $licenseDecision = Resolve-LicenseDecision $record 'staging_manifest'
    if ($null -eq $licenseDecision) { Add-Failure 'license_pair' $identity }
    else {
        if ([string]$record.license_canonical_id -cne $licenseDecision.CanonicalId -or
            [string]$record.license_canonical_name -cne $licenseDecision.CanonicalName -or
            [string]$record.license_canonical_url -cne $licenseDecision.CanonicalUrl -or
            [string]$record.license_binding_id -cne $licenseDecision.BindingId -or
            [string]$record.license_policy_sha256 -cne $PolicySha256) { Add-Failure 'license_canonical_binding' $identity }
    }
    try {
        $sourceUri=[Uri][string]$record.source_page; $originalUri=[Uri][string]$record.original_url; $downloadUri=[Uri][string]$record.download_url
        if ($sourceUri.Scheme -cne 'https' -or $sourceUri.Host -cne 'commons.wikimedia.org') { Add-Failure 'source_page_url' $identity }
        if ($originalUri.Scheme -cne 'https' -or $originalUri.Host -cne 'upload.wikimedia.org') { Add-Failure 'original_url' $identity }
        if ($downloadUri.Scheme -cne 'https' -or $downloadUri.Host -cne 'upload.wikimedia.org') { Add-Failure 'download_url' $identity }
    } catch { Add-Failure 'url_parse' "$identity $($_.Exception.Message)" }
    if ($existingPageIds.Contains([int64]$record.pageid)) { $overlaps.pageid++; Add-Failure 'existing_pageid_overlap' $identity }
    if ($existingGroups.Contains([string]$record.source_group)) { $overlaps.source_group++; Add-Failure 'existing_source_group_overlap' $identity }
    if ($existingCommonsSha1.Contains([string]$record.commons_sha1)) { $overlaps.commons_sha1++; Add-Failure 'existing_commons_sha1_overlap' $identity }
    if ($existingSha256.Contains([string]$record.download_sha256)) { $overlaps.download_sha256++; Add-Failure 'existing_sha256_overlap' $identity }
    $path = Resolve-DatasetFile -Root $Dataset -RelativePath ([string]$record.filename) -Context $identity
    if (-not $path) { continue }; [void]$manifestFiles.Add($path)
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { Add-Failure 'file_missing' $path; continue }
    if ((Get-FileSha256 $path) -cne ([string]$record.download_sha256).ToLowerInvariant()) { Add-Failure 'file_sha256' $path; continue }
    if ((Get-Item -LiteralPath $path).Length -ne [int64]$record.download_bytes) { Add-Failure 'file_size' $path }
    try { $facts=Get-ImageFacts $path } catch { Add-Failure 'image_decode' "$path $($_.Exception.Message)"; continue }
    if ($AllowedMime -cnotcontains $facts.Mime -or [string]$record.mime -cne $facts.Mime -or [string]$record.download_mime -cne $facts.Mime) { Add-Failure 'decoded_mime' $path }
    if ([Math]::Min([int]$record.original_width,[int]$record.original_height) -lt $MinimumOriginalSide) { Add-Failure 'original_dimensions' $path }
    if ([Math]::Min($facts.Width,$facts.Height) -lt $MinimumDownloadedSide -or [int]$record.download_width -ne $facts.Width -or [int]$record.download_height -ne $facts.Height) { Add-Failure 'downloaded_dimensions' $path }
    if ([string]$record.dhash64_algorithm -cne $DHashAlgorithm -or [string]$record.dhash64 -cne $facts.DHash64) { Add-Failure 'dhash_manifest' $path }
    foreach ($known in $holdoutHashes) { $minimumHoldoutDistance=[Math]::Min($minimumHoldoutDistance,(Get-HammingDistance $facts.DHash64 $known)) }
    foreach ($known in $candidateHashes) { $minimumCandidateDistance=[Math]::Min($minimumCandidateDistance,(Get-HammingDistance $facts.DHash64 $known)) }
    $candidateHashes.Add($facts.DHash64)
}
if ($minimumHoldoutDistance -le $HoldoutDHashDistance) { Add-Failure 'holdout_dhash_overlap' "min=$minimumHoldoutDistance threshold=$HoldoutDHashDistance" }
if ($records.Count -gt 1 -and $minimumCandidateDistance -le $CandidateDHashDistance) { Add-Failure 'candidate_dhash_overlap' "min=$minimumCandidateDistance threshold=$CandidateDHashDistance" }

$actualImageFiles=@(Get-ChildItem -LiteralPath (Join-Path $Dataset 'images') -Recurse -File | ForEach-Object FullName)
foreach($path in $actualImageFiles){if(-not $manifestFiles.Contains([IO.Path]::GetFullPath($path))){Add-Failure 'unmanifested_image' $path}}
if($actualImageFiles.Count -ne $records.Count){Add-Failure 'image_count' "manifest=$($records.Count) files=$($actualImageFiles.Count)"}
$temporaryFiles=@(Get-ChildItem -LiteralPath $Dataset -Recurse -File | Where-Object{$_.Name -match '\.(download|tmp)$'})
foreach($path in $temporaryFiles){Add-Failure 'temporary_file' $path.FullName}
$pendingReceipts=@(Get-ChildItem -LiteralPath (Join-Path $Dataset 'pending') -Filter '*.json' -File -ErrorAction SilentlyContinue)
foreach($path in $pendingReceipts){Add-Failure 'pending_receipt' $path.FullName}

$classCounts=[ordered]@{}
foreach($classId in $ExpectedClasses){$count=@($records|Where-Object{[string]$_.class_id -ceq $classId}).Count;$classCounts[$classId]=$count;if($count -lt $MinimumCounts[$classId]){Add-Failure 'class_count' "$classId expected_min=$($MinimumCounts[$classId]) actual=$count"};$target=[int]$summary.requested_targets.$classId;if($target -lt 1 -or $count -lt $target){Add-Failure 'requested_target' "$classId requested=$target actual=$count"};if([int]$summary.class_counts.$classId -ne $count){Add-Failure 'summary_class_count' $classId}}
$requiredUnknownHints=@('negative:bare_sand','negative:rocks','negative:blank_card','negative:hand','negative:irrelevant_object','negative:other_plant_form')
$summaryUnknownHints=@($summary.unknown_hint_requirements.required_hints|ForEach-Object{[string]$_}|Sort-Object)
if(@(Compare-Object @($requiredUnknownHints|Sort-Object) $summaryUnknownHints).Count -ne 0){Add-Failure 'unknown_hint_policy' 'required hint set mismatch'}
$unknownHintMinimum=[int]$summary.unknown_hint_requirements.minimum_per_hint
if($unknownHintMinimum -lt 1 -or $unknownHintMinimum -gt [int]$summary.diversity_caps.max_per_unknown_species_hint){Add-Failure 'unknown_hint_minimum' $unknownHintMinimum}
$unknownHintCounts=[ordered]@{};$unknownMinimumsMet=$true
foreach($hint in $requiredUnknownHints){$count=@($records|Where-Object{[string]$_.class_id -ceq 'unknown' -and [string]$_.species_hint -ceq $hint}).Count;$unknownHintCounts[$hint]=$count;if($count -lt $unknownHintMinimum){$unknownMinimumsMet=$false;Add-Failure 'unknown_hint_deficit' "$hint minimum=$unknownHintMinimum actual=$count"};$summaryCount=@($summary.unknown_hint_requirements.counts|Where-Object{[string]$_.species_hint -ceq $hint});if($summaryCount.Count -ne 1 -or [int]$summaryCount[0].count -ne $count -or [bool]$summaryCount[0].met -ne ($count -ge $unknownHintMinimum)){Add-Failure 'summary_unknown_hint_count' $hint}}
if([bool]$summary.unknown_hint_requirements.unknown_hint_minimums_met -ne $unknownMinimumsMet){Add-Failure 'summary_unknown_hint_minimums_met' "summary=$($summary.unknown_hint_requirements.unknown_hint_minimums_met) actual=$unknownMinimumsMet"}
if(-not [bool]$summary.acquisition_targets_met){Add-Failure 'summary_targets_not_met' 'acquisition_targets_met=false'}
if([int]$summary.total -ne $records.Count){Add-Failure 'summary_total' "summary=$($summary.total) manifest=$($records.Count)"}
if([string]$summary.status -cne 'STAGING_CANDIDATES_MANUAL_VISUAL_AND_LICENSE_REVIEW_REQUIRED_NOT_TRAIN_READY'){Add-Failure 'summary_status' ([string]$summary.status)}
if([string]$summary.split_status -cne 'UNASSIGNED_DO_NOT_TRAIN'){Add-Failure 'summary_split' ([string]$summary.split_status)}
foreach($key in @('pageid','source_group','commons_sha1','download_sha256')){if([int]$summary.existing_overlap_counts.$key -ne [int]$overlaps[$key]){Add-Failure 'summary_overlap' "$key summary=$($summary.existing_overlap_counts.$key) actual=$($overlaps[$key])"}}

$Checks.manifest_records=$records.Count;$Checks.image_files=$actualImageFiles.Count;$Checks.class_counts=$classCounts;$Checks.holdout_count=$holdouts.Count
$Checks.minimum_holdout_dhash_distance=$minimumHoldoutDistance;$Checks.holdout_reject_at_or_below=$HoldoutDHashDistance
$Checks.minimum_candidate_dhash_distance=$minimumCandidateDistance;$Checks.candidate_reject_at_or_below=$CandidateDHashDistance
$Checks.minimum_original_side=$MinimumOriginalSide;$Checks.minimum_downloaded_side=$MinimumDownloadedSide;$Checks.dhash_algorithm=$DHashAlgorithm
$Checks.unknown_hint_minimum=$unknownHintMinimum;$Checks.unknown_hint_counts=$unknownHintCounts;$Checks.unknown_hint_minimums_met=$unknownMinimumsMet
$Checks.existing_overlap_counts=$overlaps;$Checks.all_records_unassigned_do_not_train=@($records|Where-Object{[string]$_.split -cne 'UNASSIGNED_DO_NOT_TRAIN'}).Count -eq 0
$Checks.all_records_training_eligible_false=@($records|Where-Object{$_.training_eligible -ne $false}).Count -eq 0

$audit=[ordered]@{schema_version='rootscope.wikimedia_staging_integrity_audit.v2';generated_at_utc=[DateTime]::UtcNow.ToString('o');result=$(if($Failures.Count -eq 0){'PASS_STAGING_INTEGRITY_NOT_TRAIN_READY'}else{'FAIL'});dataset=$Dataset;existing_dataset=$ExistingDataset;manifest_sha256=$manifestSha256;summary_sha256=(Get-FileSha256 $summaryPath);license_policy_sha256=$PolicySha256;collector_script_sha256=$collectorSha256;thresholds=[ordered]@{holdout_dhash_reject_at_or_below=$HoldoutDHashDistance;candidate_dhash_reject_at_or_below=$CandidateDHashDistance};image_constraints=[ordered]@{allowed_mime=$AllowedMime;minimum_original_side=$MinimumOriginalSide;minimum_downloaded_side=$MinimumDownloadedSide;dhash_algorithm=$DHashAlgorithm};checks=$Checks;failure_count=$Failures.Count;failures=@($Failures);explicit_non_claims=@('DATA_LOCKED','TRAIN_READY','LICENSE_HUMAN_APPROVED','VISUAL_LABEL_APPROVED','SPLIT_READY')}
$json=($audit|ConvertTo-Json -Depth 30)+"`n";$temporaryOut="$Out.$PID.tmp";$parent=Split-Path $Out -Parent;if(-not(Test-Path -LiteralPath $parent)){New-Item -ItemType Directory -Path $parent|Out-Null};[IO.File]::WriteAllText($temporaryOut,$json,[Text.UTF8Encoding]::new($false));Move-Item -LiteralPath $temporaryOut -Destination $Out -Force
Write-Host($audit|ConvertTo-Json -Depth 30);if($Failures.Count -ne 0){exit 1}
