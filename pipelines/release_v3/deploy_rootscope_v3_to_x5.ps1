param(
    [string]$SshAlias = "rootscope-x5",
    [string]$ReleaseDirectory = "",
    [switch]$StageOnly
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($ReleaseDirectory)) {
    $Receipt = Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot "..\..\output\releases") -Recurse -Filter "release_build_receipt.json" |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $Receipt) { throw "No release build receipt found" }
    $ReleaseDirectory = $Receipt.Directory.FullName
}
$ReleaseDirectory = (Resolve-Path -LiteralPath $ReleaseDirectory).Path
$BuildReceipt = Get-Content -LiteralPath (Join-Path $ReleaseDirectory "release_build_receipt.json") -Raw | ConvertFrom-Json
$CandidateId = [string]$BuildReceipt.candidate_id
if ($CandidateId -notmatch '^rootscope_v3_pc_ready_20260724_[0-9a-f]{12}$') {
    throw "Invalid content-addressed candidate id"
}
$Archive = Join-Path $ReleaseDirectory "$CandidateId.tar"
$ShaFile = Join-Path $ReleaseDirectory "$CandidateId.tar.sha256"
if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) {
    throw "Missing release archive: $Archive"
}
if (-not (Test-Path -LiteralPath $ShaFile -PathType Leaf)) {
    throw "Missing release SHA file: $ShaFile"
}
$ExpectedSha = ((Get-Content -LiteralPath $ShaFile -Raw).Trim() -split '\s+')[0]
$ActualSha = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ExpectedSha -ne $ActualSha) {
    throw "Local archive SHA-256 mismatch"
}
$EvidenceDirectory = Join-Path $ReleaseDirectory (
    "x5_evidence\deploy-{0}-{1}" -f
    [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffffffZ"),
    $PID
)
New-Item -ItemType Directory -Path $EvidenceDirectory | Out-Null

$Target = $SshAlias
$SshOptions = @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=8",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "IdentitiesOnly=yes"
)
$IdentityCommand = @'
set -e
serial_path=/proc/device-tree/serial-number
[ -f "$serial_path" ] || serial_path=/sys/firmware/devicetree/base/serial-number
printf 'hostname=%s\n' "$(hostname)"
printf 'machine_id=%s\n' "$(cat /etc/machine-id)"
printf 'serial=%s\n' "$(tr -d '\000' < "$serial_path")"
printf 'wlan_mac=%s\n' "$(cat /sys/class/net/wlan0/address)"
printf 'arch=%s\n' "$(uname -m)"
printf 'user=%s\n' "$(id -un)"
printf 'home=%s\n' "$HOME"
'@
$IdentityLines = & ssh @SshOptions -- $Target $IdentityCommand
if ($LASTEXITCODE -ne 0) { throw "SSH identity check failed" }
$Identity = @{}
foreach ($Line in $IdentityLines) {
    $Pair = $Line -split "=", 2
    if ($Pair.Count -eq 2) { $Identity[$Pair[0]] = $Pair[1] }
}
$Expected = @{
    hostname = "rootscope-x5"
    machine_id = "00000000000000000000000000000001"
    serial = "3281556110220e0c002bdeab0012004"
    wlan_mac = "02:00:00:00:00:01"
    arch = "aarch64"
    user = "rootscope"
    home = "/opt/rootscope"
}
foreach ($Key in $Expected.Keys) {
    if ($Identity[$Key] -ne $Expected[$Key]) {
        throw "X5 identity mismatch for ${Key}: observed=$($Identity[$Key])"
    }
}

& ssh @SshOptions -- $Target "mkdir -p -m 700 ~/rootscope-v3-upload"
if ($LASTEXITCODE -ne 0) { throw "Unable to create remote upload directory" }
& scp @SshOptions -- $Archive "${Target}:~/rootscope-v3-upload/$CandidateId.tar"
if ($LASTEXITCODE -ne 0) { throw "Archive upload failed" }

$LocalStage = Join-Path $PSScriptRoot "x5_stage_candidate_v3.sh"
$LocalStageSha = (
    Get-FileHash -LiteralPath $LocalStage -Algorithm SHA256
).Hash.ToLowerInvariant()
$RemoteStage = "~/rootscope-v3-upload/x5_stage_candidate_v3.$LocalStageSha.sh"
& scp @SshOptions -- $LocalStage "${Target}:$RemoteStage"
if ($LASTEXITCODE -ne 0) { throw "Stage script upload failed" }
$LocalVerifier = Join-Path $PSScriptRoot "verify_rootscope_v3_release.py"
$LocalVerifierSha = (
    Get-FileHash -LiteralPath $LocalVerifier -Algorithm SHA256
).Hash.ToLowerInvariant()
$RemoteVerifier = (
    "~/rootscope-v3-upload/verify_rootscope_v3_release.$LocalVerifierSha.py"
)
& scp @SshOptions -- $LocalVerifier "${Target}:$RemoteVerifier"
if ($LASTEXITCODE -ne 0) { throw "Trusted verifier upload failed" }
$RemoteTrustCheck = (
    "set -e; chmod 700 $RemoteStage $RemoteVerifier; " +
    "[ `$(sha256sum $RemoteStage | awk '{print `$1}') = $LocalStageSha ]; " +
    "[ `$(sha256sum $RemoteVerifier | awk '{print `$1}') = $LocalVerifierSha ]"
)
& ssh @SshOptions -- $Target $RemoteTrustCheck
if ($LASTEXITCODE -ne 0) { throw "Remote trusted script hash check failed" }

$StageCommand = (
    "bash $RemoteStage $ExpectedSha " +
    "~/rootscope-v3-upload/$CandidateId.tar 0 $RemoteVerifier"
)
$StageOutput = & ssh @SshOptions -- $Target $StageCommand
$StageExitCode = $LASTEXITCODE
$StageOutput | Set-Content -LiteralPath (Join-Path $EvidenceDirectory "stage_output.jsonl") -Encoding UTF8
if ($StageExitCode -ne 0) { throw "Remote staging/verification failed" }

if (-not $StageOnly) {
    $CandidateRoot = "$($Expected.home)/.local/share/rootscope-v3/candidates/$CandidateId"
    $AcceptanceCommand = (
        "bash $CandidateRoot/tools/release_v3/x5_accept_candidate_v3.sh " +
        "$CandidateRoot"
    )
    $AcceptanceOutput = & ssh @SshOptions -- $Target $AcceptanceCommand
    $AcceptanceExitCode = $LASTEXITCODE
    $AcceptanceOutput | Set-Content -LiteralPath (Join-Path $EvidenceDirectory "acceptance_output.jsonl") -Encoding UTF8
    if ($AcceptanceExitCode -ne 0) { throw "X5 software acceptance failed closed" }

    $AcceptanceLines = @(
        $AcceptanceOutput |
            ForEach-Object { ([string]$_).Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    if ($AcceptanceLines.Count -ne 1) {
        throw "Acceptance stdout is not exactly one JSON summary"
    }
    try {
        $Acceptance = $AcceptanceLines[0] | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "Acceptance stdout is not valid JSON: $($_.Exception.Message)"
    }
    $ExpectedAcceptanceStatus = (
        "PASS_X5_OFFLINE_ZERO_AUTHORITY_SOFTWARE_NATIVE_LIBDNN_" +
        "LIVE_RESOURCE_STM32_PHYSICAL_PENDING"
    )
    if (
        [string]$Acceptance.schema -ne "rootscope.v3.x5-software-acceptance.v2" -or
        [string]$Acceptance.status -ne $ExpectedAcceptanceStatus -or
        [string]$Acceptance.candidate_id -ne $CandidateId -or
        [string]$Acceptance.release_root -ne $CandidateRoot
    ) {
        throw "Acceptance summary identity/status mismatch"
    }
    $EvidenceRoot = [string]$Acceptance.evidence_root
    $EscapedHome = [regex]::Escape([string]$Expected.home)
    $AcceptRunPattern = (
        "^$EscapedHome/\.local/share/rootscope-v3/evidence/" +
        "accept-[0-9]{8}T[0-9]{15}Z-[0-9]+$"
    )
    if ($EvidenceRoot -notmatch $AcceptRunPattern) {
        throw "Acceptance evidence root is outside one exact accept run"
    }
    $AcceptanceSummaryPath = "$EvidenceRoot/08_acceptance_summary.json"

    $ExpectedMutationBoundary = @{
        current_selected_or_modified = $false
        service_started = $false
        camera_opened = $false
        serial_opened = $false
        serial_write = $false
        gpio_touched = $false
        pump_touched = $false
        physical_completion = $false
    }
    foreach ($Name in $ExpectedMutationBoundary.Keys) {
        $Property = $Acceptance.mutation_boundary.PSObject.Properties[$Name]
        if (
            $null -eq $Property -or
            $Property.Value -isnot [bool] -or
            $Property.Value -ne $ExpectedMutationBoundary[$Name]
        ) {
            throw "Acceptance mutation boundary mismatch: $Name"
        }
    }
    $NativeQualification = (
        $Acceptance.bpu.qualification_persistent_native_libdnn
    )
    if (
        [string]$NativeQualification.status -ne
            "PASS_X5_PERSISTENT_NATIVE_LIBDNN" -or
        $NativeQualification.passed -isnot [bool] -or
        $NativeQualification.passed -ne $true -or
        $NativeQualification.selected_for_runtime -isnot [bool] -or
        $NativeQualification.selected_for_runtime -ne $false -or
        $NativeQualification.clean_worker_exit -isnot [bool] -or
        $NativeQualification.clean_worker_exit -ne $true -or
        [int]$NativeQualification.count -ne 43 -or
        [int]$NativeQualification.top1_agreement -ne 43
    ) {
        throw "Persistent native libdnn qualification boundary mismatch"
    }
    $HrtQualification = $Acceptance.bpu.canonical_hrt_oracle
    if (
        $HrtQualification.passed -isnot [bool] -or
        $HrtQualification.passed -ne $true -or
        [int]$HrtQualification.count -ne 43 -or
        [int]$HrtQualification.top1_agreement -ne 43
    ) {
        throw "Canonical HRT qualification boundary mismatch"
    }

    $ExpectedReceiptNames = @(
        "00_runtime_bootstrap.json",
        "01_release_verify.json",
        "02_cpu_bm25.json",
        "03_hrt_oracle.json",
        "04_hbm_execution.json",
        "05_native_libdnn.json",
        "06_rootmind_fast_receipt.json",
        "07_rootmind_deep_receipt.json",
        "rootmind_fast_model_binding.json",
        "rootmind_fast_model_page_cache_release.json",
        "rootmind_deep_model_binding.json",
        "rootmind_deep_model_page_cache_release.json",
        "rootmind_precondition_deep_model_binding.json",
        "rootmind_precondition_deep_model_page_cache_release.json",
        "rootmind_precondition_fast_model_binding.json",
        "rootmind_precondition_fast_model_page_cache_release.json"
    )
    $ObservedReceiptNames = @(
        $Acceptance.receipts_sha256.PSObject.Properties.Name | Sort-Object
    )
    $SortedExpectedReceiptNames = @($ExpectedReceiptNames | Sort-Object)
    if (
        $ObservedReceiptNames.Count -ne $SortedExpectedReceiptNames.Count -or
        @(
            Compare-Object $ObservedReceiptNames $SortedExpectedReceiptNames
        ).Count -ne 0
    ) {
        throw "Acceptance receipt hash coverage mismatch"
    }
    foreach ($Name in $ExpectedReceiptNames) {
        $Digest = [string]$Acceptance.receipts_sha256.$Name
        if ($Digest -notmatch '^[0-9a-f]{64}$') {
            throw "Acceptance receipt hash is malformed: $Name"
        }
    }
    foreach ($Role in @("fast", "deep")) {
        $CacheRelease = (
            $Acceptance.rootmind.$Role.model_page_cache_release
        )
        if (
            [string]$CacheRelease.schema -ne
                "rootscope.v3.rootmind-gguf-cache-release.v1" -or
            [string]$CacheRelease.status -ne "PASS" -or
            [string]$CacheRelease.method -ne "POSIX_FADV_DONTNEED" -or
            $CacheRelease.exact_file_only -isnot [bool] -or
            $CacheRelease.exact_file_only -ne $true -or
            $CacheRelease.global_drop_caches -isnot [bool] -or
            $CacheRelease.global_drop_caches -ne $false -or
            [int64]$CacheRelease.resident_bytes_after -gt 4096 -or
            [int64]$CacheRelease.resident_limit_bytes -ne 4096 -or
            [int64]$CacheRelease.cma_free_after_kib -lt 131072 -or
            [int64]$CacheRelease.cma_free_minimum_kib -ne 131072 -or
            [string]$CacheRelease.binding_sha256 -notmatch '^[0-9a-f]{64}$' -or
            [string]$CacheRelease.receipt_sha256 -notmatch '^[0-9a-f]{64}$'
        ) {
            throw "RootMind $Role exact-file cache-release boundary mismatch"
        }
    }
    foreach ($Role in @("deep", "fast")) {
        $CacheRelease = $Acceptance.rootmind_cache_precondition.$Role
        if (
            [string]$CacheRelease.schema -ne
                "rootscope.v3.rootmind-gguf-cache-release.v1" -or
            [string]$CacheRelease.status -ne "PASS" -or
            [string]$CacheRelease.method -ne "POSIX_FADV_DONTNEED" -or
            $CacheRelease.exact_file_only -isnot [bool] -or
            $CacheRelease.exact_file_only -ne $true -or
            $CacheRelease.global_drop_caches -isnot [bool] -or
            $CacheRelease.global_drop_caches -ne $false -or
            [int64]$CacheRelease.resident_bytes_after -gt 4096 -or
            [int64]$CacheRelease.resident_limit_bytes -ne 4096 -or
            [int64]$CacheRelease.cma_free_after_kib -lt 131072 -or
            [int64]$CacheRelease.cma_free_minimum_kib -ne 131072
        ) {
            throw "RootMind $Role cache-precondition boundary mismatch"
        }
    }

    $LocalAcceptanceSummary = Join-Path (
        $EvidenceDirectory
    ) "08_acceptance_summary.remote.json"
    & scp @SshOptions -- (
        "${Target}:$AcceptanceSummaryPath"
    ) $LocalAcceptanceSummary
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to retrieve the exact acceptance summary"
    }
    try {
        $FetchedAcceptance = Get-Content -LiteralPath $LocalAcceptanceSummary -Raw |
            ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "Fetched acceptance summary is not valid JSON: $($_.Exception.Message)"
    }
    if (
        [string]$FetchedAcceptance.schema -ne [string]$Acceptance.schema -or
        [string]$FetchedAcceptance.status -ne [string]$Acceptance.status -or
        [string]$FetchedAcceptance.candidate_id -ne [string]$Acceptance.candidate_id -or
        [string]$FetchedAcceptance.release_root -ne [string]$Acceptance.release_root -or
        [string]$FetchedAcceptance.evidence_root -ne [string]$Acceptance.evidence_root
    ) {
        throw "Fetched acceptance summary does not match acceptance stdout"
    }
    foreach ($Name in $ExpectedReceiptNames) {
        if (
            [string]$FetchedAcceptance.receipts_sha256.$Name -ne
            [string]$Acceptance.receipts_sha256.$Name
        ) {
            throw "Fetched acceptance receipt binding mismatch: $Name"
        }
    }
    $AcceptanceSummarySha = (
        Get-FileHash -LiteralPath $LocalAcceptanceSummary -Algorithm SHA256
    ).Hash.ToLowerInvariant()

    & ssh @SshOptions -- $Target $RemoteTrustCheck
    if ($LASTEXITCODE -ne 0) {
        throw "Remote trusted script hash changed before activation"
    }
    $ActivationCommand = (
        "bash $RemoteStage $ExpectedSha " +
        "~/rootscope-v3-upload/$CandidateId.tar 1 $RemoteVerifier " +
        "$AcceptanceSummaryPath $AcceptanceSummarySha"
    )
    $ActivationOutput = & ssh @SshOptions -- $Target $ActivationCommand
    $ActivationExitCode = $LASTEXITCODE
    $ActivationOutput | Set-Content -LiteralPath (Join-Path $EvidenceDirectory "activation_output.jsonl") -Encoding UTF8
    if ($ActivationExitCode -ne 0) {
        throw "Acceptance-bound candidate activation failed closed and rolled back"
    }
    $ActivationLines = @(
        $ActivationOutput |
            ForEach-Object { ([string]$_).Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    if ($ActivationLines.Count -ne 1) {
        throw "Activation stdout is not exactly one JSON receipt"
    }
    try {
        $Activation = $ActivationLines[0] | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "Activation stdout is not valid JSON: $($_.Exception.Message)"
    }
    if (
        [string]$Activation.schema -ne "rootscope.v3.x5-stage-receipt.v2" -or
        [string]$Activation.candidate_id -ne $CandidateId -or
        [string]$Activation.current -ne $CandidateRoot -or
        [string]$Activation.acceptance_summary -ne $AcceptanceSummaryPath -or
        [string]$Activation.acceptance_summary_sha256 -ne $AcceptanceSummarySha -or
        $Activation.activation_transaction_committed -isnot [bool] -or
        $Activation.activation_transaction_committed -ne $true
    ) {
        throw "Activation receipt binding mismatch"
    }
}

Write-Host "RootScope v3 staged on verified X5: $Target"
Write-Host "No camera, serial port, GPIO, or pump was opened; RootMind loopback smoke services were temporary and exited."
if ($StageOnly) { Write-Host "StageOnly honored: current symlink was not changed." }
