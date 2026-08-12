[CmdletBinding()]
param(
    [string]$Dataset = "",
    [ValidateRange(2, 8)]
    [int]$Columns = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ScriptPath = $MyInvocation.MyCommand.Path
$AdventureXRoot = (Resolve-Path (Join-Path (Split-Path $ScriptPath -Parent) '../..')).Path
if (-not $Dataset) { $Dataset = Join-Path $AdventureXRoot 'datasets/desert_plants_wikimedia_staging_e0' }
$Dataset = [IO.Path]::GetFullPath($Dataset)
$manifest = Join-Path $Dataset 'manifest.jsonl'
if (-not (Test-Path -LiteralPath $manifest)) { throw "Missing manifest: $manifest" }

$records = @()
foreach ($line in [IO.File]::ReadLines($manifest)) {
    if ($line.Trim()) { $records += ($line | ConvertFrom-Json) }
}
$output = Join-Path $Dataset 'contact_sheets'
New-Item -ItemType Directory -Path $output -Force | Out-Null
Add-Type -AssemblyName System.Drawing

$tileWidth = 300
$tileHeight = 225
$headerHeight = 54
$font = [Drawing.Font]::new('Arial', 10, [Drawing.FontStyle]::Regular)
$headerFont = [Drawing.Font]::new('Arial', 16, [Drawing.FontStyle]::Bold)
$smallFont = [Drawing.Font]::new('Arial', 8, [Drawing.FontStyle]::Regular)
$background = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(246, 244, 238))
$textBrush = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(25, 35, 31))
$pendingBrush = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(145, 82, 22))
$borderPen = [Drawing.Pen]::new([Drawing.Color]::FromArgb(185, 183, 175), 1)
$jpegCodec = [Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object MimeType -eq 'image/jpeg'
$encoder = [Drawing.Imaging.Encoder]::Quality
$encoderParameters = [Drawing.Imaging.EncoderParameters]::new(1)
$encoderParameters.Param[0] = [Drawing.Imaging.EncoderParameter]::new($encoder, [int64]90)

try {
    foreach ($classId in @('grass_clump','low_shrub','young_tree','unknown')) {
        $items = @($records | Where-Object class_id -eq $classId | Sort-Object pageid)
        if ($items.Count -eq 0) { continue }
        $rows = [int][Math]::Ceiling($items.Count / [double]$Columns)
        $canvas = [Drawing.Bitmap]::new($Columns * $tileWidth, $headerHeight + $rows * $tileHeight)
        try {
            $graphics = [Drawing.Graphics]::FromImage($canvas)
            try {
                $graphics.FillRectangle($background, 0, 0, $canvas.Width, $canvas.Height)
                $graphics.DrawString("$classId - $($items.Count) machine-screened candidates", $headerFont, $textBrush, 12, 7)
                $graphics.DrawString('PENDING HUMAN VISUAL + RIGHTS REVIEW / DO NOT TRAIN', $smallFont, $pendingBrush, 14, 33)
                for ($index = 0; $index -lt $items.Count; $index++) {
                    $record = $items[$index]
                    $column = $index % $Columns
                    $row = [Math]::Floor($index / $Columns)
                    $left = $column * $tileWidth
                    $top = $headerHeight + $row * $tileHeight
                    $graphics.DrawRectangle($borderPen, $left, $top, $tileWidth - 1, $tileHeight - 1)
                    $path = Join-Path $Dataset ([string]$record.filename)
                    $image = [Drawing.Image]::FromFile($path)
                    try {
                        $maxWidth = $tileWidth - 14
                        $maxHeight = 166
                        $scale = [Math]::Min($maxWidth / [double]$image.Width, $maxHeight / [double]$image.Height)
                        $drawWidth = [int][Math]::Round($image.Width * $scale)
                        $drawHeight = [int][Math]::Round($image.Height * $scale)
                        $drawLeft = $left + [int](($tileWidth - $drawWidth) / 2)
                        $drawTop = $top + 7 + [int](($maxHeight - $drawHeight) / 2)
                        $graphics.DrawImage($image, $drawLeft, $drawTop, $drawWidth, $drawHeight)
                    } finally { $image.Dispose() }
                    $labelTop = $top + 177
                    $graphics.DrawString("pageid $($record.pageid) | $($record.license)", $font, $textBrush, $left + 7, $labelTop)
                    $hint = [string]$record.species_hint
                    if ($hint.Length -gt 42) { $hint = $hint.Substring(0, 39) + '...' }
                    $graphics.DrawString($hint, $smallFont, $pendingBrush, $left + 7, $labelTop + 20)
                }
            } finally { $graphics.Dispose() }
            $pathOut = Join-Path $output "$classId.jpg"
            $canvas.Save($pathOut, $jpegCodec, $encoderParameters)
            Write-Host $pathOut
        } finally { $canvas.Dispose() }
    }
} finally {
    $font.Dispose(); $headerFont.Dispose(); $smallFont.Dispose()
    $background.Dispose(); $textBrush.Dispose(); $pendingBrush.Dispose(); $borderPen.Dispose()
    $encoderParameters.Dispose()
}
