$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$distDir = Join-Path $projectRoot "dist"
$bundlePath = Join-Path $distDir "aws-elastic-beanstalk.zip"
$stagingDir = Join-Path $distDir "aws-bundle-staging"

$excludePrefixes = @(
    ".git\",
    "venv\",
    "__pycache__\",
    ".render_local\",
    ".render_test_storage\",
    "dist\"
)

$excludeFiles = @(
    ".gitignore",
    ".ebignore"
)

if (Test-Path $stagingDir) {
    Remove-Item $stagingDir -Recurse -Force
}

New-Item -ItemType Directory -Path $stagingDir | Out-Null
New-Item -ItemType Directory -Path $distDir -Force | Out-Null

Get-ChildItem -Path $projectRoot -Recurse -File | ForEach-Object {
    $fullPath = $_.FullName
    $relativePath = $fullPath.Substring($projectRoot.Length + 1)

    if ($excludeFiles -contains $relativePath) {
        return
    }

    foreach ($prefix in $excludePrefixes) {
        if ($relativePath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            return
        }
    }

    $targetPath = Join-Path $stagingDir $relativePath
    $targetDir = Split-Path -Parent $targetPath
    if (-not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    }

    Copy-Item $fullPath $targetPath -Force
}

if (Test-Path $bundlePath) {
    Remove-Item $bundlePath -Force
}

Compress-Archive -Path (Join-Path $stagingDir '*') -DestinationPath $bundlePath -Force
Remove-Item $stagingDir -Recurse -Force

Write-Host "Created AWS Elastic Beanstalk bundle:"
Write-Host $bundlePath
