param(
    [string] $ModelPath = "/data/data/org.osh26.llama/files/models/qwen3-0.6b.gguf",
    [string] $LocalModelPath = "",
    [string] $ApkPath = "app/build/outputs/apk/debug/app-debug.apk",
    [string] $Prompt = "Write one short sentence about local inference.",
    [int] $MaxTokens = 48,
    [switch] $SkipInstall,
    [switch] $SkipCpu
)

$ErrorActionPreference = "Stop"

$PackageName = "org.osh26.llama"
$ActivityName = "$PackageName/.MainActivity"
$Port = 8000
$BaseUrl = "http://127.0.0.1:$Port"

function Resolve-Adb {
    if ($env:ADB -and (Test-Path -LiteralPath $env:ADB)) {
        return $env:ADB
    }

    $candidates = @()
    if ($env:ANDROID_HOME) {
        $candidates += (Join-Path $env:ANDROID_HOME "platform-tools/adb.exe")
        $candidates += (Join-Path $env:ANDROID_HOME "platform-tools/adb")
    }
    if ($env:ANDROID_SDK_ROOT) {
        $candidates += (Join-Path $env:ANDROID_SDK_ROOT "platform-tools/adb.exe")
        $candidates += (Join-Path $env:ANDROID_SDK_ROOT "platform-tools/adb")
    }
    if ($env:LOCALAPPDATA) {
        $candidates += (Join-Path $env:LOCALAPPDATA "Android/Sdk/platform-tools/adb.exe")
    }

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    $cmd = Get-Command adb -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    throw "adb not found. Set `$env:ADB or ANDROID_HOME/ANDROID_SDK_ROOT."
}

function Invoke-Adb {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]] $AdbArgs)
    & $script:Adb @AdbArgs
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed: $($AdbArgs -join ' ')"
    }
}

function Invoke-JsonPost {
    param(
        [string] $Path,
        [hashtable] $Body
    )

    $json = $Body | ConvertTo-Json -Depth 8 -Compress
    Invoke-RestMethod -Method Post -Uri "$BaseUrl$Path" -ContentType "application/json" -Body $json -TimeoutSec 300
}

function Stage-Model {
    param([string] $Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Local model not found: $Path"
    }

    $tmpPath = "/data/local/tmp/osh26-runtime-model.gguf"
    $appRelPath = "files/models/runtime-model.gguf"
    $appAbsPath = "/data/data/$PackageName/$appRelPath"

    Write-Host "staging model: $Path -> $appAbsPath"
    Invoke-Adb push $Path $tmpPath
    Invoke-Adb shell chmod 644 $tmpPath
    Invoke-Adb shell run-as $PackageName mkdir -p files/models
    Invoke-Adb shell run-as $PackageName cp $tmpPath $appRelPath

    return $appAbsPath
}

function Wait-Health {
    for ($i = 0; $i -lt 30; $i++) {
        try {
            return Invoke-RestMethod -Method Get -Uri "$BaseUrl/health" -TimeoutSec 2
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    throw "HTTP server did not become ready on $BaseUrl"
}

function Load-Backend {
    param(
        [string] $Backend,
        [int] $GpuLayers
    )

    $load = Invoke-JsonPost "/load_model" @{
        path = $ModelPath
        backend = $Backend
        n_gpu_layers = $GpuLayers
    }
    Write-Host "$Backend load_model:"
    $load | ConvertTo-Json -Depth 8 | Write-Host
    return $load
}

function Invoke-Completion {
    param([string] $Label)

    $completion = Invoke-JsonPost "/v1/chat/completions" @{
        model = "local-gguf"
        max_tokens = $MaxTokens
        temperature = 0.0
        top_p = 1.0
        seed = 51966
        thinking = $false
        messages = @(
            @{
                role = "user"
                content = $Prompt
            }
        )
    }
    Write-Host "$Label completion:"
    $completion | ConvertTo-Json -Depth 8 | Write-Host
    return $completion
}

function Get-CompletionText {
    param($Completion)

    if ($Completion.choices -and $Completion.choices.Count -gt 0 -and $Completion.choices[0].message) {
        return [string] $Completion.choices[0].message.content
    }
    return ""
}

function Get-Health {
    param([string] $Label)

    $stats = Invoke-RestMethod -Method Get -Uri "$BaseUrl/health" -TimeoutSec 10
    Write-Host "$Label health:"
    $stats | ConvertTo-Json -Depth 8 | Write-Host
    return $stats
}

function Get-LastTokenIds {
    param($Health)

    if ($Health.engine -and $Health.engine.last_token_ids) {
        return [string] $Health.engine.last_token_ids
    }
    return ""
}

$script:Adb = Resolve-Adb
$ProjectDir = $PSScriptRoot
$ResolvedApk = if ([System.IO.Path]::IsPathRooted($ApkPath)) {
    $ApkPath
} else {
    Join-Path $ProjectDir $ApkPath
}

if (-not (Test-Path -LiteralPath $ResolvedApk)) {
    throw "APK not found: $ResolvedApk. Build with .\gradlew.bat assembleDebug first."
}

Write-Host "adb: $script:Adb"
Write-Host "apk: $ResolvedApk"
Invoke-Adb devices

if (-not $SkipInstall) {
    Invoke-Adb install -r -t $ResolvedApk
}

$stagedModel = Stage-Model $LocalModelPath
if ($stagedModel) {
    $ModelPath = $stagedModel
}

Invoke-Adb logcat -c
Invoke-Adb shell am force-stop $PackageName
Invoke-Adb forward tcp:$Port tcp:$Port
Invoke-Adb shell am start -n $ActivityName | Out-Host

$health = Wait-Health
Write-Host "health:"
$health | ConvertTo-Json -Depth 8 | Write-Host

$cpuText = ""
if (-not $SkipCpu) {
    Load-Backend "cpu" 0 | Out-Null
    $cpuCompletion = Invoke-Completion "cpu"
    $cpuText = Get-CompletionText $cpuCompletion
    $cpuHealth = Get-Health "cpu"
    $cpuTokenIds = Get-LastTokenIds $cpuHealth
}

Load-Backend "vulkan" -1 | Out-Null
$vulkanCompletion = Invoke-Completion "vulkan"
$vulkanText = Get-CompletionText $vulkanCompletion
$vulkanHealth = Get-Health "vulkan"
$vulkanTokenIds = Get-LastTokenIds $vulkanHealth

if (-not $SkipCpu) {
    Write-Host "comparison:"
    Write-Host "CPU text    : $cpuText"
    Write-Host "Vulkan text : $vulkanText"
    if ($cpuText -eq $vulkanText) {
        Write-Host "CPU/Vulkan deterministic text match: yes"
    } else {
        Write-Host "CPU/Vulkan deterministic text match: no"
    }
    Write-Host "CPU token ids    : $cpuTokenIds"
    Write-Host "Vulkan token ids : $vulkanTokenIds"
    if ($cpuTokenIds -eq $vulkanTokenIds) {
        Write-Host "CPU/Vulkan token id match: yes"
    } else {
        Write-Host "CPU/Vulkan token id match: no"
    }
}

Write-Host "recent OSH26 logs:"
Invoke-Adb logcat -d -t 600 OSH26Vk:I OSH26Llama:I AndroidRuntime:E "*:S"
