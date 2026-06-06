param(
    [string] $ModelPath = "/data/data/org.osh26.llama/files/models/qwen3-0.6b.gguf",
    [string] $LocalModelPath = "",
    [string] $ApkPath = "app/build/outputs/apk/debug/app-debug.apk",
    [string] $Prompt = "Write one short sentence about local inference.",
    [int] $MaxTokens = 48,
    [switch] $SkipInstall,
    [switch] $SkipCpu,
    [switch] $DebugCorrectness,
    [switch] $RunAccuracySet
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
    try {
        Invoke-Adb shell run-as $PackageName ls files/models | Out-Null
    } catch {
        Invoke-Adb shell run-as $PackageName mkdir files/models
    }
    Invoke-Adb shell run-as $PackageName cp $tmpPath $appRelPath

    return $appAbsPath
}

function Wait-Health {
    for ($i = 0; $i -lt 240; $i++) {
        try {
            return Invoke-RestMethod -Method Get -Uri "$BaseUrl/health" -TimeoutSec 2
        } catch {
            if (($i % 10) -eq 0) {
                Write-Host "waiting for health... $($i + 1)s"
            }
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
        debug_correctness = [bool] $DebugCorrectness
    }
    Write-Host "$Backend load_model:"
    $load | ConvertTo-Json -Depth 8 | Write-Host
    return $load
}

function Invoke-Completion {
    param(
        [string] $Label,
        [string] $UserPrompt = $Prompt
    )

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
                content = $UserPrompt
            }
        )
    }
    Write-Host "$Label completion:"
    $completion | ConvertTo-Json -Depth 8 | Write-Host
    return $completion
}

function Assert-VulkanHealthGate {
    param($Health)

    $vk = $Health.engine.vulkan
    if (-not $vk) {
        throw "missing engine.vulkan health stats"
    }
    foreach ($field in @(
        "last_prefill_qkv_ms",
        "last_prefill_cpu_post_ms",
        "last_prefill_attention_ms",
        "last_prefill_ffn_gate_up_silu_ms",
        "last_lm_head_gemv_ms",
        "last_lm_head_local_topk_ms",
        "last_lm_head_merge_ms",
        "last_lm_head_wait_ms",
        "last_lm_head_validation_ran",
        "last_lm_head_validation_ok",
        "last_lm_head_matched_logit_max_abs_err",
        "last_lm_head_top20_overlap",
        "last_lm_head_validation_ms"
    )) {
        if (-not ($vk.PSObject.Properties.Name -contains $field)) {
            throw "missing engine.vulkan field: $field"
        }
    }
    foreach ($field in @(
        "last_load_model_ms",
        "last_prefix_warm_ms"
    )) {
        if (-not ($Health.engine.PSObject.Properties.Name -contains $field)) {
            throw "missing engine field: $field"
        }
    }
    if ($vk.attention_fallback_layers -ne 0) {
        throw "attention_fallback_layers expected 0, got $($vk.attention_fallback_layers)"
    }
    if ($DebugCorrectness -and [double] $vk.last_attention_max_abs_err -gt 1e-3) {
        throw "last_attention_max_abs_err expected <= 1e-3, got $($vk.last_attention_max_abs_err)"
    }
    if (-not $vk.last_logits_top5 -or $vk.last_logits_top5.Count -lt 5) {
        throw "last_logits_top5 missing or incomplete"
    }
    if ($DebugCorrectness -and $vk.gpu_lm_head_enabled) {
        if (-not [bool] $vk.last_lm_head_validation_ran) {
            throw "LM head validation did not run"
        }
        if (-not [bool] $vk.last_lm_head_validation_ok) {
            throw "LM head validation failed at stage $($vk.last_lm_head_validation_stage)"
        }
        if ([int] $vk.last_lm_head_top5_overlap -lt 4) {
            throw "LM head top5 overlap expected >= 4, got $($vk.last_lm_head_top5_overlap)"
        }
        if ([int] $vk.last_lm_head_top20_overlap -lt 18) {
            throw "LM head top20 overlap expected >= 18, got $($vk.last_lm_head_top20_overlap)"
        }
        if ([double] $vk.last_lm_head_cpu_top1_margin -ge 1e-3 -and -not [bool] $vk.last_lm_head_top1_match) {
            throw "LM head top1 mismatch with CPU margin $($vk.last_lm_head_cpu_top1_margin)"
        }
        if (-not [bool] $Health.engine.last_e2e_compare_ran) {
            throw "E2E CPU/GPU first-token comparison did not run"
        }
        if ([double] $Health.engine.last_e2e_cpu_top1_margin -ge 1e-3 -and -not [bool] $Health.engine.last_e2e_top1_match) {
            throw "E2E top1 mismatch with CPU margin $($Health.engine.last_e2e_cpu_top1_margin)"
        }
    } elseif ([bool] $vk.last_lm_head_validation_ran -or [bool] $Health.engine.last_e2e_compare_ran) {
        throw "correctness comparison ran while debug_correctness=false"
    }
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
    if ($stats.engine -and $stats.engine.vulkan) {
        $vk = $stats.engine.vulkan
        Write-Host ("{0} timings: load={1}ms prefixWarm={2}ms qkv={3}ms cpuPost={4}ms attn={5}ms ffn={6}ms lmHead={7}ms lmWait={8}ms prefill={9}ms decode={10}ms ttft={11}ms tps={12}" -f `
            $Label,
            $stats.engine.last_load_model_ms,
            $stats.engine.last_prefix_warm_ms,
            $vk.last_prefill_qkv_ms,
            $vk.last_prefill_cpu_post_ms,
            $vk.last_prefill_attention_ms,
            $vk.last_prefill_ffn_gate_up_silu_ms,
            $vk.last_lm_head_ms,
            $vk.last_lm_head_wait_ms,
            $vk.last_prefill_ms,
            $vk.last_decode_ms,
            $stats.engine.last_ttft_ms,
            $stats.engine.last_tokens_per_second)
    }
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
if (-not $SkipCpu -and -not $RunAccuracySet) {
    Load-Backend "cpu" 0 | Out-Null
    $cpuCompletion = Invoke-Completion "cpu"
    $cpuText = Get-CompletionText $cpuCompletion
    $cpuHealth = Get-Health "cpu"
    $cpuTokenIds = Get-LastTokenIds $cpuHealth
}

$accuracyPrompts = if ($RunAccuracySet) {
    @(
        "Please say the word apple.",
        [System.Text.Encoding]::UTF8.GetString([byte[]](231,148,168,228,184,173,230,150,135,229,155,158,231,173,148,239,188,154,49,43,49,231,173,137,228,186,142,229,135,160,239,188,159)),
        [System.Text.Encoding]::UTF8.GetString([byte[]](232,175,183,229,134,153,228,184,128,229,143,165,228,184,173,230,150,135,233,151,174,229,128,153))
    )
} else {
    @($Prompt)
}

Load-Backend "vulkan" -1 | Out-Null
$vulkanText = ""
$vulkanTokenIds = ""
foreach ($casePrompt in $accuracyPrompts) {
    Write-Host "vulkan prompt: $casePrompt"
    $vulkanCompletion = Invoke-Completion "vulkan" $casePrompt
    $vulkanText = Get-CompletionText $vulkanCompletion
    $vulkanHealth = Get-Health "vulkan"
    Assert-VulkanHealthGate $vulkanHealth
    $vulkanTokenIds = Get-LastTokenIds $vulkanHealth
    Write-Host "vulkan token ids: $vulkanTokenIds"
}

if (-not $SkipCpu -and -not $RunAccuracySet) {
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

