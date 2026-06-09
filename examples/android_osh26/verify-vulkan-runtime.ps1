param(
    [string] $ApkPath = "app/build/outputs/apk/debug/app-debug.apk",
    [switch] $SkipInstall,
    [switch] $SkipCpu,
    [switch] $DebugCorrectness,
    [switch] $RunLongStress,
    [switch] $RunFullPromptSet,
    [switch] $RunModel06BFull,
    [switch] $RunModel06BQ8,
    [switch] $RunModel17BQ8,
    [string] $Model06BFullPath = "",
    [string] $Model06BQ8Path = "",
    [string] $Model17BQ8Path = ""
)

$ErrorActionPreference = "Stop"

$PackageName = "org.osh26.llama"
$ActivityName = "$PackageName/.MainActivity"
$Port = 8000
$BaseUrl = "http://127.0.0.1:$Port"
$DefaultSeed = 51966

function From-Utf8Bytes {
    param([byte[]] $Bytes)
    return [System.Text.Encoding]::UTF8.GetString($Bytes)
}

# Default model paths on device
$DeviceModel06BFull = "/data/data/$PackageName/files/models/qwen3-0.6b.gguf"
$DeviceModel06BQ8 = "/data/data/$PackageName/files/models/qwen3-0.6b-base-q8_0.gguf"
$DeviceModel17BQ8 = "/data/data/$PackageName/files/models/qwen3-1.7b-q8_0.gguf"

$TempModels = @{
    "0.6B-full" = @{ DevicePath = $DeviceModel06BFull; LocalPath = $Model06BFullPath }
    "0.6B-Q8_0" = @{ DevicePath = $DeviceModel06BQ8; LocalPath = $Model06BQ8Path }
    "1.7B-Q8_0" = @{ DevicePath = $DeviceModel17BQ8; LocalPath = $Model17BQ8Path }
}

$TestPrompts = @(
    # 1. English word instruction
    @{ Label = "english-word-instruction"; Prompt = "Please say the word apple."; MaxTokens = 6; Category = "short"; LongStress = $false }
    # 2. English fact question
    @{ Label = "english-fact-qa"; Prompt = "What is the capital of France?"; MaxTokens = 6; Category = "short"; LongStress = $false }
    # 3. Chinese to English translation
    @{ Label = "chinese-to-english"; Prompt = (From-Utf8Bytes ([byte[]](232,175,183,230,138,138,227,128,140,230,151,169,228,184,138,229,165,189,227,128,141,231,191,187,232,175,145,230,136,144,232,139,177,232,175,173,227,128,130))); MaxTokens = 6; Category = "short"; LongStress = $false }
    # 4. Chinese arithmetic
    @{ Label = "chinese-arithmetic"; Prompt = (From-Utf8Bytes ([byte[]](231,148,168,228,184,173,230,150,135,229,155,158,231,173,148,239,188,154,49,55,32,43,32,50,53,32,61,32,63))); MaxTokens = 6; Category = "short"; LongStress = $false }
    # 5. Chinese short sentence generation
    @{ Label = "chinese-sentence"; Prompt = (From-Utf8Bytes ([byte[]](232,175,183,229,134,153,228,184,128,229,143,165,229,133,179,228,186,142,230,156,172,229,156,176,230,142,168,231,144,134,231,154,132,228,184,173,230,150,135,231,159,173,229,143,165,227,128,130))); MaxTokens = 6; Category = "short"; LongStress = $false }
    # 6. Simple code generation
    @{ Label = "simple-code"; Prompt = "Write a one-line Python function to add two numbers."; MaxTokens = 6; Category = "short"; LongStress = $false }
    # 7. Long prompt stress. Disabled by default because CPU reference generation is slow on device.
    @{ Label = "long-prompt"; Prompt = "Please write a detailed explanation of how transformer neural networks work, including the attention mechanism, multi-head attention, feed-forward networks, layer normalization, positional encoding, and how these components work together to process sequential data. Explain each component's role and how they interact. Also discuss how transformers differ from RNNs and why they have become the dominant architecture for natural language processing tasks. Cover the key innovations in the original transformer paper and how subsequent research has built upon them."; MaxTokens = 6; Category = "long"; LongStress = $true }
    # 8. Same prompt three times (reproducibility test)
    @{ Label = "reproducibility-run1"; Prompt = "Explain briefly why local inference is useful."; MaxTokens = 6; Category = "reproducibility"; LongStress = $false }
    @{ Label = "reproducibility-run2"; Prompt = "Explain briefly why local inference is useful."; MaxTokens = 6; Category = "reproducibility"; LongStress = $false }
    @{ Label = "reproducibility-run3"; Prompt = "Explain briefly why local inference is useful."; MaxTokens = 6; Category = "reproducibility"; LongStress = $false }
)

$Results = @{}
$AbortVerification = $false

$PerfBaselines = @{
    "0.6B-Q8_0" = @{ Tps = 4.7800; LmHeadMs = 162.996; TtftMs = 392.802; SubmitCount = 2.0 }
    "1.7B-Q8_0" = @{ Tps = 2.2154; LmHeadMs = 363.742; TtftMs = 841.991; SubmitCount = 2.0 }
}

function Test-FlagEnabled {
    param([string] $Value)
    return -not [string]::IsNullOrEmpty($Value) -and
        $Value -ne "0" -and
        $Value -ne "false" -and
        $Value -ne "FALSE" -and
        $Value -ne "off" -and
        $Value -ne "OFF"
}

$SingleSubmitRequested = Test-FlagEnabled $env:OSH26_SINGLE_SUBMIT

function Get-Median {
    param([double[]] $Values)
    $clean = @($Values | Where-Object { -not [double]::IsNaN($_) } | Sort-Object)
    if ($clean.Count -eq 0) { return $null }
    $mid = [int][Math]::Floor($clean.Count / 2)
    if (($clean.Count % 2) -eq 1) {
        return [double]$clean[$mid]
    }
    return ([double]$clean[$mid - 1] + [double]$clean[$mid]) / 2.0
}

function Get-PerformanceSummary {
    param($ModelResults)

    $rows = @($ModelResults.vulkanResults | Where-Object { -not $_.error -and -not $_.FailReason })
    if ($rows.Count -eq 0) { return $null }

    return @{
        SampleCount = $rows.Count
        MedianTTFT = Get-Median ([double[]] @($rows | ForEach-Object { [double]$_.TTFT }))
        MedianTPS = Get-Median ([double[]] @($rows | ForEach-Object { [double]$_.TPS }))
        MedianLmHeadMs = Get-Median ([double[]] @($rows | ForEach-Object { [double]$_.LmHeadMs }))
        MedianDecodeMs = Get-Median ([double[]] @($rows | ForEach-Object { [double]$_.DecodeMs }))
        MedianSubmitCount = Get-Median ([double[]] @($rows | ForEach-Object {
            if ($_.PSObject.Properties.Name -contains "ForwardSubmitCount") { [double]$_.ForwardSubmitCount } else { [double]$_.LayerSubmitCount + [double]$_.LmHeadSubmitCount }
        }))
        MedianDescriptorAllocCount = Get-Median ([double[]] @($rows | ForEach-Object { [double]$_.DescriptorAllocCount }))
        MedianDescriptorUpdateCount = Get-Median ([double[]] @($rows | ForEach-Object { [double]$_.DescriptorUpdateCount }))
        MedianForwardGpuMs = Get-Median ([double[]] @($rows | ForEach-Object { [double]$_.LastForwardGpuMs }))
        MedianForwardLayersGpuMs = Get-Median ([double[]] @($rows | ForEach-Object { [double]$_.LastForwardLayersGpuMs }))
        MedianForwardLmHeadGpuMs = Get-Median ([double[]] @($rows | ForEach-Object { [double]$_.LastForwardLmHeadGpuMs }))
        MedianForwardFinalNormGpuMs = Get-Median ([double[]] @($rows | ForEach-Object { [double]$_.LastForwardFinalNormGpuMs }))
    }
}

function Assert-PerformanceGate {
    param([string] $ModelLabel, $ModelResults)

    if ($DebugCorrectness -or -not $PerfBaselines.ContainsKey($ModelLabel)) {
        return
    }

    $summary = Get-PerformanceSummary $ModelResults
    $ModelResults.performanceSummary = $summary
    if ($null -eq $summary) {
        $msg = "[$ModelLabel] no successful Vulkan rows for performance gate"
        $ModelResults.failures += $msg
        return
    }

    $baseline = $PerfBaselines[$ModelLabel]
    $submitLimit = if ($SingleSubmitRequested) { 1.0 } else { 3.0 }
    if ([double]$summary.MedianSubmitCount -gt $submitLimit) {
        $ModelResults.failures += "[$ModelLabel] median forward submit count $($summary.MedianSubmitCount) expected <= $submitLimit"
    }
    if ([double]$summary.MedianTPS -lt ([double]$baseline.Tps * 0.98)) {
        $ModelResults.failures += "[$ModelLabel] median TPS $($summary.MedianTPS) regressed more than 2% from baseline $($baseline.Tps)"
    }
    $ttftRegression = if ($SingleSubmitRequested) { 1.03 } else { 1.05 }
    if ([double]$summary.MedianTTFT -gt ([double]$baseline.TtftMs * $ttftRegression)) {
        $ModelResults.failures += "[$ModelLabel] median TTFT $($summary.MedianTTFT) ms regressed more than $([Math]::Round(($ttftRegression - 1.0) * 100.0, 1))% from $($baseline.TtftMs) ms"
    }
    if ([double]$summary.MedianLmHeadMs -gt ([double]$baseline.LmHeadMs * 1.05)) {
        $ModelResults.failures += "[$ModelLabel] median LM head $($summary.MedianLmHeadMs) ms regressed more than 5% from $($baseline.LmHeadMs) ms"
    }
}

function Add-PerformanceNotes {
    param([hashtable] $AllResults)

    if ($DebugCorrectness) { return }

    $notePath = Join-Path $ProjectDir "PERFORMANCE_NOTES.md"
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $lines = @(
        "",
        "## $stamp LM Head Q8 Verification",
        "",
        "Single submit requested: $SingleSubmitRequested",
        "",
        "| Model | Samples | LM head memory | Device-local bytes | Median TTFT ms | Median TPS | Median LM head ms | Median decode ms | Median forward submits | Median descriptor alloc | Forward GPU ms | Layers GPU ms | LM head GPU ms | Final norm GPU ms |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    foreach ($modelLabel in @("0.6B-Q8_0", "1.7B-Q8_0")) {
        if ($AllResults.ContainsKey($modelLabel) -and $AllResults[$modelLabel].performanceSummary) {
            $s = $AllResults[$modelLabel].performanceSummary
            $first = @($AllResults[$modelLabel].vulkanResults | Where-Object { $_.LmHeadMemoryPath } | Select-Object -First 1)
            $memPath = if ($first.Count -gt 0) { $first[0].LmHeadMemoryPath } else { "" }
            $devBytes = if ($first.Count -gt 0) { $first[0].LmHeadDeviceLocalBytes } else { 0 }
            $lines += "| $modelLabel | $($s.SampleCount) | $memPath | $devBytes | $([Math]::Round([double]$s.MedianTTFT, 3)) | $([Math]::Round([double]$s.MedianTPS, 4)) | $([Math]::Round([double]$s.MedianLmHeadMs, 3)) | $([Math]::Round([double]$s.MedianDecodeMs, 3)) | $([Math]::Round([double]$s.MedianSubmitCount, 1)) | $([Math]::Round([double]$s.MedianDescriptorAllocCount, 1)) | $([Math]::Round([double]$s.MedianForwardGpuMs, 3)) | $([Math]::Round([double]$s.MedianForwardLayersGpuMs, 3)) | $([Math]::Round([double]$s.MedianForwardLmHeadGpuMs, 3)) | $([Math]::Round([double]$s.MedianForwardFinalNormGpuMs, 3)) |"
        }
    }
    $lines += ""
    $lines += "Gate criteria: forward submit count <= $(if ($SingleSubmitRequested) { 1 } else { 3 }), TTFT regresses no more than $(if ($SingleSubmitRequested) { '3%' } else { '5%' }) from the device-local baseline, TPS regresses no more than 2%, and LM head regresses no more than 5%."
    foreach ($modelLabel in @("0.6B-Q8_0", "1.7B-Q8_0")) {
        if ($AllResults.ContainsKey($modelLabel) -and $AllResults[$modelLabel].performanceSummary) {
            $modelFailures = @($AllResults[$modelLabel].failures | Where-Object { $_ -like "*median*" })
            if ($modelFailures.Count -eq 0) {
                $lines += "Gate status ${modelLabel}: PASS"
            } else {
                $lines += "Gate status ${modelLabel}: FAIL - $($modelFailures -join '; ')"
            }
        }
    }
    Add-Content -Path $notePath -Value $lines
}

function Get-ActivePrompts {
    $active = @($TestPrompts)
    if (-not $RunFullPromptSet) {
        $quickLabels = @(
            "english-word-instruction",
            "chinese-arithmetic",
            "reproducibility-run1",
            "reproducibility-run2",
            "reproducibility-run3"
        )
        $active = @($active | Where-Object { ($quickLabels -contains $_.Label) -or ($RunLongStress -and [bool]$_.LongStress) })
    } elseif (-not $RunLongStress) {
        $active = @($active | Where-Object { -not [bool]$_.LongStress })
    }
    return $active
}

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
    Invoke-RestMethod -Method Post -Uri "$BaseUrl$Path" -ContentType "application/json" -Body $json -TimeoutSec 900
}

function Wait-Health {
    for ($i = 0; $i -lt 300; $i++) {
        try {
            return Invoke-RestMethod -Method Get -Uri "$BaseUrl/health" -TimeoutSec 10
        } catch {
            if (($i % 15) -eq 0) {
                Write-Host "waiting for health... $($i + 1)s"
            }
            Start-Sleep -Seconds 2
        }
    }
    throw "HTTP server did not become ready on $BaseUrl"
}

function Load-Backend {
    param(
        [string] $ModelPath,
        [string] $Backend,
        [int] $GpuLayers
    )

    $load = Invoke-JsonPost "/load_model" @{
        path = $ModelPath
        backend = $Backend
        n_gpu_layers = $GpuLayers
        debug_correctness = [bool] $DebugCorrectness
    }
    Write-Host "$Backend load_model ($ModelPath): $($load | ConvertTo-Json -Depth 2 -Compress)"
    return $load
}

function Assert-DeviceFile {
    param(
        [string] $ModelLabel,
        [string] $DevicePath
    )

    try {
        Invoke-Adb shell run-as $PackageName ls $DevicePath | Out-Null
    } catch {
        throw "[$ModelLabel] model file is not readable by app: $DevicePath"
    }
}

function Get-Health {
    param([string] $Label)

    $stats = Invoke-RestMethod -Method Get -Uri "$BaseUrl/health" -TimeoutSec 30
    return $stats
}

function Invoke-Completion {
    param(
        [string] $Label,
        [string] $Prompt,
        [int] $MaxTokens = 16,
        [string] $PromptLabel = ""
    )

    if ([string]::IsNullOrEmpty($PromptLabel)) {
        $PromptLabel = $Label
    }

    Write-Host "  [$PromptLabel] sending prompt: $($Prompt.Substring(0, [Math]::Min(80, $Prompt.Length)))..."

    $completion = Invoke-JsonPost "/v1/chat/completions" @{
        model = "local-gguf"
        max_tokens = $MaxTokens
        temperature = 0.0
        top_p = 1.0
        seed = $DefaultSeed
        thinking = $false
        messages = @(
            @{
                role = "user"
                content = $Prompt
            }
        )
    }

    return $completion
}

function Get-CompletionText {
    param($Completion)

    if ($Completion.choices -and $Completion.choices.Count -gt 0 -and $Completion.choices[0].message) {
        return [string] $Completion.choices[0].message.content
    }
    return ""
}

function Get-LastTokenIds {
    param($Health)

    if ($Health.engine -and $Health.engine.last_token_ids) {
        return [string] $Health.engine.last_token_ids
    }
    return ""
}

function Assert-HealthGate {
    param($Health, [string] $ModelLabel, [string] $Backend, [hashtable] $ResultObj)

    $vk = $Health.engine.vulkan
    if (-not $vk) {
        if ($Backend -eq "cpu") {
            Write-Host "  CPU backend: skipping Vulkan health gate"
            return
        }
        throw "[$ModelLabel] missing engine.vulkan health stats"
    }

    foreach ($field in @(
        "last_load_model_ms",
        "last_prefix_warm_ms",
        "last_ttft_ms",
        "last_tokens_per_second",
        "last_logits_sanity_ok",
        "last_logits_sanity_reason",
        "last_token_ids",
        "last_e2e_compare_ran",
        "last_e2e_top1_match",
        "last_e2e_top5_overlap",
        "last_e2e_cpu_top1_margin",
        "last_e2e_compare_ms"
    )) {
        if (-not ($Health.engine.PSObject.Properties.Name -contains $field)) {
            throw "[$ModelLabel] missing engine field: $field"
        }
    }

    foreach ($field in @(
        "attention_fallback_layers",
        "last_attention_max_abs_err",
        "last_logits_top5",
        "single_submit_enabled",
        "last_single_submit_used",
        "lm_head_q8_enabled",
        "lm_head_path",
        "lm_head_memory_path",
        "lm_head_device_local_bytes",
        "gpu_subgroup_size",
        "gpu_integer_dot_product_supported",
        "gpu_shader_int8_supported",
        "gpu_timestamp_period_ns",
        "gpu_timestamp_valid_bits",
        "last_lm_head_validation_ran",
        "last_lm_head_validation_ok",
        "last_lm_head_top1_match",
        "last_lm_head_top5_overlap",
        "last_lm_head_top20_overlap",
        "last_lm_head_cpu_top1_margin",
        "last_lm_head_matched_logit_max_abs_err",
        "last_lm_head_validation_ms",
        "last_forward_submit_count",
        "last_forward_gpu_ms",
        "last_forward_layers_gpu_ms",
        "last_forward_lm_head_gpu_ms",
        "last_forward_final_norm_gpu_ms",
        "last_lm_head_gpu_ms",
        "last_lm_head_actq8_gpu_ms",
        "last_lm_head_dot_gpu_ms",
        "last_lm_head_topk_gpu_ms",
        "last_lm_head_merge_gpu_ms",
        "last_descriptor_alloc_count",
        "last_descriptor_update_count",
        "last_prefix_cache_store_gpu_ms",
        "last_prefix_cache_restore_gpu_ms"
    )) {
        if (-not ($vk.PSObject.Properties.Name -contains $field)) {
            throw "[$ModelLabel] missing engine.vulkan field: $field"
        }
    }

    # Fill result object
    $ResultObj.TTFT = $Health.engine.last_ttft_ms
    $ResultObj.TPS = $Health.engine.last_tokens_per_second
    $ResultObj.PrefillMs = $vk.last_prefill_ms
    $ResultObj.DecodeMs = $vk.last_decode_ms
    $ResultObj.LmHeadMs = $vk.last_lm_head_ms
    $ResultObj.ForwardSubmitCount = $vk.last_forward_submit_count
    $ResultObj.PrefillSubmitCount = $vk.last_prefill_submit_count
    $ResultObj.LayerSubmitCount = $vk.last_layer_submit_count
    $ResultObj.LmHeadSubmitCount = $vk.last_lm_head_submit_count
    $ResultObj.TtftSubmitCount = $vk.last_ttft_submit_count
    $ResultObj.DescriptorAllocCount = $vk.last_descriptor_alloc_count
    $ResultObj.DescriptorUpdateCount = $vk.last_descriptor_update_count
    $ResultObj.PrefixCacheStoreGpuMs = $vk.last_prefix_cache_store_gpu_ms
    $ResultObj.PrefixCacheRestoreGpuMs = $vk.last_prefix_cache_restore_gpu_ms
    $ResultObj.LastForwardLayersMs = $vk.last_forward_layers_ms
    $ResultObj.LastForwardAttentionMs = $vk.last_forward_attention_ms
    $ResultObj.LastForwardKvUpdateMs = $vk.last_forward_kv_update_ms
    $ResultObj.LastForwardLmHeadMs = $vk.last_forward_lm_head_ms
    $ResultObj.LastForwardGpuMs = $vk.last_forward_gpu_ms
    $ResultObj.LastForwardLayersGpuMs = $vk.last_forward_layers_gpu_ms
    $ResultObj.LastForwardLmHeadGpuMs = $vk.last_forward_lm_head_gpu_ms
    $ResultObj.LastForwardFinalNormGpuMs = $vk.last_forward_final_norm_gpu_ms
    $ResultObj.PrefillQkvMs = $vk.last_prefill_qkv_ms
    $ResultObj.PrefillQkNormRopeMs = $vk.last_prefill_qk_norm_rope_ms
    $ResultObj.PrefillOProjMs = $vk.last_prefill_o_proj_ms
    $ResultObj.PrefillDownMs = $vk.last_prefill_down_ms
    $ResultObj.PrefillCpuPostMs = $vk.last_prefill_cpu_post_ms
    $ResultObj.PrefillAttentionMs = $vk.last_prefill_attention_ms
    $ResultObj.PrefillFfnGateUpSiluMs = $vk.last_prefill_ffn_gate_up_silu_ms
    $ResultObj.LastSubmitWaitMs = $vk.last_submit_wait_ms
    $ResultObj.LastTokenTps = $vk.last_token_tps
    $ResultObj.PrefillQ8Enabled = $vk.prefill_q8_enabled
    $ResultObj.DecodeQ8Enabled = $vk.decode_q8_enabled
    $ResultObj.LmHeadQ8Enabled = $vk.lm_head_q8_enabled
    $ResultObj.LmHeadPath = $vk.lm_head_path
    $ResultObj.LmHeadMemoryPath = $vk.lm_head_memory_path
    $ResultObj.LmHeadDeviceLocalBytes = $vk.lm_head_device_local_bytes
    $ResultObj.SingleSubmitEnabled = $vk.single_submit_enabled
    $ResultObj.LastSingleSubmitUsed = $vk.last_single_submit_used
    $ResultObj.GpuSubgroupSize = $vk.gpu_subgroup_size
    $ResultObj.GpuIntegerDotProductSupported = $vk.gpu_integer_dot_product_supported
    $ResultObj.GpuShaderInt8Supported = $vk.gpu_shader_int8_supported
    $ResultObj.GpuTimestampPeriodNs = $vk.gpu_timestamp_period_ns
    $ResultObj.GpuTimestampValidBits = $vk.gpu_timestamp_valid_bits
    $ResultObj.LmHeadGpuMs = $vk.last_lm_head_gpu_ms
    $ResultObj.LmHeadActQ8GpuMs = $vk.last_lm_head_actq8_gpu_ms
    $ResultObj.LmHeadDotGpuMs = $vk.last_lm_head_dot_gpu_ms
    $ResultObj.LmHeadTopKGpuMs = $vk.last_lm_head_topk_gpu_ms
    $ResultObj.LmHeadMergeGpuMs = $vk.last_lm_head_merge_gpu_ms
    $ResultObj.DebugCorrectness = $vk.debug_correctness
    $ResultObj.PrefixCacheEnabled = $Health.engine.prefix_cache_enabled
    $ResultObj.PrefixCacheHit = $Health.engine.last_prefix_cache_hit
    $ResultObj.LastPrefixTokens = $Health.engine.last_prefix_tokens
    $ResultObj.CpuContextActive = $Health.engine.cpu_context_active
    $ResultObj.ModelLoadMs = $Health.engine.last_load_model_ms
    $ResultObj.LastE2ECompareRan = $Health.engine.last_e2e_compare_ran
    $ResultObj.LastE2ETop1Match = $Health.engine.last_e2e_top1_match
    $ResultObj.LastE2ETop5Overlap = $Health.engine.last_e2e_top5_overlap
    $ResultObj.LastE2ECpuTop1Margin = $Health.engine.last_e2e_cpu_top1_margin
    $ResultObj.LastE2ECompareMs = $Health.engine.last_e2e_compare_ms
    $ResultObj.LastLogitsSanityOk = $Health.engine.last_logits_sanity_ok

    # Record top-5 from health
    if ($vk.last_logits_top5) {
        $ResultObj.LogitsTop5 = @($vk.last_logits_top5 | ForEach-Object { @{id = $_.id; logit = $_.logit} })
    }

    if ($Health.engine.last_e2e_cpu_top5) {
        $ResultObj.E2ECpuTop5 = @($Health.engine.last_e2e_cpu_top5)
    }
    if ($Health.engine.last_e2e_gpu_top5) {
        $ResultObj.E2EGpuTop5 = @($Health.engine.last_e2e_gpu_top5)
    }

    # Assertions
    if ($Backend -eq "vulkan") {
        if ($vk.attention_fallback_layers -ne 0) {
            $ResultObj.FailReason = "attention_fallback_layers expected 0, got $($vk.attention_fallback_layers)"
            throw "[$ModelLabel] $($ResultObj.FailReason)"
        }
        if (-not $vk.last_logits_top5 -or $vk.last_logits_top5.Count -lt 5) {
            $ResultObj.FailReason = "last_logits_top5 missing or incomplete"
            throw "[$ModelLabel] $($ResultObj.FailReason)"
        }
        $seenTokens = @{}
        $previousLogit = $null
        foreach ($item in $vk.last_logits_top5) {
            if ($null -eq $item.id -or [int]$item.id -lt 0) {
                $ResultObj.FailReason = "top5 contained invalid token id: $($item.id)"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
            if ($seenTokens.ContainsKey([int]$item.id)) {
                $ResultObj.FailReason = "top5 contained duplicate token id: $($item.id)"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
            $seenTokens[[int]$item.id] = $true
            $logitVal = [double]$item.logit
            if ([double]::IsNaN($logitVal) -or [double]::IsInfinity($logitVal)) {
                $ResultObj.FailReason = "logit is not finite: $($item.logit)"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
            if ($null -ne $previousLogit -and $logitVal -gt $previousLogit) {
                $ResultObj.FailReason = "top5 logits are not sorted descending"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
            $previousLogit = $logitVal
        }
        if ($Health.engine.last_logits_sanity_ok -eq $false) {
            $ResultObj.FailReason = "logits sanity check failed: $($Health.engine.last_logits_sanity_reason)"
            throw "[$ModelLabel] $($ResultObj.FailReason)"
        }
        if (($ModelLabel -eq "0.6B-Q8_0" -or $ModelLabel -eq "1.7B-Q8_0") -and -not [bool] $vk.lm_head_q8_enabled) {
            $ResultObj.FailReason = "lm_head_q8_enabled expected true for Q8 model, path=$($vk.lm_head_path)"
            throw "[$ModelLabel] $($ResultObj.FailReason)"
        }
        if ($SingleSubmitRequested -and -not $DebugCorrectness -and -not [bool] $vk.last_single_submit_used) {
            $ResultObj.FailReason = "OSH26_SINGLE_SUBMIT requested but last_single_submit_used=false"
            throw "[$ModelLabel] $($ResultObj.FailReason)"
        }
        if (($ModelLabel -eq "0.6B-Q8_0" -or $ModelLabel -eq "1.7B-Q8_0") -and -not $DebugCorrectness) {
            if ($vk.lm_head_memory_path -ne "device_local" -and $vk.lm_head_memory_path -ne "mixed_device_local") {
                $ResultObj.FailReason = "lm_head_memory_path expected device_local for Q8 performance run, got $($vk.lm_head_memory_path)"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
            if ([double] $vk.lm_head_device_local_bytes -le 0) {
                $ResultObj.FailReason = "lm_head_device_local_bytes expected > 0 for Q8 performance run"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
        }
        if ($DebugCorrectness) {
            if ([double] $vk.last_attention_max_abs_err -gt 1e-3) {
                $ResultObj.FailReason = "last_attention_max_abs_err expected <= 1e-3, got $($vk.last_attention_max_abs_err)"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
            if ($vk.gpu_lm_head_enabled -and -not [bool] $vk.last_lm_head_validation_ran) {
                $ResultObj.FailReason = "LM head validation did not run"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
            if ($vk.gpu_lm_head_enabled -and -not [bool] $vk.last_lm_head_validation_ok) {
                $ResultObj.FailReason = "LM head validation failed"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
            if ($vk.gpu_lm_head_enabled -and [int] $vk.last_lm_head_top5_overlap -lt 4) {
                $ResultObj.FailReason = "LM head top5 overlap expected >= 4, got $($vk.last_lm_head_top5_overlap)"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
            if ($vk.gpu_lm_head_enabled -and [int] $vk.last_lm_head_top20_overlap -lt 18) {
                $ResultObj.FailReason = "LM head top20 overlap expected >= 18, got $($vk.last_lm_head_top20_overlap)"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
            if ($vk.gpu_lm_head_enabled -and [double] $vk.last_lm_head_cpu_top1_margin -ge 1e-3 -and -not [bool] $vk.last_lm_head_top1_match) {
                $ResultObj.FailReason = "LM head top1 mismatch with CPU margin $($vk.last_lm_head_cpu_top1_margin)"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
            if (-not [bool] $Health.engine.last_e2e_compare_ran) {
                $ResultObj.FailReason = "E2E CPU/GPU first-token comparison did not run"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
            if ([int] $Health.engine.last_e2e_top5_overlap -lt 4) {
                $ResultObj.FailReason = "E2E top5 overlap expected >= 4, got $($Health.engine.last_e2e_top5_overlap)"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
            if ([double] $Health.engine.last_e2e_cpu_top1_margin -ge 1e-3 -and -not [bool] $Health.engine.last_e2e_top1_match) {
                $ResultObj.FailReason = "E2E top1 mismatch with CPU margin $($Health.engine.last_e2e_cpu_top1_margin)"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
        } else {
            if ([bool] $vk.last_lm_head_validation_ran -or [bool] $Health.engine.last_e2e_compare_ran) {
                $ResultObj.FailReason = "correctness comparison ran while debug_correctness=false"
                throw "[$ModelLabel] $($ResultObj.FailReason)"
            }
        }
    }
}

function Test-Model {
    param(
        [string] $ModelLabel,
        [string] $DeviceModelPath
    )

    Write-Host ""
    Write-Host "============================================"
    Write-Host "  Testing: $ModelLabel"
    Write-Host "  Model: $DeviceModelPath"
    Write-Host "============================================"

    Assert-DeviceFile $ModelLabel $DeviceModelPath

    $modelResults = @{
        modelLabel = $ModelLabel
        cpuResults = @()
        vulkanResults = @()
        reproTokenIds = $null
        failures = @()
    }

    # ---- CPU run ----
    # Q8 gates use Vulkan health plus the debug E2E comparison; standalone CPU
    # generation is too slow for the 1.7B matrix and does not exercise LM head.
    $runStandaloneCpu = (-not $SkipCpu) -and ($ModelLabel -eq "0.6B-full")
    if ($runStandaloneCpu) {
        Write-Host ""
        Write-Host "--- CPU Backend ---"
        $cpuLoadResult = Load-Backend $DeviceModelPath "cpu" 0
        $cpuLoadMsg = if ($cpuLoadResult -is [string]) { $cpuLoadResult } else { $cpuLoadResult.message }
        Write-Host "CPU load: $cpuLoadMsg"

        $firstRunTokenIds = $null

        foreach ($testPrompt in (Get-ActivePrompts)) {
            $label = $testPrompt.Label
            $prompt = $testPrompt.Prompt
            $maxTokens = $testPrompt.MaxTokens
            $category = $testPrompt.Category

            Write-Host "  CPU prompt: $label"
            try {
                $cpuCompletion = Invoke-Completion -Label "CPU" -Prompt $prompt -MaxTokens $maxTokens -PromptLabel $label
                $cpuHealth = Get-Health "CPU $label"
                $cpuTokenIds = Get-LastTokenIds $cpuHealth

                $resultObj = @{
                    promptLabel = $label
                    category = $category
                    maxTokens = $maxTokens
                    promptLength = $prompt.Length
                    completionText = Get-CompletionText $cpuCompletion
                    tokenIds = $cpuTokenIds
                }

                # Check reproducibility for the first reproducibility run
                if ($category -eq "reproducibility" -and $label -eq "reproducibility-run1") {
                    $firstRunTokenIds = $cpuTokenIds
                }
                if ($category -eq "reproducibility" -and $label -ne "reproducibility-run1") {
                    if ($cpuTokenIds -ne $firstRunTokenIds) {
                        $errMsg = "CPU reproducibility mismatch: $label token IDs differ from run1"
                        Write-Host "    FAIL: $errMsg"
                        Write-Host "    run1: $firstRunTokenIds"
                        Write-Host "    $($label): $cpuTokenIds"
                        $resultObj.FailReason = $errMsg
                        $modelResults.failures += $errMsg
                        $modelResults.cpuResults += $resultObj
                        return $modelResults
                    }
                }

                $modelResults.cpuResults += $resultObj
                Write-Host "    text: $($resultObj.completionText)"
                Write-Host "    tokens: $cpuTokenIds"
            } catch {
                $errMsg = "CPU $label failed: $_"
                Write-Host "    FAIL: $errMsg"
                $modelResults.failures += $errMsg
                $modelResults.cpuResults += @{
                    promptLabel = $label
                    category = $category
                    error = $errMsg
                }
                return $modelResults
            }
        }
    } elseif (-not $SkipCpu) {
        Write-Host ""
        Write-Host "--- CPU Backend ---"
        Write-Host "Skipping standalone CPU backend for $ModelLabel; Vulkan/debug gates provide the correctness checks for this run."
    }

    # ---- Vulkan run ----
    Write-Host ""
    Write-Host "--- Vulkan Backend ---"
    $vkLoadResult = Load-Backend $DeviceModelPath "vulkan" -1
    $vkLoadMsg = if ($vkLoadResult -is [string]) { $vkLoadResult } else { $vkLoadResult.message }
    Write-Host "Vulkan load: $vkLoadMsg"

    $firstRunTokenIds = $null
    $perfRepeatCount = if ((-not $DebugCorrectness) -and ($ModelLabel -eq "0.6B-Q8_0" -or $ModelLabel -eq "1.7B-Q8_0")) { 3 } else { 1 }

    foreach ($testPrompt in (Get-ActivePrompts)) {
        $label = $testPrompt.Label
        $prompt = $testPrompt.Prompt
        $maxTokens = $testPrompt.MaxTokens
        $category = $testPrompt.Category

        for ($repeatIndex = 1; $repeatIndex -le $perfRepeatCount; $repeatIndex++) {
            $displayLabel = if ($perfRepeatCount -gt 1) { "$label#$repeatIndex" } else { $label }
            Write-Host "  Vulkan prompt: $displayLabel"
            try {
            $vkCompletion = Invoke-Completion -Label "Vulkan" -Prompt $prompt -MaxTokens $maxTokens -PromptLabel $displayLabel
            $vkHealth = Get-Health "Vulkan $displayLabel"
            $vkTokenIds = Get-LastTokenIds $vkHealth

            $resultObj = @{
                promptLabel = $label
                repeatIndex = $repeatIndex
                category = $category
                maxTokens = $maxTokens
                promptLength = $prompt.Length
                completionText = Get-CompletionText $vkCompletion
                tokenIds = $vkTokenIds
            }

            # Run health assertions (fills resultObj with timing data)
            Assert-HealthGate $vkHealth $ModelLabel "vulkan" $resultObj

            # Check reproducibility
            if ($category -eq "reproducibility" -and $label -eq "reproducibility-run1") {
                $firstRunTokenIds = $vkTokenIds
            }
            if ($category -eq "reproducibility" -and $label -ne "reproducibility-run1") {
                if ($vkTokenIds -ne $firstRunTokenIds) {
                    $errMsg = "Vulkan reproducibility mismatch: $label token IDs differ from run1"
                    Write-Host "    FAIL: $errMsg"
                    Write-Host "    run1: $firstRunTokenIds"
                    Write-Host "    $($label): $vkTokenIds"
                    $resultObj.FailReason = $errMsg
                    $modelResults.failures += $errMsg
                    $modelResults.vulkanResults += $resultObj
                    return $modelResults
                } elseif ($modelResults.reproTokenIds -eq $null) {
                    $modelResults.reproTokenIds = $firstRunTokenIds
                }
            }

            # Compare full greedy token sequence with CPU reference when available.
            $cpuResult = $modelResults.cpuResults | Where-Object { $_ -and $_.promptLabel -eq $label } | Select-Object -First 1
            if ($cpuResult -and $cpuResult.tokenIds -and $resultObj.tokenIds) {
                if ($cpuResult.tokenIds -ne $resultObj.tokenIds) {
                    $errMsg = "CPU/Vulkan token ID mismatch for $label"
                    Write-Host "    FAIL: $errMsg"
                    Write-Host "    CPU tokens: $($cpuResult.tokenIds)"
                    Write-Host "    VK  tokens: $($resultObj.tokenIds)"
                    $resultObj.FailReason = $errMsg
                    $modelResults.failures += $errMsg
                    $modelResults.vulkanResults += $resultObj
                    return $modelResults
                }
            }

            $modelResults.vulkanResults += $resultObj
            Write-Host "    text: $($resultObj.completionText)"
            Write-Host "    tokens: $vkTokenIds"
            Write-Host "    TTFT: $($resultObj.TTFT) ms, TPS: $($resultObj.TPS)"
            } catch {
            $errMsg = "Vulkan $label failed: $_"
            Write-Host "    FAIL: $errMsg"
            $modelResults.failures += $errMsg
            $modelResults.vulkanResults += @{
                promptLabel = $label
                repeatIndex = $repeatIndex
                category = $category
                error = $errMsg
            }
            return $modelResults
            }
        }
    }

    Assert-PerformanceGate $ModelLabel $modelResults
    return $modelResults
}

# ---- Main ----
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
Write-Host "debug_correctness: $DebugCorrectness"
Write-Host "run_long_stress: $RunLongStress"
Write-Host "run_full_prompt_set: $RunFullPromptSet"
Write-Host "single_submit_requested: $SingleSubmitRequested"

# Install APK
if (-not $SkipInstall) {
    Write-Host "Installing APK..."
    Invoke-Adb install -r -t $ResolvedApk
}

# Stage models that need to be copied
$TempModels.Keys | ForEach-Object {
    $key = $_
    $info = $TempModels[$key]
    if (-not [string]::IsNullOrEmpty($info.LocalPath)) {
        Write-Host "Staging model $key from $($info.LocalPath)..."
        $tmpPath = "/data/local/tmp/osh26-runtime-model-$key.gguf"
        Invoke-Adb push $info.LocalPath $tmpPath
        Invoke-Adb shell chmod 644 $tmpPath
        Invoke-Adb shell run-as $PackageName mkdir -p files/models
        Invoke-Adb shell run-as $PackageName cp $tmpPath $info.DevicePath
    }
}

# Force stop app and start fresh
Invoke-Adb logcat -c
Invoke-Adb shell am force-stop $PackageName
Invoke-Adb shell setprop debug.osh26.single_submit $(if ($SingleSubmitRequested) { "1" } else { "0" })
Invoke-Adb forward tcp:$Port tcp:$Port
Invoke-Adb shell am start -n $ActivityName | Out-Host

$health = Wait-Health
Write-Host "Initial health:"
$health | ConvertTo-Json -Depth 4 | Write-Host

# Determine which models to test
$modelsToTest = @{}
if (-not $RunModel06BFull -and -not $RunModel06BQ8 -and -not $RunModel17BQ8) {
    # Default: test all available models
    $modelsToTest["0.6B-full"] = $TempModels["0.6B-full"].DevicePath
    $modelsToTest["0.6B-Q8_0"] = $TempModels["0.6B-Q8_0"].DevicePath
    $modelsToTest["1.7B-Q8_0"] = $TempModels["1.7B-Q8_0"].DevicePath
} else {
    if ($RunModel06BFull) { $modelsToTest["0.6B-full"] = $TempModels["0.6B-full"].DevicePath }
    if ($RunModel06BQ8) { $modelsToTest["0.6B-Q8_0"] = $TempModels["0.6B-Q8_0"].DevicePath }
    if ($RunModel17BQ8) { $modelsToTest["1.7B-Q8_0"] = $TempModels["1.7B-Q8_0"].DevicePath }
}

Write-Host ""
Write-Host "Models to test: $($modelsToTest.Keys -join ', ')"

$allResults = @{}
$globalFailures = @()

foreach ($modelLabel in $modelsToTest.Keys) {
    $modelPath = $modelsToTest[$modelLabel]
    Write-Host ""
    Write-Host "============================================"
    Write-Host "  Starting model: $modelLabel"
    Write-Host "  Path: $modelPath"
    Write-Host "============================================"

    $modelResults = Test-Model $modelLabel $modelPath
    $allResults[$modelLabel] = $modelResults
    $globalFailures += $modelResults.failures
    if ($modelResults.failures.Count -gt 0) {
        $AbortVerification = $true
        break
    }
}

# Save results to JSON
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$resultsDir = Join-Path $ProjectDir "verify-results"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
}
$resultsFile = Join-Path $resultsDir "verify-results-$timestamp.json"
$globalVersion = @{
    timestamp = $timestamp
    debugCorrectness = [bool]$DebugCorrectness
    singleSubmitRequested = [bool]$SingleSubmitRequested
    runLongStress = [bool]$RunLongStress
    runFullPromptSet = [bool]$RunFullPromptSet
    seed = $DefaultSeed
    models = $allResults
    failures = $globalFailures
    totalFailures = $globalFailures.Count
    passed = ($globalFailures.Count -eq 0)
}

$globalVersion | ConvertTo-Json -Depth 8 | Out-File -Encoding utf8 -FilePath $resultsFile
Add-PerformanceNotes $allResults
Write-Host ""
Write-Host "Results saved to: $resultsFile"

# Summary
Write-Host ""
Write-Host "============================================"
Write-Host "  SUMMARY"
Write-Host "============================================"
foreach ($modelLabel in $allResults.Keys) {
    $mr = $allResults[$modelLabel]
    $vkCount = ($mr.vulkanResults | Measure-Object).Count
    $vkFails = ($mr.vulkanResults | Where-Object { $_.error -or $_.FailReason } | Measure-Object).Count
    $totalFails = ($mr.failures | Measure-Object).Count
    Write-Host ("${modelLabel}: ${vkCount} tests, ${vkFails} vulkan failures, ${totalFails} total failures")
    if ($mr.failures.Count -gt 0) {
        foreach ($f in $mr.failures) {
            Write-Host "  FAIL: $f"
        }
    }
}

Write-Host ""
Write-Host "Total failures: $($globalFailures.Count)"
if ($globalFailures.Count -eq 0) {
    Write-Host "All tests PASSED."
} else {
    Write-Host "Some tests FAILED."
}

Invoke-Adb shell setprop debug.osh26.single_submit 0

# Return nonzero exit code on failures
if ($globalFailures.Count -gt 0) {
    exit 1
}
