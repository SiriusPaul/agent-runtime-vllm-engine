param(
    [string] $GlslcPath = "C:\Users\18259\AppData\Local\Android\Sdk\ndk\28.2.13676358\shader-tools\windows-x86_64\glslc.exe"
)

$ErrorActionPreference = "Stop"

function Write-SpvHeader {
    param(
        [Parameter(Mandatory = $true)][string] $SourcePath,
        [Parameter(Mandatory = $true)][string] $HeaderPath,
        [Parameter(Mandatory = $true)][string] $SymbolName
    )

    $spvPath = [System.IO.Path]::ChangeExtension($HeaderPath, ".spv")
    & $GlslcPath -fshader-stage=compute '--target-env=vulkan1.1' $SourcePath -o $spvPath
    if ($LASTEXITCODE -ne 0) {
        throw "glslc failed for $SourcePath"
    }

    $bytes = [System.IO.File]::ReadAllBytes($spvPath)
    $sb = [System.Text.StringBuilder]::new()
    [void]$sb.AppendLine("#pragma once")
    [void]$sb.AppendLine("const unsigned char $SymbolName[]={")

    for ($i = 0; $i -lt $bytes.Length; $i += 16) {
        $count = [Math]::Min(16, $bytes.Length - $i)
        $slice = $bytes[$i..($i + $count - 1)]
        $line = "    " + (($slice | ForEach-Object { ('0x{0:X2}' -f $_) }) -join ", ")
        if ($i + $count -lt $bytes.Length) {
            $line += ","
        }
        [void]$sb.AppendLine($line)
    }

    [void]$sb.AppendLine("};")
    [void]$sb.AppendLine("const unsigned int ${SymbolName}_len = $($bytes.Length);")
    [System.IO.File]::WriteAllText($HeaderPath, $sb.ToString())
    Remove-Item $spvPath -Force
}

Write-SpvHeader `
    -SourcePath "examples/android_osh26/app/src/main/cpp/attention_prefill_kblock_qk.comp" `
    -HeaderPath "examples/android_osh26/app/src/main/cpp/attention_prefill_kblock_qk.spv.h" `
    -SymbolName "_tmp_attention_prefill_kblock_qk_spv"

Write-SpvHeader `
    -SourcePath "examples/android_osh26/app/src/main/cpp/attention_prefill_kblock_softmax_online.comp" `
    -HeaderPath "examples/android_osh26/app/src/main/cpp/attention_prefill_kblock_softmax_online.spv.h" `
    -SymbolName "_tmp_attention_prefill_kblock_softmax_online_spv"

Write-SpvHeader `
    -SourcePath "examples/android_osh26/app/src/main/cpp/attention_prefill_kblock_qkv_acc.comp" `
    -HeaderPath "examples/android_osh26/app/src/main/cpp/attention_prefill_kblock_qkv_acc.spv.h" `
    -SymbolName "_tmp_attention_prefill_kblock_qkv_acc_spv"

Write-SpvHeader `
    -SourcePath "examples/android_osh26/app/src/main/cpp/attention_prefill_kblock_finalize.comp" `
    -HeaderPath "examples/android_osh26/app/src/main/cpp/attention_prefill_kblock_finalize.spv.h" `
    -SymbolName "_tmp_attention_prefill_kblock_finalize_spv"
