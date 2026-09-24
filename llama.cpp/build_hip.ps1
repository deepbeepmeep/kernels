param(
    [string]$Python = 'D:\AMD\qwen-hip210\Scripts\python.exe',
    [string]$Architectures = 'gfx1201',
    [string]$OutputDirectory = 'D:\AMD\dist',
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$sdkRoot = (& $Python -c "from pathlib import Path; import sys; print(Path(sys.prefix) / 'Lib/site-packages/_rocm_sdk_devel')").Trim()
if (!(Test-Path -LiteralPath "$sdkRoot\lib\llvm\bin\clang-cl.exe")) { throw 'Run rocm-sdk init in the HIP environment first.' }
$vcRoot = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC' -Directory | Sort-Object Name | Select-Object -Last 1
$winSdk = 'C:\Program Files (x86)\Windows Kits\10'
$winVersion = (Get-ChildItem "$winSdk\Lib" -Directory | Sort-Object Name | Select-Object -Last 1).Name
$env:ROCM_HOME = $sdkRoot
$env:HIP_PATH = $sdkRoot
$env:HIP_PLATFORM = 'amd'
$env:CC = 'clang-cl'
$env:CXX = 'clang-cl'
$env:DISTUTILS_USE_SDK = '1'
$env:PYTORCH_ROCM_ARCH = $Architectures
$env:MAX_JOBS = '2'
$env:PATH = "$sdkRoot\lib\llvm\bin;$sdkRoot\bin;$($vcRoot.FullName)\bin\Hostx64\x64;$winSdk\bin\$winVersion\x64;$(Split-Path $Python);$env:PATH"
$env:INCLUDE = "$($vcRoot.FullName)\include;$winSdk\Include\$winVersion\ucrt;$winSdk\Include\$winVersion\shared;$winSdk\Include\$winVersion\um"
$env:LIB = "$($vcRoot.FullName)\lib\x64;$winSdk\Lib\$winVersion\ucrt\x64;$winSdk\Lib\$winVersion\um\x64"
Push-Location $PSScriptRoot
try {
    $wheelArgs = @('setup.py', 'bdist_wheel', '--dist-dir', $OutputDirectory)
    if ($SkipBuild) { $wheelArgs += '--skip-build' }
    & $Python @wheelArgs
    if ($LASTEXITCODE -ne 0) { throw "HIP build failed ($LASTEXITCODE)." }
} finally { Pop-Location }
