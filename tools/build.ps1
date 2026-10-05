param([switch]$RunTests)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$sharedPackages = python -c "import sysconfig; print(sysconfig.get_path('purelib'))"
if (-not (Test-Path -LiteralPath $python)) {
    python -m venv --system-site-packages (Join-Path $workspace '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
# Reuse the existing CPU PyTorch runtime without installing a second multi-GB copy.
$sharedPackages | Set-Content -LiteralPath (Join-Path $workspace '.venv\Lib\site-packages\buckshot_shared_runtime.pth') -Encoding ascii
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (Test-Path -LiteralPath $vswhere) {
    $visualStudio = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($visualStudio) {
        $developerShellModule = Join-Path $visualStudio 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
        Import-Module $developerShellModule
        Enter-VsDevShell -VsInstallPath $visualStudio -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
    }
}
$env:CMAKE_GENERATOR = 'Ninja'
& $python -m pip install --no-build-isolation --no-deps -e $workspace
if ($LASTEXITCODE -ne 0) { throw 'Native build failed' }
& $python -m pip install 'tensorboard>=2.18'
if ($LASTEXITCODE -ne 0) { throw 'TensorBoard installation failed' }
if ($RunTests) {
    rtk proxy $python -m pytest -q --basetemp (Join-Path $workspace 'build\pytest') $workspace\tests
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
}
