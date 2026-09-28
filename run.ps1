$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $projectPython)) {
    py -m venv venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the project Python environment." }
}
& $projectPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& $projectPython -m streamlit run app.py
