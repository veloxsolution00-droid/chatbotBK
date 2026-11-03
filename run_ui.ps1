$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment not found at .venv. Create it and install dependencies first:`n  uv venv .venv`n  uv pip install --python .venv\Scripts\python.exe -r requirements.txt"
}
& $python -m streamlit run app/ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
