$ErrorActionPreference = "Stop"

function Invoke-Step {
    param([string]$Command)
    Write-Host ">> $Command"
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Abbruch: Befehl fehlgeschlagen (Exit-Code $LASTEXITCODE): $Command"
        exit $LASTEXITCODE
    }
}

.\.venv\Scripts\activate

# Both projector hidden widths, so _results_dimension_sweep.ipynb can overlay them:
#   1. README recipe width (hidden 1024): D = 16 ... 1024, D=768 is the canonical run.
#   2. Wide projector (hidden 4096):      D = 16 ... 2048, the only way to reach 2048.
# Both are resumable; completed (dimension, seed) pairs are skipped on re-run.
Invoke-Step "python 4_sweep_embedding_dimensions.py run --hidden_dim 1024 --dims 16,32,64,128,256,512,768,1024"
Invoke-Step "python 4_sweep_embedding_dimensions.py run --hidden_dim 4096 --dims 16,32,64,128,256,512,768,1024,2048"

# Render tables, curves and the hidden-width comparison.
Invoke-Step "jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 _results_dimension_sweep.ipynb"
