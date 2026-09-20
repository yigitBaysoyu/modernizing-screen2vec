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

# 1. The thesis sweep: strict protocol (the sweep's default), recipe width, three seeds.
#    Each point is evaluated with the app half and on the screen vector alone (no_app).
#    D=768 / seed 42 reuses the canonical strict checkpoints if they exist. Never pass --force.
# 2. No-app backfill for the permissive sweeps already recorded (evaluation only; only runs
#    whose checkpoints still exist qualify, i.e. normally the seed-42 points).
# The permissive sweeps themselves (hidden 1024, D = 16...1024; hidden 4096, D = 16...2048)
# are complete and are not re-run; they remain a within-protocol hidden-width check.
Invoke-Step "python 4_sweep_embedding_dimensions.py run --protocol strict --hidden_dim 1024 --dims 16,64,256,768 --seeds 42,43,44"
Invoke-Step "python 4_sweep_embedding_dimensions.py noapp --protocol permissive --hidden_dim 1024"
Invoke-Step "python 4_sweep_embedding_dimensions.py noapp --protocol permissive --hidden_dim 4096"

# Render tables, curves and the hidden-width comparison.
Invoke-Step "jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 _results_dimension_sweep.ipynb"
