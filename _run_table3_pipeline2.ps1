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

Invoke-Step "python 9_train_joint.py --proj_dim 768"
Invoke-Step "python 6_generate_app_context_embeddings.py --embeddings_dir embeddings_seq_768 --output_dir embeddings_seq_768_with_app"
Invoke-Step "python 4_evaluate_screen_prediction.py --embeddings_dir embeddings_seq_768_with_app --lstm_model lstm_predictor_768d.pth"
