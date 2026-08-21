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

# GUI component prediction (paper Table 2). Step 11 needs a GPU.
Invoke-Step "python 10_extract_components.py --out_dir component_data"
Invoke-Step "python 11_embed_component_vocab.py --data_dir component_data"

Invoke-Step "python 12_train_component_predictor.py --data_dir component_data --distance euclidean --plot"
Invoke-Step "python 13_evaluate_component_prediction.py --data_dir component_data --model component_predictor_euclidean.pth"

Invoke-Step "python 12_train_component_predictor.py --data_dir component_data --distance hierarchical --plot"
Invoke-Step "python 13_evaluate_component_prediction.py --data_dir component_data --model component_predictor_hierarchical.pth"
