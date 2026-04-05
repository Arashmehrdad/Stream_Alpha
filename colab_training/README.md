# Colab Training – Stream Alpha M20

Run the M20 training pipeline on Google Colab with GPU, bypassing local RAM limitations.

## Two-phase workflow (recommended)

Split GPU-intensive fitting from CPU-intensive scoring:

### Phase 1: Fit on Colab (GPU)
```bash
python -m app.training --config configs/training.m20.colab.json \
    --parquet-dir /content/drive/MyDrive/Stream_Alpha/exports/feature_ohlc_for_colab \
    --fit-only
```
This fits models on each fold and the full dataset, saving fitted estimators to
`artifacts/training/<run_id>/fitted_models/`. Copy the `fitted_models/` folder to Drive.

### Phase 2: Score locally (CPU)
Download `fitted_models/` from Drive, then:
```bash
python -m app.training --config configs/training.m20.json \
    --score-only path/to/fitted_models
```
This loads the pre-fitted models, runs scoring against your local PostgreSQL data,
and produces the full output (OOF predictions, fold metrics, winner selection, summary).

## Full pipeline (single machine)

If your machine has enough RAM for both fitting and scoring:
```bash
python -m app.training --config configs/training.m20.colab.json \
    --parquet-dir /content/drive/MyDrive/Stream_Alpha/exports/feature_ohlc_for_colab
```

## Prerequisites

1. **Export the dataset** from your local PostgreSQL:
   ```bash
   cd D:\Github\Stream_Alpha
   python -m scripts.export_feature_ohlc_for_colab --config configs/training.m20.json --out exports/feature_ohlc_for_colab
   ```
   This creates a folder `exports/feature_ohlc_for_colab/` with Parquet files partitioned by symbol (BTC_USD, ETH_USD, SOL_USD).

2. **Upload to Google Drive**:
   - Create a folder on Google Drive: `My Drive/Stream_Alpha/`
   - Upload the entire `exports/` folder there (keeping the subfolder structure)
   - Final path should be: `My Drive/Stream_Alpha/exports/feature_ohlc_for_colab/BTC_USD/*.parquet`

3. **Open the notebook** in Colab:
   - Upload `colab_training/M20_Training.ipynb` to Google Colab
   - Or open directly from the GitHub repo

## What the notebook does

1. Clones the Stream_Alpha repo
2. Installs Python dependencies
3. Mounts Google Drive to access the exported parquet dataset
4. Runs the training pipeline (fit-only or full, depending on configuration)
5. Saves artifacts back to Google Drive

## Runtime recommendations

- Use **T4 GPU** (free tier) or **A100** (Colab Pro) runtime
- T4 has 15GB GPU RAM + ~12GB system RAM
- A100 has 40GB GPU RAM + ~83GB system RAM
- Training with NHITS + PatchTST on 3 symbols × 5 folds takes ~4-5 hours on T4
