"""Generate the M20 Colab training notebook."""
import json
from pathlib import Path

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": [],
}


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(True),
        "execution_count": None,
        "outputs": [],
    }


notebook["cells"] = [
    md(
        "# Stream Alpha \u2013 M20 Training on Colab\n"
        "\n"
        "Train NHITS + PatchTST ensemble on BTC/USD, ETH/USD, SOL/USD using NeuralForecast.\n"
        "\n"
        "**Before running:**\n"
        "1. Set runtime to **GPU** (Runtime \u2192 Change runtime type \u2192 T4 GPU)\n"
        "2. Upload your exported `exports/` folder to Google Drive at:\n"
        "   `My Drive/Stream_Alpha/exports/feature_ohlc_for_colab/`\n"
    ),
    md("## 1. Mount Google Drive"),
    code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "\n"
        "import os\n"
        "DATASET_DIR = '/content/drive/MyDrive/Stream_Alpha/exports/feature_ohlc_for_colab'\n"
        "assert os.path.isdir(DATASET_DIR), f'Dataset not found at {DATASET_DIR}'\n"
        "print('Dataset folders:', os.listdir(DATASET_DIR))\n"
    ),
    md("## 2. Clone the repo"),
    code(
        "!git clone https://github.com/Arashmehrdad/Stream_Alpha.git /content/Stream_Alpha\n"
        "%cd /content/Stream_Alpha\n"
    ),
    md("## 3. Install dependencies"),
    code(
        "!pip install -q \\\n"
        "    'neuralforecast>=1.7,<2' \\\n"
        "    'lightning>=2.2,<3' \\\n"
        "    'autogluon.tabular[all]==1.5.0' \\\n"
        "    'scikit-learn>=1.6,<1.7' \\\n"
        "    'numpy>=1.26,<2' \\\n"
        "    'pyarrow' \\\n"
        "    'asyncpg>=0.29,<0.30'\n"
        "\n"
        "print('\\nDependencies installed!')\n"
    ),
    md("## 4. Check GPU"),
    code(
        "import torch\n"
        "print(f'PyTorch: {torch.__version__}')\n"
        "print(f'CUDA available: {torch.cuda.is_available()}')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n"
        "    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')\n"
        "\n"
        "import psutil\n"
        "print(f'System RAM: {psutil.virtual_memory().total / 1e9:.1f} GB')\n"
    ),
    md(
        "## 5. Set artifact output to Google Drive\n"
        "\n"
        "Artifacts are saved to Drive so they survive runtime disconnections."
    ),
    code(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "# Use the A100-optimized config (larger batch sizes for GPU utilization)\n"
        "config_path = Path('/content/Stream_Alpha/configs/training.m20.colab.json')\n"
        "config = json.loads(config_path.read_text())\n"
        "\n"
        "DRIVE_ARTIFACTS = '/content/drive/MyDrive/Stream_Alpha/artifacts/training/m20'\n"
        "config['artifact_root'] = DRIVE_ARTIFACTS\n"
        "os.makedirs(DRIVE_ARTIFACTS, exist_ok=True)\n"
        "\n"
        "# Write back with Drive artifact path\n"
        "config_path.write_text(json.dumps(config, indent=2))\n"
        "print(f'Artifacts will be saved to {DRIVE_ARTIFACTS}')\n"
        "print(f'NHITS batch_size={config[\"models\"][\"neuralforecast_nhits\"][\"batch_size\"]}, '\n"
        "      f'windows_batch_size={config[\"models\"][\"neuralforecast_nhits\"][\"model_kwargs\"][\"windows_batch_size\"]}')\n"
        "print(f'PatchTST batch_size={config[\"models\"][\"neuralforecast_patchtst\"][\"batch_size\"]}, '\n"
        "      f'windows_batch_size={config[\"models\"][\"neuralforecast_patchtst\"][\"model_kwargs\"][\"windows_batch_size\"]}')\n"
    ),
    md(
        "## 6. Run M20 Training\n"
        "\n"
        "Loads data from parquet files (no PostgreSQL needed) and runs the full\n"
        "walk-forward evaluation with NHITS + PatchTST."
    ),
    code(
        "!cd /content/Stream_Alpha && python -m app.training \\\n"
        "    --config configs/training.m20.colab.json \\\n"
        '    --parquet-dir "$DATASET_DIR"\n'
    ),
    md("## 7. Check artifacts"),
    code(
        "for root, dirs, files in os.walk(DRIVE_ARTIFACTS):\n"
        "    level = root.replace(DRIVE_ARTIFACTS, '').count(os.sep)\n"
        "    indent = ' ' * 2 * level\n"
        "    print(f'{indent}{os.path.basename(root)}/')\n"
        "    subindent = ' ' * 2 * (level + 1)\n"
        "    for file in files[:10]:\n"
        "        print(f'{subindent}{file}')\n"
        "    if len(files) > 10:\n"
        "        print(f'{subindent}... and {len(files) - 10} more')\n"
    ),
    md(
        "## 8. Resume from checkpoint (optional)\n"
        "\n"
        "If the runtime disconnects, reconnect and run cells 1-5, then this cell instead of cell 6:"
    ),
    code(
        "# Uncomment to resume from a partial run:\n"
        "# !cd /content/Stream_Alpha && python -m app.training \\\n"
        "#     --config configs/training.m20.colab.json \\\n"
        '#     --parquet-dir "$DATASET_DIR" \\\n'
        '#     --resume "$DRIVE_ARTIFACTS"\n'
    ),
]

output_path = Path(__file__).parent / "M20_Training.ipynb"
output_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {output_path}")
