# VLM Property Assessment Pipeline

## Project Structure

```
DSM-property-condition-assessment/
├── src/                    # Core source code
│   ├── config.py          # Configuration & API keys
│   ├── data_loader.py     # XML parsing & image loading
│   └── providers/         # VLM provider implementations
│       ├── base.py        # Abstract base class
│       ├── local.py       # Ollama/Local VLM
│       ├── openai.py      # OpenAI GPT-4o
│       ├── google.py      # Google Gemini
│       └── together.py    # Together AI
├── data/
│   ├── prompts/           # Prompt templates
│   └── outputs/           # Generated results (CSV/JSON)
├── pipelines/             # Batch processing scripts
│   ├── 01_data_quality.py
│   ├── 02_score_zeroshot.py
│   └── 03_score_fewshot.py
└── notebooks/             # Interactive Jupyter notebooks
    ├── 00_setup_test.ipynb
    ├── 00_data_acquisition.ipynb
    ├── 01_data_quality_visual.ipynb
    ├── 02_score_zeroshot_visual.ipynb
    ├── 03_score_fewshot_visual.ipynb
    └── 04_regression_analysis.ipynb
```

## Quick Start

1. **Setup Environment:**
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure API Keys:**
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_key
   GOOGLE_API_KEY=your_key
   TOGETHER_API_KEY=your_key
   ```

3. **Test Setup:**
   Run `notebooks/00_setup_test.ipynb` to verify all components work.

4. **Run Analysis:**
   - Data Quality: `notebooks/01_data_quality_visual.ipynb`
   - Zero-Shot Scoring: `notebooks/02_score_zeroshot_visual.ipynb`
   - Few-Shot Scoring: `notebooks/03_score_fewshot_visual.ipynb`
   - Regression Analysis: `notebooks/04_regression_analysis.ipynb`

## Pipeline Steps

See `PIPELINE.md` for detailed execution plan.

