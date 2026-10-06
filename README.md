# MLflowExhibit

**Experiment tracking, model lifecycle management, and AI agent evaluation with MLflow.**

MLflowExhibit demonstrates practical MLflow workflows through two applications: electricity-demand forecasting and a banking assistant using sample account data. It covers training and tracking models, registering and loading a selected model version, tracing agent execution, managing versioned prompts, and evaluating response quality and tool use.

## MLflow capabilities demonstrated

| Capability | Implementation |
| --- | --- |
| Experiment tracking | Log model parameters, MAE, RMSE, MAPE, and system metrics across forecasting runs. |
| Training observability | Record RNN training and validation loss at each epoch. |
| Model artifacts | Save trained XGBoost models with their experiment runs. |
| Model Registry | Register model versions and load a selected version through the `champion` alias. |
| Agent tracing | Capture LangChain agent execution with MLflow autologging. |
| Prompt Registry | Retrieve versioned prompts for agent requests. |
| GenAI evaluation | Evaluate correctness, relevance, tool-call correctness, and tool-call efficiency alongside a custom scorer. |
| AI Gateway | Send a chat request to a separately configured gateway endpoint. |

## Applications

### Energy-demand forecasting

The forecasting workflow converts demand observations into hourly averages and builds lag, rolling-statistic, and calendar features. It trains three SARIMA configurations, three XGBoost configurations, and a PyTorch RNN baseline, recording results in the `Time_Series_Forecasting` experiment.

Saved XGBoost models can be registered as `EnergyDemandXGBoost`. The inference example loads the version assigned the `champion` alias, demonstrating how inference code can reference a selected model independently of its training run.

### Banking-agent evaluation

A LangChain agent powered by `gpt-4.1-mini` answers questions through account-balance and transaction-history tools backed by mock data. The `Agentic_AI` experiment records execution traces and evaluates two sample questions using built-in scorers and a custom response-length check. Registered prompts demonstrate versioned prompt retrieval.

The gateway client provides a separate example of calling a centrally configured model endpoint over HTTP.

## Technology stack

Python · MLflow · pandas · scikit-learn · statsmodels · XGBoost · PyTorch · LangChain · OpenAI · SQLite · uv

## Project structure

```text
MLflowExhibit/
├── data/
│   ├── raw/                      # Source energy-demand data
│   └── processed/                # Hourly series and engineered features
├── preprocess/
│   └── preprocessing.py          # Resampling and feature engineering
├── experiments/
│   ├── time_series.py            # Forecasting experiments and tracking
│   ├── model_registry.py         # XGBoost model registration
│   ├── model_load.py             # Inference using the champion alias
│   ├── agentic_ai.py             # Agent tracing, prompts, and evaluation
│   └── endpoint_call.py          # AI Gateway request example
├── src/mlflowexhibit/             # Package scaffold
├── pyproject.toml
└── uv.lock
```

## Getting started

Use Python 3.11 and `uv`. Run commands from the project root. The forecasting and agent workflows can be run independently.

### 1. Install dependencies and start MLflow

```bash
uv sync --locked
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

Open <http://127.0.0.1:5000>. Leave the server running and use a second terminal for the following commands. Scripts are configured to use this address. Local MLflow metadata, artifacts, and `.env` credentials are excluded from Git.

### 2. Run forecasting experiments

The input file, `data/raw/Energy_Demand.csv`, must contain `SETTLEMENTDATE` and `TOTALDEMAND` columns.

```bash
mkdir -p data/processed
uv run python preprocess/preprocessing.py
uv run --with torch python experiments/time_series.py
```

Open `Time_Series_Forecasting` in MLflow to inspect run parameters, forecast errors, system metrics, and RNN learning curves. PyTorch is supplied through `--with torch` because it is not currently declared in the project lockfile.

### 3. Register and load a model

Copy the run ID of a completed XGBoost run into `run_id` in `experiments/model_registry.py`, then run:

```bash
uv run python experiments/model_registry.py
```

In the MLflow Model Registry, assign the alias `champion` to the desired version of `EnergyDemandXGBoost`. Run inference with:

```bash
uv run python experiments/model_load.py
```

Predictions are printed for the processed feature table.

### 4. Run the banking agent

Add your API key to `.env` in the project root:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

The agent requires prompt versions `Trans/3` and `Trans/4` in the connected MLflow Prompt Registry. Create them using the setup example below, or update the two prompt URIs in `experiments/agentic_ai.py` to reference your own versions.

<details>
<summary>Prompt setup for a fresh MLflow database</summary>

Run this once when no prompt named `Trans` exists. It creates four sample versions so the script's version references resolve. Subsequent executions create additional versions.

```bash
uv run python - <<'PYTHON'
import mlflow
import mlflow.genai

mlflow.set_tracking_uri("http://127.0.0.1:5000")
templates = [
    "What is the balance of account A100?",
    "Show me the recent transactions for account A100.",
    "List the recent transactions for account A100 briefly.",
    "What is the balance and recent transaction history of account A100?",
]
for template in templates:
    prompt = mlflow.genai.register_prompt(name="Trans", template=template)
    print(f"Created Trans version {prompt.version}")
PYTHON
```

</details>

```bash
uv run python experiments/agentic_ai.py
```

Open `Agentic_AI` in MLflow to inspect traces and evaluation results. Agent and model-based evaluation calls require API access and may incur charges; evaluation scorers may use additional judge models.

### 5. Call the AI Gateway (optional)

In the MLflow UI, configure a chat-compatible AI Gateway endpoint named `agentic-ai-endpoint`, including its model provider credentials. Then run:

```bash
uv run python experiments/endpoint_call.py
```

The client calls `/gateway/agentic-ai-endpoint/mlflow/invocations` on the local server and prints the JSON response. This endpoint is configured independently of the banking agent; running the agent script does not deploy its Python tools to the gateway.

## Evaluation scope

This project demonstrates local MLflow workflows. Forecasting metrics should be interpreted with their evaluation settings: SARIMA forecasts the full test period, while XGBoost and the RNN use observed history for their input features. SARIMA and XGBoost use an 80/20 split; the RNN uses a 70/15/15 split. These runs do not constitute a controlled comparison across identical forecast horizons and test sets.

Model persistence and registration are implemented for XGBoost. The model-loading script demonstrates inference on the processed dataset. Agent evaluation uses two illustrative questions and mock banking data; the custom conciseness scorer checks for responses shorter than five words.
