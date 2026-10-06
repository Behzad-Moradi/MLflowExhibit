from pathlib import Path
import mlflow
import mlflow.statsmodels
import mlflow.xgboost
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SARIMA_DATA_PATH = Path("data/processed/hourly_demand.csv")
XGB_DATA_PATH = Path("data/processed/xgboost_features.csv")

TARGET = "TOTALDEMAND"

TRAIN_RATIO = 0.8

RNN_TRAIN_RATIO = 0.7
RNN_VAL_RATIO = 0.15

SEQUENCE_LENGTH = 24
HIDDEN_SIZE = 32
NUM_LAYERS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Time_Series_Forecasting")

def evaluate(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    mape = (
        np.mean(
            np.abs(
                (y_true - y_pred) / y_true
            )
        )
        * 100
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }


sarima_df = pd.read_csv(
    SARIMA_DATA_PATH,
    parse_dates=["SETTLEMENTDATE"],
)

sarima_df = sarima_df.set_index(
    "SETTLEMENTDATE"
)

sarima_split = int(
    len(sarima_df) * TRAIN_RATIO
)

sarima_train = sarima_df.iloc[
    :sarima_split
]

sarima_test = sarima_df.iloc[
    sarima_split:
]

sarima_configs = [

    {
        "order": (1, 0, 1),
        "seasonal_order": (1, 0, 1, 24),
    },

    {
        "order": (2, 0, 1),
        "seasonal_order": (1, 0, 1, 24),
    },

    {
        "order": (2, 0, 2),
        "seasonal_order": (1, 0, 1, 24),
    },
]


for config in sarima_configs:

    order = config["order"]
    seasonal_order = config["seasonal_order"]

    run_name = (
        f"SARIMA_{order}_"
        f"{seasonal_order}"
    )

    print(f"\nRunning {run_name}")

    with mlflow.start_run(
        run_name=run_name,
        log_system_metrics=True,
    ):

        mlflow.set_tag(
            "model",
            "SARIMA",
        )

        mlflow.log_param(
            "order",
            str(order),
        )

        mlflow.log_param(
            "seasonal_order",
            str(seasonal_order),
        )

        model = SARIMAX(
            sarima_train[TARGET],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted_model = model.fit(
            disp=False
        )

        predictions = fitted_model.forecast(
            steps=len(sarima_test)
        )

        metrics = evaluate(
            sarima_test[TARGET].values,
            predictions.values,
        )

        mlflow.log_metrics(metrics)

        print("Metrics:", metrics)

        # mlflow.statsmodels.log_model(
        #     fitted_model,
        #     name="SARIMA",
        # )


xgb_df = pd.read_csv(
    XGB_DATA_PATH,
    parse_dates=["SETTLEMENTDATE"],
)

xgb_df = xgb_df.set_index(
    "SETTLEMENTDATE"
)

xgb_split = int(
    len(xgb_df) * TRAIN_RATIO
)

xgb_train = xgb_df.iloc[
    :xgb_split
]

xgb_test = xgb_df.iloc[
    xgb_split:
]


X_train = xgb_train.drop(
    columns=[TARGET]
)

y_train = xgb_train[TARGET]


X_test = xgb_test.drop(
    columns=[TARGET]
)

y_test = xgb_test[TARGET]


xgb_configs = [

    {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
    },

    {
        "n_estimators": 600,
        "max_depth": 3,
        "learning_rate": 0.03,
    },

    {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
    },
]


for config in xgb_configs:

    run_name = (
        f"XGBoost_depth{config['max_depth']}_"
        f"trees{config['n_estimators']}_"
        f"lr{config['learning_rate']}"
    )

    print(f"\nRunning {run_name}")

    with mlflow.start_run(
        run_name=run_name,
        log_system_metrics=True,
    ):

        mlflow.set_tag(
            "model",
            "XGBoost",
        )

        mlflow.log_params(
            config
        )

        model = XGBRegressor(
            **config,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        metrics = evaluate(
            y_test.values,
            predictions,
        )

        mlflow.log_metrics(
            metrics
        )

        print("Metrics:", metrics)
        
        mlflow.xgboost.log_model(
             model,
             name="XGBoost",
        )

print("\nPreparing RNN data...")


rnn_df = pd.read_csv(
    SARIMA_DATA_PATH,
    parse_dates=["SETTLEMENTDATE"],
)

rnn_df = rnn_df.sort_values(
    "SETTLEMENTDATE"
)

demand = rnn_df[
    TARGET
].values.reshape(-1, 1)


n = len(demand)

train_end = int(
    n * RNN_TRAIN_RATIO
)

val_end = int(
    n * (
        RNN_TRAIN_RATIO
        + RNN_VAL_RATIO
    )
)


train_raw = demand[
    :train_end
]

val_raw = demand[
    train_end:val_end
]

test_raw = demand[
    val_end:
]


scaler = StandardScaler()


scaler.fit(
    train_raw
)


train_scaled = scaler.transform(
    train_raw
)

val_scaled = scaler.transform(
    val_raw
)

test_scaled = scaler.transform(
    test_raw
)


def create_sequences(
    data,
    sequence_length,
):

    X = []
    y = []

    for i in range(
        len(data) - sequence_length
    ):

        X.append(
            data[
                i:i + sequence_length
            ]
        )

        y.append(
            data[
                i + sequence_length
            ]
        )

    return (

        torch.tensor(
            np.array(X),
            dtype=torch.float32,
        ),

        torch.tensor(
            np.array(y),
            dtype=torch.float32,
        ),
    )


X_train_rnn, y_train_rnn = (
    create_sequences(
        train_scaled,
        SEQUENCE_LENGTH,
    )
)

X_val_rnn, y_val_rnn = (
    create_sequences(
        val_scaled,
        SEQUENCE_LENGTH,
    )
)

X_test_rnn, y_test_rnn = (
    create_sequences(
        test_scaled,
        SEQUENCE_LENGTH,
    )
)


print(
    "RNN training:",
    X_train_rnn.shape,
    y_train_rnn.shape,
)

print(
    "RNN validation:",
    X_val_rnn.shape,
    y_val_rnn.shape,
)

print(
    "RNN test:",
    X_test_rnn.shape,
    y_test_rnn.shape,
)


train_loader = DataLoader(

    TensorDataset(
        X_train_rnn,
        y_train_rnn,
    ),

    batch_size=BATCH_SIZE,

    shuffle=False,
)


val_loader = DataLoader(

    TensorDataset(
        X_val_rnn,
        y_val_rnn,
    ),

    batch_size=BATCH_SIZE,

    shuffle=False,
)

class RNNModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
    ):

        super().__init__()

        self.rnn = nn.RNN(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size,
        )


    def forward(self, x):

        output, hidden = self.rnn(x)

        final_hidden = output[
            :, -1, :
        ]

        prediction = self.fc(
            final_hidden
        )

        return prediction


model = RNNModel(

    input_size=1,

    hidden_size=HIDDEN_SIZE,

    num_layers=NUM_LAYERS,

    output_size=1,
)


criterion = nn.MSELoss()


optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE,
)


print("\nRunning RNN_baseline")


with mlflow.start_run(
    run_name="RNN_baseline",
    log_system_metrics=True,
):

    mlflow.set_tag(
        "model",
        "RNN",
    )

    mlflow.log_params({

        "sequence_length":
            SEQUENCE_LENGTH,

        "hidden_size":
            HIDDEN_SIZE,

        "num_layers":
            NUM_LAYERS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "epochs":
            EPOCHS,

        "optimizer":
            "Adam",

        "loss_function":
            "MSE",
    })

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0


        for (
            X_batch,
            y_batch,
        ) in train_loader:

            optimizer.zero_grad()

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch,
            )

            loss.backward()

            optimizer.step()


            train_loss += (
                loss.item()
                * X_batch.size(0)
            )


        train_loss /= len(
            train_loader.dataset
        )

        model.eval()

        val_loss = 0.0


        with torch.no_grad():

            for (
                X_batch,
                y_batch,
            ) in val_loader:

                predictions = model(
                    X_batch
                )

                loss = criterion(
                    predictions,
                    y_batch,
                )

                val_loss += (
                    loss.item()
                    * X_batch.size(0)
                )


        val_loss /= len(
            val_loader.dataset
        )

        mlflow.log_metric(
            "train_loss",
            train_loss,
            step=epoch + 1,
        )

        mlflow.log_metric(
            "val_loss",
            val_loss,
            step=epoch + 1,
        )


        print(
            f"Epoch "
            f"{epoch + 1:02d}/{EPOCHS} "
            f"Train Loss: "
            f"{train_loss:.6f} "
            f"Validation Loss: "
            f"{val_loss:.6f}"
        )

    model.eval()

    with torch.no_grad():

        predictions_scaled = model(
            X_test_rnn
        ).numpy()
    predictions = (
        scaler.inverse_transform(
            predictions_scaled
        )
    )


    actual = (
        scaler.inverse_transform(
            y_test_rnn.numpy()
        )
    )

    predictions = (
        predictions.flatten()
    )

    actual = (
        actual.flatten()
    )


    metrics = evaluate(
        actual,
        predictions,
    )


    mlflow.log_metrics(
        metrics
    )


    print(
        "\nRNN Test Metrics:",
        metrics,
    )

print(
    "\nAll experiments completed."
)