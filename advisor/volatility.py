"""LSTM volatility forecaster (TensorFlow/Keras) feeding the covariance matrix.

Black-Litterman needs a covariance matrix, and the usual choice -- the sample
covariance of trailing returns -- assumes the next quarter looks like the last
two years. It does not. Volatility clusters: calm periods follow calm periods,
and a crisis raises variance for months. A covariance estimated through
February 2020 was catastrophically wrong by March.

The standard fix is GARCH. This uses an LSTM over sequences of realised
volatility instead, which is the same idea with a more flexible functional form,
and decomposes the problem the way multivariate GARCH does:

    Sigma = D . R . D

where `R` is the correlation matrix from history and `D` is a diagonal of
*forecast* volatilities. Correlations are far more stable than variances, so
forecasting only the diagonal is where the value is and where the risk of
overfitting is lowest. Getting a 20x20 correlation matrix out of a neural
network would need vastly more data than a decade of daily returns provides.

Falls back to trailing volatility if TensorFlow is unavailable or the fit fails,
so the backtest never depends on it being installed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Keras is noisy on import and TensorFlow logs three lines about CPU features
# before doing anything. Neither is useful inside a backtest loop.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SEQUENCE_LEN = 30      # days of realised vol fed to the model
HORIZON = 21           # forecast one trading month ahead
VOL_WINDOW = 21        # window used to realise volatility in the first place
TRADING_DAYS = 252


def available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class VolForecast:
    volatility: pd.Series   # annualised, per asset
    source: str             # "lstm" or "trailing"
    train_mae: float | None = None
    val_mae: float | None = None
    baseline_mae: float | None = None

    @property
    def beat_baseline(self) -> bool:
        if self.val_mae is None or self.baseline_mae is None:
            return False
        return self.val_mae < self.baseline_mae


def realised_volatility(returns: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """Rolling annualised volatility per asset."""
    return returns.rolling(window).std().dropna() * np.sqrt(TRADING_DAYS)


def _sequences(vol: pd.DataFrame, seq_len: int, horizon: int):
    """Windows of past volatility -> volatility `horizon` days later.

    Pooled across assets, and scaled so the model predicts a *ratio* rather than
    a level: each window is divided by its own last observation, and the target
    becomes future_vol / current_vol.

    That normalisation is what makes pooling work. Trained on raw levels, one
    shared model learns the average volatility of the universe and returns
    almost the same number for every asset -- forecasts came out as 0.2499,
    0.2526, 0.2491, 0.2527 across four very different companies, which is a
    model that has thrown away the cross-section the covariance matrix exists to
    capture. Predicting the ratio keeps each asset's own level and asks the
    network only for the dynamics: mean reversion, clustering, decay.
    """
    X, y = [], []
    values = vol.to_numpy(dtype=np.float32)
    for col in range(values.shape[1]):
        series = values[:, col]
        for i in range(len(series) - seq_len - horizon):
            window = series[i:i + seq_len]
            anchor = window[-1]
            if anchor <= 1e-6:
                continue
            X.append(window / anchor)
            y.append(series[i + seq_len + horizon - 1] / anchor)
    if not X:
        return np.empty((0, seq_len, 1), np.float32), np.empty((0,), np.float32)
    return (np.asarray(X, dtype=np.float32)[..., None],
            np.asarray(y, dtype=np.float32))


def _build(seq_len: int):
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, 1)),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="softplus"),  # volatility cannot be negative
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    return model


def forecast(
    returns: pd.DataFrame,
    seq_len: int = SEQUENCE_LEN,
    horizon: int = HORIZON,
    epochs: int = 12,
    verbose: bool = False,
) -> VolForecast:
    """Forecast each asset's volatility `horizon` days ahead.

    Trained only on data in `returns`, which the backtest has already truncated
    to the rebalance date -- so this inherits the point-in-time discipline
    rather than needing its own.
    """
    vol = realised_volatility(returns)
    trailing = vol.iloc[-1]

    if not available() or len(vol) < seq_len + horizon + 60:
        return VolForecast(volatility=trailing, source="trailing")

    try:
        from tensorflow import keras

        keras.utils.set_random_seed(0)
        X, y = _sequences(vol, seq_len, horizon)
        if len(X) < 200:
            return VolForecast(volatility=trailing, source="trailing")

        # Chronological split. Shuffling would leak the future into training,
        # which on a volatility series is the difference between a model and a
        # look-ahead machine.
        cut = int(len(X) * 0.85)
        X_train, y_train, X_val, y_val = X[:cut], y[:cut], X[cut:], y[cut:]

        model = _build(seq_len)
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=256, verbose=1 if verbose else 0,
            callbacks=[keras.callbacks.EarlyStopping(
                patience=3, restore_best_weights=True, monitor="val_loss")],
        )

        # The honest benchmark: "tomorrow's volatility equals today's". Random
        # walk is genuinely hard to beat on volatility and any forecast that
        # cannot is not worth wiring into a portfolio.
        baseline_mae = float(np.mean(np.abs(X_val[:, -1, 0] - y_val)))
        val_mae = float(min(history.history["val_loss"]))

        windows = np.stack([vol[col].to_numpy(dtype=np.float32)[-seq_len:]
                            for col in vol.columns])
        anchors = windows[:, -1].copy()
        anchors[anchors <= 1e-6] = 1e-6
        ratios = model.predict((windows / anchors[:, None])[..., None], verbose=0).ravel()
        # Back to levels: the network supplies the shape, the asset supplies the scale.
        predicted = ratios * anchors

        forecast_series = pd.Series(predicted, index=vol.columns)
        # Never let the model produce something absurd; clip to a sane band
        # around what the market has actually been doing.
        forecast_series = forecast_series.clip(trailing * 0.4, trailing * 2.5)

        if val_mae > baseline_mae:
            # Beaten by persistence. Use trailing vol and say so.
            return VolForecast(volatility=trailing, source="trailing",
                               val_mae=val_mae, baseline_mae=baseline_mae,
                               train_mae=float(min(history.history["loss"])))

        return VolForecast(volatility=forecast_series, source="lstm",
                           train_mae=float(min(history.history["loss"])),
                           val_mae=val_mae, baseline_mae=baseline_mae)
    except Exception:
        return VolForecast(volatility=trailing, source="trailing")


def rebuild_covariance(sample_cov: pd.DataFrame, forecast_vol: pd.Series) -> pd.DataFrame:
    """Replace the volatilities in a covariance matrix, keep the correlations.

    Sigma = D R D, with R taken from the sample covariance and D from the
    forecast. Correlations are stable enough to trust from history; variances
    are not.
    """
    assets = sample_cov.columns
    forecast_vol = forecast_vol.reindex(assets)

    sample_vol = pd.Series(np.sqrt(np.diag(sample_cov.to_numpy())), index=assets)
    sample_vol = sample_vol.replace(0.0, np.nan)
    correlation = sample_cov.div(sample_vol, axis=0).div(sample_vol, axis=1).fillna(0.0)

    # np.fill_diagonal on `.values` fails under pandas 3 -- the backing array is
    # read-only. Work on an explicit copy.
    corr = np.array(correlation.to_numpy(), dtype=float, copy=True)
    np.fill_diagonal(corr, 1.0)

    forecast_vol = forecast_vol.fillna(sample_vol).replace(0.0, np.nan).fillna(sample_vol)
    scale = forecast_vol.to_numpy(dtype=float)
    return pd.DataFrame(corr * np.outer(scale, scale), index=assets, columns=assets)
