import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- 1) Definir alvo e split
ycol      = 'TARGET_1'          # ajuste se precisar
TRAIN_END = '2025-01-31'        # ajuste se precisar

y = df[ycol].astype('float64').copy()
y_train = y.loc[:TRAIN_END]
y_test  = y.loc[pd.to_datetime(TRAIN_END) + pd.Timedelta(days=1):]

# --- 2) Exógenas determinísticas SEM get_dummies, fixas e 100% numéricas
# Índices
idx_tr = y_train.index
idx_te = y_test.index

# Colunas (constante, início de mês, dummies de DOW 1..6; DOW 0 é a base)
X_train = pd.DataFrame({
    'const':       np.ones(len(idx_tr), dtype='float64'),
    'month_start': idx_tr.is_month_start.astype('int8').astype('float64'),
    'dow_1':       (idx_tr.dayofweek.values == 1).astype('float64'),
    'dow_2':       (idx_tr.dayofweek.values == 2).astype('float64'),
    'dow_3':       (idx_tr.dayofweek.values == 3).astype('float64'),
    'dow_4':       (idx_tr.dayofweek.values == 4).astype('float64'),
    'dow_5':       (idx_tr.dayofweek.values == 5).astype('float64'),
    'dow_6':       (idx_tr.dayofweek.values == 6).astype('float64'),
}, index=idx_tr)

X_test = pd.DataFrame({
    'const':       np.ones(len(idx_te), dtype='float64'),
    'month_start': idx_te.is_month_start.astype('int8').astype('float64'),
    'dow_1':       (idx_te.dayofweek.values == 1).astype('float64'),
    'dow_2':       (idx_te.dayofweek.values == 2).astype('float64'),
    'dow_3':       (idx_te.dayofweek.values == 3).astype('float64'),
    'dow_4':       (idx_te.dayofweek.values == 4).astype('float64'),
    'dow_5':       (idx_te.dayofweek.values == 5).astype('float64'),
    'dow_6':       (idx_te.dayofweek.values == 6).astype('float64'),
}, index=idx_te)

# Sanidade dura
assert X_train.shape[1] == X_test.shape[1]
assert list(X_train.columns) == list(X_test.columns)
assert not X_train.isna().any().any()
assert not X_test.isna().any().any()
assert np.isfinite(X_train.values).all() and np.isfinite(X_test.values).all()
assert y_train.dtype == 'float64' and y_test.dtype == 'float64'

# --- 3) Fit SARIMAX enxuto (d=1, s=7; intercepto via 'const' nas exógenas)
mod = SARIMAX(
    y_train,
    exog=X_train,
    order=(1,1,0),
    seasonal_order=(0,0,0,7),
    trend='n',
    enforce_stationarity=False,
    enforce_invertibility=False
)
res = mod.fit(disp=False)

# --- 4) Forecast H=3 (curto)
H = 3
pred = res.get_forecast(steps=H, exog=X_test.iloc[:H]).predicted_mean
real = y_test.iloc[:H]

# --- 5) Métricas rápidas
err   = (real - pred).to_numpy()
mae   = np.abs(err).mean()
smape = (np.abs(err) / ((np.abs(real.to_numpy()) + np.abs(pred.to_numpy()))/2)).mean()*100
bias  = err.mean()

print("AIC:", round(res.aic, 2))
print("Real H1..H3:", real.to_numpy())
print("Pred  H1..H3:", pred.to_numpy())
print("Err   H1..H3:", err)
print(f"MAE={mae:.3f} | sMAPE={smape:.2f}% | Bias={bias:.3f}")