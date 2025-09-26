import numpy as np

print("y_train NaN:", np.isnan(y_train.values).sum(), "| Inf:", np.isinf(y_train.values).sum())
print("X_train NaN por coluna:\n", X_train.isna().sum())
print("X_train tem Inf?:", np.isinf(X_train.values).any())

# troca Inf por NaN só pra filtrar
X_train = X_train.replace([np.inf, -np.inf], np.nan)

mask = np.isfinite(y_train.values) & np.isfinite(X_train.values).all(axis=1)
print("linhas removidas:", int((~mask).sum()))

y_train_clean = y_train.loc[mask].astype("float64")
X_train_clean = X_train.loc[mask].astype("float64")

from statsmodels.tsa.statespace.sarimax import SARIMAX

mod = SARIMAX(y_train_clean,
              exog=X_train_clean,
              order=(1,1,0), seasonal_order=(0,0,0,7),
              trend='n',
              enforce_stationarity=False, enforce_invertibility=False)
res = mod.fit(disp=False)
print