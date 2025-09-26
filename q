import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# sanidade leve
assert list(X_train.columns) == list(X_test.columns)
assert not X_train.isna().any().any() and not X_test.isna().any().any()

orders  = [(1,1,0),(0,1,1),(1,1,1),(0,1,0)]
sorders = [(0,0,0,7),(1,0,0,7),(0,0,1,7),(1,0,1,7)]

best = None
for o in orders:
    for so in sorders:
        try:
            m = SARIMAX(y_train, exog=X_train, order=o, seasonal_order=so,
                        trend='n', enforce_stationarity=False, enforce_invertibility=False)
            r = m.fit(disp=False)
            if (best is None) or (r.aic < best[0]):
                best = (r.aic, o, so, r)
        except Exception as e:
            pass

aic, o_best, so_best, res = best
H = 3
pred = res.get_forecast(steps=H, exog=X_test.iloc[:H]).predicted_mean
real = y_test.iloc[:H]
err  = (real - pred).to_numpy()

mae   = np.abs(err).mean()
smape = (np.abs(err) / ((np.abs(real.to_numpy())+np.abs(pred.to_numpy()))/2)).mean()*100
bias  = err.mean()

print("Best order:", o_best, "Best seasonal:", so_best, "AIC:", round(aic,2))
print("Real:", real.to_numpy())
print("Pred:", pred.to_numpy())
print("Err :", err)
print(f"MAE={mae:.3f} | sMAPE={smape:.2f}% | Bias={bias:.3f}")