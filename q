import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- CONFIG BÁSICA (ajuste aqui) ---
ycol    = 'TARGET_1'          # sua target
order   = (1,1,1)
sorder  = (1,0,1,7)
H       = 3                   # horizonte
train_sizes = {'short':90, 'medium':180, 'long':365, 'full':'full'}  # 4 modos
max_windows_per_type = 6      # por tamanho e por tipo (para rodar rápido)

# filtra base (começa em 2023-01-01)
y = df[ycol].astype(float).copy()
y = y.asfreq('D')
y = y.loc['2023-01-01':]

def make_exog(idx: pd.DatetimeIndex) -> pd.DataFrame:
    dow = idx.dayofweek.values
    X = pd.DataFrame({
        'const':       np.ones(len(idx)),
        'month_start': idx.is_month_start.astype(int).values,
        'dow_1': (dow==1).astype(int),
        'dow_2': (dow==2).astype(int),
        'dow_3': (dow==3).astype(int),
        'dow_4': (dow==4).astype(int),
        'dow_5': (dow==5).astype(int),
        'dow_6': (dow==6).astype(int),
    }, index=idx)
    return X

def make_cutoffs(y: pd.Series, train_len, want_h1_month_start: bool, limit: int):
    idx = y.index
    valid = idx[:-H]                         # fim do treino (t)
    h1 = valid + pd.Timedelta(days=1)        # primeiro dia previsto
    mask = (pd.Index(h1).is_month_start == want_h1_month_start)
    if train_len != 'full':
        starts = valid - pd.to_timedelta(train_len-1, unit='D')
        mask &= (starts >= idx.min())
    cut = valid[mask]
    return cut.sort_values().to_list()[-limit:]

def backtest(y: pd.Series, order, sorder, H, train_sizes: dict, max_windows_per_type=6):
    rows = []
    for label,len_ in train_sizes.items():
        for want,label_type in [(True,'H1_month_start'), (False,'H1_not_month_start')]:
            cutoffs = make_cutoffs(y, len_, want, max_windows_per_type)
            for t in cutoffs:
                # janela de treino
                if len_ == 'full':
                    tr_idx = pd.date_range(y.index.min(), t, freq='D')
                else:
                    tr_idx = pd.date_range(t - pd.Timedelta(days=len_-1), t, freq='D')
                te_idx  = pd.date_range(t + pd.Timedelta(days=1), t + pd.Timedelta(days=H), freq='D')

                y_tr, y_te = y.loc[tr_idx], y.loc[te_idx]
                X_tr, X_te = make_exog(tr_idx), make_exog(te_idx)

                try:
                    m = SARIMAX(y_tr, exog=X_tr, order=order, seasonal_order=sorder,
                                trend='n', enforce_stationarity=False, enforce_invertibility=False)
                    r  = m.fit(disp=False)
                    fc = r.get_forecast(steps=H, exog=X_te).predicted_mean
                except Exception as e:
                    rows.append({'train_size':label, 'type':label_type, 'train_end':t,
                                 'fit_ok':0, 'err':str(e)[:140]})
                    continue

                real = y_te.to_numpy(); pred = fc.to_numpy()
                err  = real - pred
                smape = (np.abs(err)/((np.abs(real)+np.abs(pred))/2)).mean()*100

                rows.append({
                    'train_size':label, 'type':label_type, 'train_end':t, 'fit_ok':1,
                    'H1_abs':abs(err[0]), 'H2_abs':abs(err[1]) if H>1 else np.nan,
                    'H3_abs':abs(err[2]) if H>2 else np.nan,
                    'MAE_H1H3':np.nanmean(np.abs(err)),
                    'sMAPE_H1H3':smape, 'Bias_H1H3':np.nanmean(err),
                    'H1_dow': te_idx[0].dayofweek
                })
    bt = pd.DataFrame(rows).sort_values(['train_size','type','train_end'])
    summary = (bt[bt.fit_ok==1]
               .groupby(['train_size','type'])
               [['H1_abs','H2_abs','H3_abs','MAE_H1H3','sMAPE_H1H3','Bias_H1H3']]
               .mean().round(3))
    return bt, summary

# ---- Executar ----
bt, summary = backtest(y, order, sorder, H, train_sizes, max_windows_per_type)

print("Windows testadas:", bt.shape[0], "| Fits OK:", int(bt['fit_ok'].sum()))
print("\n== Resumo por tamanho e tipo ==")
print(summary)

print("\n== Piores H1 (top 5) ==")
print(bt[bt.fit_ok==1].nlargest(5, 'H1_abs')[['train_size','type','train_end','H1_abs','H1_dow','sMAPE_H1H3']])