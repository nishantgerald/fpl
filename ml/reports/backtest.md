### Walk-forward backtest — 2024-25

Model: `hgb_poisson`. Protocol: expanding-window walk-forward, retrained every 2 gameweek(s) from GW8. 16 folds, 11,832 player-gameweeks scored.

| Model | MAE | RMSE | MAE (starters) | Spearman (per GW) | P@20 | Actual pts, top 20 | Transfer call acc. | Realised gain/call |
|---|---|---|---|---|---|---|---|---|
| **ML model** | 0.757 | 1.678 | 1.565 | 0.761 | 0.512 | 8.662 | 0.838 | 3.175 |
| Baseline: train mean | 1.535 | 2.490 | 2.207 | nan | 0.044 | 1.941 | nan | nan |
| Baseline: points per game | 1.398 | 2.367 | 2.524 | 0.566 | 0.138 | 3.778 | 0.553 | 1.707 |
| Baseline: form (last 4 GW) | 1.118 | 2.258 | 2.377 | 0.684 | 0.138 | 4.134 | 0.701 | 2.211 |
| Incumbent: FCPS | 1.128 | 2.184 | 2.145 | 0.617 | 0.153 | 4.428 | 0.724 | 2.447 |
| FPL's own expected points | 0.975 | 2.043 | 1.944 | 0.752 | 0.319 | 6.475 | 0.744 | 2.677 |

Every baseline is refit at each origin from that origin's history only. FCPS rank metrics use the raw 0-1000 score; its MAE/RMSE use an isotonic calibration to points fitted on the same history.