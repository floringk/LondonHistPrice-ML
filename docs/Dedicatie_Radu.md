# Dedicatie / raspunsuri pentru Radu

> Document scris fara diacritice pentru email / slide / print. Versiunea telegrafica: [`FEEDBACK_REPLIES_SCURT_RO.md`](FEEDBACK_REPLIES_SCURT_RO.md).
> Pentru cifrele curente, consultati intotdeauna [`results/model_results_summary.csv`](../results/model_results_summary.csv) si [`results/run_report.md`](../results/run_report.md) — sunt regenerate de pipeline.

Acest document raspunde punctual la cele patru intrebari de feedback si explica, pentru fiecare, **(a)** ce s-a livrat in repo si **(b)** ce inseamna numeric rezultatul (interpretare, nu doar valori).

---

## Sumar executiv (1 ecran)

**Setup experimental:** preturile istorice de tranzactionare in Londra (`history_price`, GBP), 315,674 randuri dupa curatare, splitare **calendaristica** 70/10/20 (train 1995-01 -> 2015-10, val 2015-10 -> 2018-10, test 2018-10 -> 2024-09). Acelasi preprocessor (mediana + StandardScaler pentru numerice, most_frequent + OneHotEncoder sparse pentru categorice) este aplicat la **toate** modelele.

**Tablou actual al rezultatelor (test set, RMSE in GBP):**

| Model | Familie | Test RMSE | Test R^2 | Test MAPE | Within 10% | Uplift vs naive |
|---|---|---:|---:|---:|---:|---:|
| **HistGBR** | Trees | **393,489** | **0.720** | 33.8% | 24.3% | **57.6%** |
| RandomForest | Trees | 396,947 | 0.715 | 32.9% | 23.2% | 57.2% |
| Ridge | Linear | 562,968 | 0.426 | 74.5% | 10.9% | 39.3% |
| ElasticNet | Linear | 563,113 | 0.426 | 74.4% | 10.9% | 39.3% |
| MLP_medium | NN (sklearn) | 645,104 | 0.247 | 85.2% | 10.9% | 30.5% |
| MLP_large | NN (sklearn) | 721,999 | 0.056 | 88.7% | 11.4% | 22.2% |
| TorchMLP | NN (PyTorch) | 729,943 | 0.035 | 79.6% | 10.6% | 21.3% |
| MLP_small | NN (sklearn) | 838,729 | -0.274 | 117.3% | 5.9% | 9.6% |
| NaiveMedian | Baseline | 927,646 | -0.558 | 57.0% | 3.5% | 0.0% |

**Price-bin classification (HistGBR, 5 cuantile pe `y_true`):** accuracy = **49.51%**, macro F1 = **0.495**, weighted F1 = 0.497 ([`results/price_bin_classification_summary.csv`](../results/price_bin_classification_summary.csv)).

**Walk-forward (4 fold-uri calendaristice):**

| Familie | Mean RMSE | RMSE std | **CV%** | Castiguri / 4 fold-uri |
|---|---:|---:|---:|---:|
| RandomForest | 347,500 | 84,775 | 24.4% | 2/4 (fold 1, fold 4) |
| **MLP** | 356,658 | 53,909 | **15.1%** | **2/4** (fold 2, fold 3) |

MLP este **mai stabil** (CV mai mic) decat RandomForest in walk-forward, chiar daca pe single holdout pierde la HistGBR.

---

## 1) „Accuracy" — au fost calculate?

### Ce am livrat

Problema este **regresie** pe pret continuu, deci nu exista accuracy de clasificare pe tinta originala. In schimb, raportam **patru perspective** complementare:

1. **Erori absolute de regresie** in [`results/model_results_summary.csv`](../results/model_results_summary.csv): `MAE`, `RMSE`, `test_rmse_improvement_vs_naive_pct`.
2. **Metrici „de tip accuracy" pentru regresie** (acelasi CSV, coloane noi):
   - `test_r2` = 0.720 — fractie de varianta explicata.
   - `test_mape` = 0.338 — eroare medie relativa.
   - `test_within_10pct_rate` = 0.243 — fractie de predictii cu eroare relativa sub 10%.
3. **Accuracy de clasificare pe 5 benzi de pret** (proxy didactic): discretizam `y_true` in 5 cuantile (`KBinsDiscretizer(strategy='quantile')`) si comparam cu predictiile HistGBR discretizate prin **acelasi** binner.
   - [`results/price_bin_classification_summary.csv`](../results/price_bin_classification_summary.csv): **accuracy = 49.51%**, **macro F1 = 0.495**.
   - [`results/price_bin_classification_report.csv`](../results/price_bin_classification_report.csv): precision / recall / F1 / support per banda.
   - [`results/price_bin_confusion.csv`](../results/price_bin_confusion.csv) si [`.png`](../results/price_bin_confusion.png).
4. **Per-bandeze edges** in [`results/price_bin_edges.csv`](../results/price_bin_edges.csv) — interval de pret pentru fiecare banda.

In [`results/run_report.md`](../results/run_report.md), sectiunea „Baseline Summary" include si linia `Price-bin accuracy (5 quantile bins): 49.51%`.

### Ce inseamna asta

- **R^2 = 0.720** -> modelul explica ~72% din variatie. Pentru un set rezidential cu split temporal de 6 ani in viitor, asta e un nivel respectabil; comparatia cu literatura pe predictia preturilor imobiliare arata ca R^2 in zona 0.65-0.85 este normal pentru holdout temporal.
- **MAPE = 33.8%** -> in medie, predictia se abate cu ~34% de la valoarea reala. Pare mare, dar este puternic influentat de tranzactiile mici (sub 200k GBP) unde imparteala la o valoare mica amplifica eroarea relativa.
- **Within-10% = 24.3%** -> ~1 din 4 tranzactii are eroare relativa sub 10%. Este metrica cea mai apropiata de „accuracy practic" pentru un evaluator imobiliar.
- **Price-bin accuracy = 49.51%** -> modelul nimereste banda corecta in jumatate din cazuri (vs 20% asteptat aleator pentru 5 benzi). **Macro F1 = 0.495** confirma ca performanta este uniforma pe toate cele 5 benzi, nu doar pe banda dominanta (vezi raportul per-bandeze).

Recomandare pentru slide: „Modelul atinge **49.51% accuracy pe 5 benzi de pret** si **R^2 = 0.72** pe regresia continua; eroarea medie absoluta este 216k GBP iar 24% din tranzactii cad sub 10% eroare relativa."

---

## 2) De ce RandomForest si HistGBR, nu doar regresie liniara?

### Ce am livrat

Acum in tabel apar **trei familii** rulate pe **acelasi preprocessor si aceeasi splitare calendaristica** ([`src/london_pipeline.py`](../src/london_pipeline.py)):

- **Liniare:** `Ridge(alpha=1.0)` -> 562,968 RMSE, `ElasticNet(alpha=0.001, l1_ratio=0.2)` -> 563,113 RMSE.
- **Arbori / boosting:** `HistGradientBoostingRegressor(max_depth=7, learning_rate=0.05, max_iter=160)` -> **393,489 RMSE**, `RandomForestRegressor(n_estimators=120, max_depth=20)` -> 396,947 RMSE.
- **Retele neuronale:** doua implementari (vezi punctul 3).

Toate familiile produc cate un rand in [`results/model_results_summary.csv`](../results/model_results_summary.csv); cititorul poate verifica „liniar vs arbori vs NN" direct in CSV, nu trebuie sa accepte alegerea pe incredere.

### Ce inseamna asta

- **Arbori bat modele liniare cu ~30% RMSE** (393k vs 563k). Diferenta este **mare si consistenta** pe toate metricile (R^2 0.72 vs 0.43, within-10% 24% vs 11%, MAE 216k vs 375k).
- **De ce e justificat empiric:** datele tabulare imobiliare au interactii non-liniare puternice intre feature-uri (ex: `outcode_area` x `propertyType` x `floorAreaSqM` — pretul pe metru patrat variaza dramatic de la o zona la alta, si in fiecare zona variaza intre case si apartamente). Modelele liniare pot capta doar **suma efectelor independente**, in timp ce arborii pot invata **conditional**: „daca outcode = SW si propertyType = Detached, atunci coeficientul pentru floorArea este X; altfel Y".
- **De ce nu am ales doar Ridge:** un model liniar pe acest set s-ar fi inselat sistematic pe extreme (case foarte mari sau zone foarte scumpe) iar segment-gate-ul ar fi fost si mai dur. RMSE-ul liniar de 563k este peste pragul de 1.8x al gate-ului in mai multe segmente.

Recomandare pentru slide: „Comparatia liniar / arbori / NN pe aceleasi feature-uri arata ca arborii reduc RMSE cu ~30% fata de modele liniare; aceasta este motivatia empirica pentru HistGBR / RF."

---

## 3) Retele neuronale (curs centrat pe NN)

### Ce am livrat

Doua baseline-uri NN explicite, complete cu **loss curves** si **protocol identic cu arborii** (acelasi split, acelasi preprocessor):

#### 3a. Capacity scan sklearn (`MLPRegressor`)
Trei arhitecturi `(64,)`, `(128, 64)`, `(256, 128, 64)`, toate cu:
- Activare: ReLU
- Optimizator: Adam (lr=1e-3, alpha=1e-4)
- Early stopping cu val 15% holdout intern (n_iter_no_change=15, tol=1e-4)
- `batch_size=256`, `max_iter=200`
- Tinta standardizata via `TransformedTargetRegressor(transformer=StandardScaler())` — invata pe z-score, inversa este liniara (nu amplifica erori asimetric)
- `random_state=42`

Rezultate (test RMSE): `MLP_small=838,729`, `MLP_medium=645,104`, `MLP_large=721,999`. Cea mai buna capacitate este `MLP_medium`; randul `MLPRegressor` din CSV este o oglinda a celei mai bune capacitati pentru retrocompatibilitate. Loss curves: `results/MLP_small_loss_curve.png`, `results/MLP_medium_loss_curve.png`, `results/MLP_large_loss_curve.png`.

#### 3b. PyTorch MLP ([`src/torch_mlp.py`](../src/torch_mlp.py))
- **Arhitectura:** `Linear(d, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 1)`.
- **Antrenare:** `AdamW(lr=1e-3, weight_decay=1e-4)`, `SmoothL1Loss` (Huber), `batch_size=1024`, max 80 epoci, **early stopping pe val RMSE in spatiul GBP** (patience 10).
- **Tinta:** `log1p(y)` cu **clip safe** in intervalul de log-pret antrenat (+/-0.5 marja) + initializare a bias-ului final la log-mean (evita explozia `expm1` la primul epoch).
- **Output:** randul `TorchMLP` la `model_results_summary.csv` (val RMSE **428,184**, test RMSE **729,943**), coloana `pred_TorchMLP` la `test_predictions.csv`, [`results/torch_mlp_loss_curve.png`](../results/torch_mlp_loss_curve.png), [`results/torch_mlp_history.csv`](../results/torch_mlp_history.csv) (epoch, train_loss, val_rmse).
- Se ruleaza prin `python src/torch_mlp.py` sau prin `python src/main.py --torch-mlp` / `--all`.

#### 3c. NN in walk-forward
[`src/walk_forward_validation.py`](../src/walk_forward_validation.py) ruleaza **doua familii** (RandomForest + MLP) pe 4 fold-uri calendaristice si scrie [`results/walk_forward_results.csv`](../results/walk_forward_results.csv) cu coloana `model_family`. Run report ([`results/run_report.md`](../results/run_report.md)) afiseaza mean RMSE si CV% pe fiecare familie.

#### 3d. Sectiune metodologica NN
[`methodology.md`](../methodology.md) §6.3 contine:
- Argumentul teoretic (universal approximation, Hornik et al. 1989) — ca **garantie de reprezentabilitate**, nu de generalizare.
- Argumentul empiric (Shwartz-Ziv & Armon 2022, Grinsztajn et al. 2022) — arborii bat NN-urile pe date tabulare eterogene de dimensiune medie.
- O diagrama mermaid a arhitecturii PyTorch.
- Motivatii pentru: log-target safe, dropout, early stopping, scalare a feature-urilor.

### Ce inseamna asta

- **Pe holdout-ul calendaristic single (test 2018-2024)**, NN-urile nu bat boosting-ul. Cea mai buna NN (`MLP_medium`) este la 645k RMSE, fata de 393k pentru HistGBR — **un decalaj de ~64%**. Pe TorchMLP, val RMSE = 428k (apropiat de HistGBR pe val) dar test RMSE = 730k. **Diferenta val/test = 302k** evidentiaza problema reala: **calendar shift**. Modelul NN invata distributia preturilor din epoca de antrenare; nu poate extrapola la 2024 (cand preturile medii Londra sunt cu ~70% mai mari decat in 1995-2015) pentru ca activarile continue nu permit „salt" la o noua banda de pret. Arborii o pot face prin split-uri pe `history_year`.
- **Pe walk-forward (4 fold-uri)**, MLP-ul este **mai stabil** in CV decat RandomForest (15.1% vs 24.4%) si bate RF in 2 fold-uri din 4 (2006-2012 si 2012-2018). Pe ferestre scurte unde calendar shift e mic, NN-ul este **competitiv** cu RandomForest. Aceasta este o observatie pozitiva pentru NN, contra-balansand pierderea pe single holdout.
- **Justificarea publicarii NN ca baseline si nu ca model principal:** literatura pe date tabulare arata acelasi tipar. Includerea NN-urilor este **aliniere la curs** + **comparatie corecta**, **nu cherry-picking** in favoarea NN.

Recomandare pentru slide: „NN-urile sunt incluse ca baseline pe acelasi split. Pe holdout single pierd la HistGBR din cauza calendar shift, dar pe walk-forward sunt mai stabile decat RandomForest. Aceasta este observatia publicata pe date tabulare (Shwartz-Ziv & Armon 2022)."

---

## 4) Analiza detaliata — Confusion Matrix?

### Ce am livrat

O matrice de confuzie pe regresia continua nu este standard, dar livram **trei niveluri** de analiza:

#### 4a. Diagrame de regresie (pe cel mai bun model dupa `test_rmse`)
- [`results/regression_pred_vs_actual.png`](../results/regression_pred_vs_actual.png) — scatter actual vs predictie cu linia identitatii `y = y_hat`.
- [`results/regression_residuals.png`](../results/regression_residuals.png) — doua panouri:
  - Stanga: histograma reziduurilor `y - y_hat`.
  - Dreapta: reziduuri vs valoarea prezisa (verifica daca eroarea creste cu pretul).

#### 4b. Confusion matrix pe 5 benzi de pret (proxy classification task)
- Matricea: [`results/price_bin_confusion.csv`](../results/price_bin_confusion.csv) si [`results/price_bin_confusion.png`](../results/price_bin_confusion.png) (heatmap).
- Edges (intervale de pret per banda): [`results/price_bin_edges.csv`](../results/price_bin_edges.csv).
- **Raport de clasificare:** [`results/price_bin_classification_report.csv`](../results/price_bin_classification_report.csv) cu precision/recall/F1/support per banda.
- **Sumar:** [`results/price_bin_classification_summary.csv`](../results/price_bin_classification_summary.csv) cu **accuracy = 49.51%**, **macro F1 = 0.495**, **weighted F1 = 0.497**.

#### 4c. Walk-forward + segmente
- [`results/walk_forward_results.csv`](../results/walk_forward_results.csv) — RMSE pe fold pe `model_family` (RandomForest + MLP).
- [`results/segment_metrics.csv`](../results/segment_metrics.csv) — MAE / RMSE pe `outcode_area`, `propertyType`, `price_band`.
- [`results/test_rmse_by_year.csv`](../results/test_rmse_by_year.csv) — RMSE pe an pe test set.

### Ce inseamna asta

- **Pred vs actual:** punctele formeaza un nor difuz in jurul liniei `y = y_hat`, **mai stramt sub 1M GBP** si mai larg deasupra. Modelul este precis pe segmentul de masa al pietei dar incert pe tranzactiile foarte scumpe (>1M).
- **Reziduuri:** distributia este **right-skewed** (coada lunga la dreapta), cu o concentrare in jurul lui 0 si valori mari pozitive pentru predictii care **sub-estimeaza** tranzactiile foarte scumpe. Asta e si segmentul care produce **FAIL** la segment gate in [`results/run_report.md`](../results/run_report.md) pentru `price_band:(1M, 4.4M]`, `propertyType:Semi-Detached Property`, `propertyType:Detached House`.
- **Confusion matrix pe benzi de pret:** matricea concentreaza masa pe **diagonala principala si pe diagonalele imediat adiacente**. Adica modelul, cand se inseala, se inseala in banda **vecina**, nu in extrem opus — coerent cu un model de regresie bun. Erori intre banda 1 si banda 5 sunt foarte rare.
- **Walk-forward:** mean RMSE = 347k pentru RF si 357k pentru MLP. Ambele familii cresc RMSE in fold-ul 3 (2012-2018) — perioada de crestere accelerata a preturilor. CV% pentru MLP (15%) este mai mic decat pentru RF (24%), arata **stabilitate temporala** mai buna a NN-ului in ferestre scurte.
- **Pe an** (`test_rmse_by_year.csv`): RMSE creste in 2019-2024 pentru toate modelele — semnatura tipica a calendar shift-ului. Trend-ul nu poate fi reparat doar prin features curent disponibile; ar avea nevoie de un model temporal mai bogat (ex: ARIMA pe trend + ML pe reziduuri).

Recomandare pentru slide: „Reziduurile arata sub-estimare sistematica pe tranzactiile peste 1M; matricea de confuzie pe benzi este puternic concentrata pe diagonala (confuziile sunt intre benzi vecine, nu intre extreme); walk-forward confirma stabilitatea modelului si arata NN-ul mai stabil in CV decat RandomForest."

---

## Unde apar artefactele (catalog)

| Artefact | Rol |
|----------|-----|
| [`results/model_results_summary.csv`](../results/model_results_summary.csv) | Tabel principal: Naive, Ridge, ElasticNet, RF, HistGBR, MLP_small/medium/large, MLPRegressor (mirror), TorchMLP |
| [`results/test_predictions.csv`](../results/test_predictions.csv) | Coloane `pred_*` per model pentru diagrame si confuzie |
| [`results/regression_pred_vs_actual.png`](../results/regression_pred_vs_actual.png) | Scatter actual vs predictie |
| [`results/regression_residuals.png`](../results/regression_residuals.png) | Reziduuri (histograma + vs predictie) |
| [`results/price_bin_confusion.csv`](../results/price_bin_confusion.csv) / [`.png`](../results/price_bin_confusion.png) | Confusion matrix pe 5 benzi de pret |
| [`results/price_bin_edges.csv`](../results/price_bin_edges.csv) | Intervalele de pret ale fiecarei benzi |
| [`results/price_bin_classification_summary.csv`](../results/price_bin_classification_summary.csv) | Accuracy + macro F1 + weighted F1 ale task-ului derivat |
| [`results/price_bin_classification_report.csv`](../results/price_bin_classification_report.csv) | Precision / recall / F1 / support per banda |
| `results/MLP_*_loss_curve.png` | Loss curves pentru capacity scan sklearn |
| [`results/torch_mlp_loss_curve.png`](../results/torch_mlp_loss_curve.png), [`results/torch_mlp_history.csv`](../results/torch_mlp_history.csv) | Loss curve si log epoci pentru PyTorch MLP |
| [`results/walk_forward_results.csv`](../results/walk_forward_results.csv) | RMSE pe fold, pe `model_family` (RandomForest + MLP) cu agregate per familie |
| [`results/run_report.md`](../results/run_report.md) | Sumar consolidat (baseline + accuracy de benzi + walk-forward per familie + segmente + blockers) |

---

## Cum se ruleaza totul

```powershell
# Genereaza toate cele de mai sus (sklearn pipeline + torch + walk-forward + raport + sync).
python "src/main.py" --all --sync-results
```

Sau individual:

```powershell
python "src/run_london_pipeline.py"          # baseline + Ridge/ElasticNet + MLP scan + diagnostics + bin classification
python "src/torch_mlp.py"                     # PyTorch MLP (TorchMLP row + loss curve)
python "src/walk_forward_validation.py"       # RandomForest + MLP per fold
python "src/main.py" --report-only --sync-results
```

PyTorch CPU wheel pentru Windows (daca nu e instalat):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Anexe — implementare in cod

- [`src/london_pipeline.py`](../src/london_pipeline.py) — `Ridge` / `ElasticNet` / capacity scan MLP cu `StandardScaler` target wrap, helperi `_make_mlp` (configurabil pentru `standard` / `log` / `none`), `_densify`, `_extract_loss_curve`; metrici extinse R^2/MAPE/within-10%.
- [`src/regression_diagnostics.py`](../src/regression_diagnostics.py) — diagrame + confuzie pe benzi + raport de clasificare + `plot_loss_curve`.
- [`src/run_london_pipeline.py`](../src/run_london_pipeline.py) — apel diagrame dupa predictii, persista loss curves, populeaza `experiment_registry.json` cu noile artefacte.
- [`src/torch_mlp.py`](../src/torch_mlp.py) — PyTorch MLP, log-target cu clip safe + bias init la log-mean, early stopping pe val RMSE in spatiul GBP, append-row la CSV.
- [`src/walk_forward_validation.py`](../src/walk_forward_validation.py) — walk-forward pe `RandomForest` + `MLP`, schema CSV `fold, model_family, train_rows, val_rows, rmse, rmse_mean, rmse_std, cv_pct`.
- [`src/main.py`](../src/main.py) — flag `--torch-mlp`, lista extinsa `_FROZEN_ARTIFACT_NAMES` (22 fisiere), raport per familie in walk-forward.

---

## Pentru prezentare orala — sapte propozitii

1. **Problema.** Predictia pretului de tranzactionare in Londra (`history_price`) din feature-uri ale proprietatii, evaluata pe **split temporal** (nu randomizat).
2. **Trei familii pe acelasi split.** Liniare (Ridge, ElasticNet), arbori (RandomForest, HistGBR), retele neuronale (sklearn capacity scan + PyTorch MLP). Toate cu acelasi preprocessor.
3. **Cel mai bun model.** `HistGBR` cu **test RMSE 393k** si **R^2 = 0.72**; uplift de 57.6% fata de baseline-ul naive.
4. **Accuracy de tip clasificare.** Pe 5 benzi de pret, modelul atinge **49.51% accuracy** si **macro F1 = 0.495** — clar peste sansa aleatoare (20%) si uniform pe benzi.
5. **NN includem fair.** Pe holdout single NN-urile nu bat HistGBR (calendar shift), dar pe **walk-forward** sunt **mai stabile** (CV 15% vs 24%) si castiga 2 fold-uri din 4 contra RandomForest.
6. **Analiza detaliata.** Reziduurile arata sub-estimare pe tranzactiile peste 1M GBP; matricea de confuzie pe benzi este concentrata pe diagonala; segment gate marcheaza FAIL pe Detached / Semi-Detached / price_band peste 1M.
7. **Limitari recunoscute deschis.** Calendar shift mare 2018-2024 (preturi Londra in crestere); coada heavy pe pret; assisted track depinde de date externe si nu intra in mainline.

---

*Document pentru prezentare si feedback. Pentru cifre actualizate, citati direct CSV-urile din `results/`.*
