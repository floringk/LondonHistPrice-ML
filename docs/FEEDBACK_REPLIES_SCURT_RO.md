# Feedback Radu

Pentru email / slide. Versiune extinsa cu artefacte: [`Dedicatie_Radu.md`](Dedicatie_Radu.md). Cifrele curente: [`results/model_results_summary.csv`](../results/model_results_summary.csv) si [`results/run_report.md`](../results/run_report.md).

**Cifre actuale (test set, RMSE in GBP, holdout calendaristic 2018-09 -> 2024-09):**

| Familie | Cel mai bun model | Test RMSE | Uplift vs naive |
|---|---|---:|---:|
| Trees (boosting + forest) | **HistGBR** | **393,489** | 57.6% |
| Linear | Ridge | 562,968 | 39.3% |
| NN (sklearn capacity scan) | MLP_medium | 645,104 | 30.5% |
| NN (PyTorch) | TorchMLP | 729,943 | 21.3% |
| Naive median (control) | NaiveMedian | 927,646 | 0.0% |

**Price-bin classification (HistGBR, 5 cuantile pe `y_true`):** accuracy = **49.51%**, macro F1 = **0.495**.

**Walk-forward 4 fold-uri calendaristice:** RandomForest mean RMSE 347,500 (CV 24.4%) vs MLP mean RMSE 356,658 (CV **15.1%**). MLP este **mai stabil** decat RF in CV si bate RF in 2 fold-uri din 4.

---

## 1) „Accuracy" — au fost calculate?

**Raspuns:** Problema este **regresie** pe pret continuu (`history_price` in GBP), deci nu exista *accuracy* de clasificare aplicat direct tintei. Pentru a raspunde totusi la asteptarea de „cat de bine prezice modelul", raportam:

- **R^2 test = 0.720** pentru HistGBR (varianta explicata).
- **MAPE test = 33.8%** (eroare medie relativa).
- **Within-10% rate = 24.3%** (fractie de predictii cu eroare relativa sub 10%).
- **Price-bin accuracy = 49.51%** + **macro F1 = 0.495** (proxy didactic: discretizam `y_true` in 5 cuantile si comparam cu predictiile aceluiasi model). Detalii in [`results/price_bin_classification_summary.csv`](../results/price_bin_classification_summary.csv) si [`results/price_bin_classification_report.csv`](../results/price_bin_classification_report.csv).

**Ce inseamna asta:** HistGBR explica ~72% din variatie, niimereste banda de pret corecta in 49.5% din cazuri (vs 20% aleator pentru 5 benzi), si in ~1 din 4 cazuri ajunge sub 10% eroare relativa. Pe slide poate fi citat ca „accuracy de 49.5% pe 5 benzi de pret" + „R^2 de 0.72 pe regresia continua".

---

## 2) De ce RandomForest si HistGBR, nu doar regresie liniara?

**Raspuns:** Acum in tabel apar **trei familii** rulate pe **acelasi preprocessor si aceeasi splitare temporala**:

- **Liniare:** `Ridge` 562,968 RMSE, `ElasticNet` 563,113 RMSE.
- **Arbori / boosting:** `HistGBR` 393,489 RMSE, `RandomForest` 396,947 RMSE.
- **Retele neuronale:** vezi punctul 3.

**Ce inseamna asta:** Arborii bat modelele liniare cu **~30% RMSE** (393k vs 563k) pe acelasi feature set. Decalajul nu vine din preprocesare diferita, ci din capacitatea arborilor de a captura interactii non-liniare in feature-uri tabulare eterogene (ex: `propertyType` x `outcode_area` x `floorAreaSqM`). Alegerea arborilor pentru titlu este acum justificata **empiric**, nu doar **verbal**.

---

## 3) Retele neuronale (curs centrat pe NN)

**Raspuns:** S-au adaugat **doua baseline-uri NN explicite**, complete cu loss curves si protocol identic cu arborii:

- **sklearn capacity scan** (`MLP_small`, `MLP_medium`, `MLP_large`) cu tinta standardizata via `TransformedTargetRegressor(transformer=StandardScaler())`, Adam, early stopping. Cel mai bun: `MLP_medium` (128, 64) la **645,104 RMSE test**. Loss curves: `results/MLP_*_loss_curve.png`.
- **PyTorch MLP** ([`src/torch_mlp.py`](../src/torch_mlp.py)): arhitectura `Linear -> ReLU -> Dropout(0.2) -> Linear -> ReLU -> Dropout(0.2) -> Linear`, AdamW, SmoothL1Loss, **log-target cu clip safe + initializare a bias-ului final la log-mean**, early stopping pe val RMSE in spatiul GBP. Val RMSE **428,184** (competitiv cu HistGBR pe validation), test RMSE **729,943**.
- **Walk-forward (4 fold-uri):** MLP mean RMSE = 356,658 cu **CV = 15.1%**, fata de RandomForest CV = 24.4%. **MLP-ul este mai stabil decat RF in fold-uri** si bate RF in fold-urile 2006-2012 si 2012-2018.

**Ce inseamna asta:** Pe holdout-ul calendaristic (test 2018-2024), NN-urile nu bat boosting-ul. Diferenta provine din **calendar shift**: pretul mediu in test e cu ~70% mai mare decat in train (1995-2015), iar arborii extrapoleaza mai bine prin split-urile pe `history_year` decat un NN cu activari continue. Aceasta este o observatie documentata in literatura (Shwartz-Ziv & Armon, 2022; Grinsztajn et al., 2022). Argumentul teoretic (universal approximation, Hornik et al., 1989) este mentionat in [`methodology.md`](../methodology.md) §6.3 ca **garantie de reprezentabilitate**, nu **garantie de generalizare**.

**Punctul pozitiv pentru NN:** la analiza walk-forward (4 ferestre temporale), MLP-ul este **mai stabil** in CV decat RandomForest si **castiga in 2 fold-uri din 4**. Deci pe ferestre mai scurte (unde calendar shift e mai mic), NN-ul e competitiv cu arborii.

---

## 4) Analiza detaliata — Confusion Matrix?

**Raspuns:** O matrice de confuzie pe regresia continua nu este standard, dar livram trei lucruri pentru o analiza mai amanuntita:

1. **Diagrame de regresie** pentru cel mai bun model (HistGBR):
   - [`results/regression_pred_vs_actual.png`](../results/regression_pred_vs_actual.png) — scatter cu linia identitatii.
   - [`results/regression_residuals.png`](../results/regression_residuals.png) — histograma reziduuri + reziduuri vs predictie.
2. **Confusion matrix pe 5 benzi de pret** (proxy classification task pe 5 cuantile ale `y_true`):
   - Matricea: [`results/price_bin_confusion.csv`](../results/price_bin_confusion.csv) si [`results/price_bin_confusion.png`](../results/price_bin_confusion.png).
   - Edges: [`results/price_bin_edges.csv`](../results/price_bin_edges.csv).
   - **Raport de clasificare:** [`results/price_bin_classification_report.csv`](../results/price_bin_classification_report.csv) cu precision/recall/F1/support per banda, plus [`results/price_bin_classification_summary.csv`](../results/price_bin_classification_summary.csv) cu **accuracy** = 49.51% si **macro F1** = 0.495.
3. **Walk-forward stability** pe doua familii (RandomForest + MLP) — vezi [`results/walk_forward_results.csv`](../results/walk_forward_results.csv) — si **eroare pe segmente** (outcode area, property type, price band) + pe an: `results/segment_metrics.csv`, `results/test_rmse_by_year.csv`.

**Ce inseamna asta:** Matricea de confuzie pe benzi de pret arata ca modelul confunda preponderent benzi **adiacente** (predictia cade in banda urmatoare, nu in extrem opus), ceea ce este coerent cu un model de regresie bun. Reziduurile au coada heavy-tailed la dreapta — modelul **sub-estimeaza** sistematic tranzactiile foarte scumpe (>1M GBP), zona pe care segment gate-ul o marcheaza ca **FAIL** in [`results/run_report.md`](../results/run_report.md). Pe ani, RMSE creste in 2019-2024 (calendar shift). Walk-forward arata ca aceasta variabilitate este intrinseca datasetului, nu un artefact al unei singure splitari.
