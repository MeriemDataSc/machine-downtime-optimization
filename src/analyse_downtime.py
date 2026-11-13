# -*- coding: utf-8 -*-
# %%
"""
Machine Downtime – EDA, préparation, modèles ensemblistes, évaluation, analyse économique.
Exécution : python src/analyse_downtime.py [--data FICHIER.csv] [--out-dir DOSSIER]
"""
# %% 0. Imports & configuration ------------------------------------------------
import argparse, os, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:                     # repli si xgboost n'est pas installé
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
RS = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser(description="Analyse exploratoire et prédiction des pannes machine.")
parser.add_argument("--data", default=os.path.join(BASE_DIR, "data", "raw", "Machine_Downtime.csv"),
                    help="Chemin du CSV source (défaut : data/raw/Machine_Downtime.csv).")
parser.add_argument("--out-dir", default=os.path.join(BASE_DIR, "outputs", "downtime_analysis"),
                    help="Dossier de sortie des figures et tableaux.")
args = parser.parse_args()
CSV_PATH = args.data
OUT_DIR = args.out_dir
FIG = os.path.join(OUT_DIR, "figures"); os.makedirs(FIG, exist_ok=True)
def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name), dpi=110); plt.close()
def log(*a):
    print(*a, flush=True)

# %% 1. Chargement & exploration générale -------------------------------------
raw = pd.read_csv(CSV_PATH)
df = raw.copy()
log("=== 1. EXPLORATION ===")
log("Dimensions :", df.shape)
log(df.dtypes.to_string())
log("\nDoublons (lignes entières) :", df.duplicated().sum())
miss = pd.DataFrame({"n_manquants": df.isna().sum(), "pct": (df.isna().mean()*100).round(2)})
log("\nValeurs manquantes :\n", miss[miss.n_manquants > 0].to_string())
log("\nStatistiques descriptives :\n", df.describe().T.round(2).to_string())

plt.figure(figsize=(10, 5))
miss[miss.n_manquants > 0].sort_values("pct").pct.plot.barh(color="tab:orange")
plt.xlabel("% de valeurs manquantes"); plt.title("Valeurs manquantes par variable")
save("01_missing.png")

# %% 2. Formatage / renommage -------------------------------------------------
log("\n=== 2. FORMATAGE ===")
# a) noms de colonnes : suppression des unités et caractères corrompus ('?C' = '°C', '?m' = 'µm')
rename = {
    "Hydraulic_Pressure(bar)": "Hydraulic_Pressure",
    "Coolant_Pressure(bar)": "Coolant_Pressure",
    "Air_System_Pressure(bar)": "Air_Pressure",
    "Coolant_Temperature": "Coolant_Temp",
    "Hydraulic_Oil_Temperature(?C)": "Hydraulic_Oil_Temp",
    "Spindle_Bearing_Temperature(?C)": "Spindle_Bearing_Temp",
    "Spindle_Vibration(?m)": "Spindle_Vibration",
    "Tool_Vibration(?m)": "Tool_Vibration",
    "Spindle_Speed(RPM)": "Spindle_Speed",
    "Voltage(volts)": "Voltage",
    "Torque(Nm)": "Torque",
    "Cutting(kN)": "Cutting_Force",
}
df = df.rename(columns=rename)
# b) Date en datetime + variables dérivées
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
# c) Machine_ID = Marque-Ligne-Unité-AnnéeInstall ; Assembly_Line_No est redondante
log("Machine_ID vs Assembly_Line_No (identiques à 100 %) :\n", pd.crosstab(raw.Machine_ID, raw.Assembly_Line_No))
parts = df["Machine_ID"].str.split("-", expand=True)
df["Machine_Year"] = parts[3].astype(int)
df["Machine"] = df["Machine_ID"].str.replace("Makino-", "", regex=False).str.replace("-Unit1", "", regex=False)  # L1-2013 ...
df = df.drop(columns=["Assembly_Line_No", "Machine_ID"])
# d) cible : renommer les modalités + encodage binaire
df["Downtime"] = df["Downtime"].map({"Machine_Failure": "Failure", "No_Machine_Failure": "No_Failure"})
df["Target"] = (df["Downtime"] == "Failure").astype(int)
log("Modalités de la cible :\n", df.Downtime.value_counts().to_string())
log("Machines :", df.Machine.unique())

# %% 3. Valeurs aberrantes / incohérences physiques -----------------------------
log("\n=== 3. INCOHÉRENCES ===")
num_cols = ["Hydraulic_Pressure", "Coolant_Pressure", "Air_Pressure", "Coolant_Temp",
            "Hydraulic_Oil_Temp", "Spindle_Bearing_Temp", "Spindle_Vibration", "Tool_Vibration",
            "Spindle_Speed", "Voltage", "Torque", "Cutting_Force"]
neg_pressure = (df["Hydraulic_Pressure"] < 0).sum()
neg_vib = (df["Spindle_Vibration"] < 0).sum()
log(f"Pression hydraulique négative : {neg_pressure} | Vibration broche négative : {neg_vib}")
log(f"Vitesse broche = 0 : {(df.Spindle_Speed==0).sum()} | Couple = 0 : {(df.Torque==0).sum()} "
    f"| Coolant_Temp > 60 : {(df.Coolant_Temp>60).sum()}")
# valeurs physiquement impossibles -> NaN (imputées ensuite)
df.loc[df["Hydraulic_Pressure"] < 0, "Hydraulic_Pressure"] = np.nan
df.loc[df["Spindle_Vibration"] < 0, "Spindle_Vibration"] = np.nan
# IQR : décompte (on ne supprime rien : les extrêmes peuvent être des signaux de panne)
q1, q3 = df[num_cols].quantile(.25), df[num_cols].quantile(.75)
iqr = q3 - q1
out = ((df[num_cols] < q1 - 1.5*iqr) | (df[num_cols] > q3 + 1.5*iqr)).sum()
log("Outliers (règle IQR) par variable :\n", out.to_string())

# %% 4. Imputation (exploration) ----------------------------------------------
# Pour l'EDA : médiane par machine (pas par cible -> pas de fuite).
# Pour la modélisation : SimpleImputer(median) dans le Pipeline (fit sur le train uniquement).
log("\n=== 4. IMPUTATION ===")
log("Manquants avant :", int(df[num_cols].isna().sum().sum()))
df_eda = df.copy()
df_eda[num_cols] = df_eda.groupby("Machine")[num_cols].transform(lambda s: s.fillna(s.median()))
log("Manquants après :", int(df_eda[num_cols].isna().sum().sum()))
# contrôle : la distribution est-elle préservée ?
chk = pd.DataFrame({"moy_avant": df[num_cols].mean(), "moy_apres": df_eda[num_cols].mean(),
                    "std_avant": df[num_cols].std(), "std_apres": df_eda[num_cols].std()}).round(3)
log(chk.to_string())
# les manquants sont-ils liés à la cible ? (MCAR ?)
miss_by_target = df.groupby("Downtime")[num_cols].apply(lambda g: g.isna().mean()*100).round(2).T
log("\n% manquants selon la cible :\n", miss_by_target.to_string())

# %% 5. Analyse univariée / bivariée ------------------------------------------
log("\n=== 5. ANALYSE GRAPHIQUE ===")
palette = {"No_Failure": "#2a9d8f", "Failure": "#e63946"}
d = df_eda

# 5.1 Cible
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(data=d, x="Downtime", hue="Downtime", palette=palette, ax=ax[0], legend=False)
ax[0].set_title("Répartition de la cible")
for c in ax[0].containers: ax[0].bar_label(c)
sns.countplot(data=d, x="Machine", hue="Downtime", palette=palette, ax=ax[1]); ax[1].set_title("Pannes par machine")
sns.countplot(data=d, x="Month", hue="Downtime", palette=palette, ax=ax[2]); ax[2].set_title("Pannes par mois")
save("02_target_categ.png")

# 5.2 Histogrammes + KDE par classe
def grid(plot_fn, name, ncols=4, h=3.2):
    nrows = int(np.ceil(len(num_cols)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, h*nrows)); axes = axes.ravel()
    for ax, c in zip(axes, num_cols):
        plot_fn(c, ax); ax.set_title(c); ax.set_xlabel("")
    for ax in axes[len(num_cols):]: ax.axis("off")
    save(name)

grid(lambda c, ax: sns.histplot(data=d, x=c, hue="Downtime", palette=palette, kde=True, bins=30, ax=ax, legend=(c == num_cols[0])),
     "03_hist_hue.png")
grid(lambda c, ax: sns.kdeplot(data=d, x=c, hue="Downtime", palette=palette, fill=True, common_norm=False, ax=ax, legend=False),
     "04_kde_hue.png")
grid(lambda c, ax: sns.boxplot(data=d, x="Downtime", y=c, hue="Downtime", palette=palette, ax=ax, legend=False),
     "05_boxplot_target.png")

# 5.3 Boxplots par machine x cible
grid(lambda c, ax: sns.boxplot(data=d, x="Machine", y=c, hue="Downtime", palette=palette, ax=ax, legend=False),
     "06_boxplot_machine_hue.png")

# 5.4 Barplot des moyennes par cible (valeurs standardisées pour comparer)
z = d[num_cols].apply(lambda s: (s - s.mean())/s.std()).assign(Downtime=d.Downtime)
zm = z.groupby("Downtime")[num_cols].mean().T
zm["ecart"] = zm["Failure"] - zm["No_Failure"]
zm = zm.sort_values("ecart")
plt.figure(figsize=(9, 5))
plt.barh(zm.index, zm["ecart"], color=np.where(zm["ecart"] > 0, "#e63946", "#2a9d8f"))
plt.axvline(0, color="k", lw=.8); plt.xlabel("Écart de moyenne standardisée (Failure − No_Failure)")
plt.title("Quelles variables séparent le mieux les deux classes ?")
save("07_bar_effect_size.png")

# 5.5 Pairplot sur les variables les plus discriminantes
top = zm.ecart.abs().sort_values(ascending=False).index[:4].tolist()
g = sns.pairplot(d[top + ["Downtime"]], hue="Downtime", palette=palette, corner=True, plot_kws=dict(alpha=.4, s=12))
g.savefig(os.path.join(FIG, "08_pairplot_top4.png"), dpi=100); plt.close("all")

# 5.6 Séries temporelles : taux de panne par semaine
wk = d.set_index("Date").Target.resample("W").agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(wk.index, wk["mean"]*100, marker="o", color="#e63946"); ax.set_ylabel("% de pannes")
ax.set_title("Taux de panne hebdomadaire"); ax.axhline(d.Target.mean()*100, ls="--", color="grey")
save("09_weekly_rate.png")

# 5.7 Corrélations
corr = d[num_cols + ["Target"]].corr()
plt.figure(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .7})
plt.title("Matrice de corrélation (Pearson)")
save("10_corr_matrix.png")
log("Corrélation avec la cible :\n", corr["Target"].drop("Target").sort_values(key=abs, ascending=False).round(3).to_string())

# 5.8 Crosstab + agg
log("\n--- Crosstab Machine x Downtime (%) ---")
ct = pd.crosstab(d.Machine, d.Downtime, margins=True, normalize="index").mul(100).round(1)
log(ct.to_string())
log("\n--- Crosstab Machine x Downtime avec aggfunc (moyenne de variables clés) ---")
ct_agg = pd.crosstab(d.Machine, d.Downtime, values=d.Hydraulic_Pressure, aggfunc="mean").round(1)
log("Pression hydraulique moyenne :\n", ct_agg.to_string())
log("\n--- groupby + agg multiple ---")
agg = d.groupby(["Machine", "Downtime"]).agg(
    n=("Target", "size"),
    hyd_p_moy=("Hydraulic_Pressure", "mean"),
    torque_med=("Torque", "median"),
    cutting_moy=("Cutting_Force", "mean"),
    speed_moy=("Spindle_Speed", "mean"),
    coolant_p_moy=("Coolant_Pressure", "mean"),
    coolant_t_std=("Coolant_Temp", "std"),
).round(2)
log(agg.to_string())
# Crosstab sur variables discrétisées
d["Pressure_Band"] = pd.qcut(d.Hydraulic_Pressure, 4, labels=["Q1 bas", "Q2", "Q3", "Q4 haut"])
d["Cutting_Band"] = pd.qcut(d.Cutting_Force, 4, labels=["Q1 bas", "Q2", "Q3", "Q4 haut"])
ct2 = pd.crosstab(d.Pressure_Band, d.Cutting_Band, values=d.Target, aggfunc="mean").mul(100).round(1)
log("\nTaux de panne (%) : pression hydraulique (lignes) x force de coupe (colonnes)\n", ct2.to_string())
plt.figure(figsize=(6, 4.5)); sns.heatmap(ct2, annot=True, fmt=".0f", cmap="Reds", cbar_kws={"label": "% pannes"})
plt.title("Taux de panne : pression hydraulique × force de coupe"); save("11_crosstab_heatmap.png")

# Tests statistiques rapides
from scipy import stats
rows = []
for c in num_cols:
    a, b = d.loc[d.Target == 1, c], d.loc[d.Target == 0, c]
    u, p = stats.mannwhitneyu(a, b)
    rows.append((c, a.median(), b.median(), p))
mw = pd.DataFrame(rows, columns=["variable", "med_Failure", "med_NoFailure", "p_MannWhitney"]).set_index("variable")
log("\nTest de Mann-Whitney (Failure vs No_Failure) :\n", mw.round(4).to_string())

# %% 6. Préparation à la modélisation -----------------------------------------
log("\n=== 6. MODÉLISATION ===")
features_num = num_cols
features_cat = ["Machine"]
X = df[features_num + features_cat + ["Month", "DayOfWeek"]]   # df (avec NaN) : l'imputation est dans le pipeline
y = df["Target"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, stratify=y, random_state=RS)
log("Train :", X_tr.shape, "| Test :", X_te.shape, "| % pannes train/test :", round(y_tr.mean()*100, 1), round(y_te.mean()*100, 1))

def make_prep(scale=False):
    num = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("num", Pipeline(num), features_num + ["Month", "DayOfWeek"]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))]), features_cat),
    ])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
cv_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=RS)

def wf(clf, scale=False):   # workflow = prétraitement + modèle
    return Pipeline([("prep", make_prep(scale)), ("clf", clf)])

if HAS_XGB:
    xgb_est = XGBClassifier(eval_metric="logloss", random_state=RS, n_jobs=1)
    xgb_grid = {"clf__n_estimators": [150, 300, 500], "clf__max_depth": [3, 4, 6],
                "clf__learning_rate": [.03, .07, .15], "clf__subsample": [.7, .9, 1.0],
                "clf__colsample_bytree": [.6, .8, 1.0]}
    XGB_NAME = "XGBoost"
else:
    xgb_est = HistGradientBoostingClassifier(random_state=RS)
    xgb_grid = {"clf__max_iter": [150, 300, 500], "clf__max_depth": [3, 4, None],
                "clf__learning_rate": [.03, .07, .15], "clf__l2_regularization": [0, .1, 1.0],
                "clf__max_leaf_nodes": [15, 31, 63]}
    XGB_NAME = "HistGradientBoosting (substitut XGBoost)"
log("Modèle de boosting 'XGBoost' :", XGB_NAME)

search_space = {
    "Random Forest": (wf(RandomForestClassifier(random_state=RS, n_jobs=1)),
        {"clf__n_estimators": [200, 400, 600], "clf__max_depth": [None, 8, 12, 20],
         "clf__min_samples_leaf": [1, 2, 4], "clf__max_features": ["sqrt", "log2", .5]}),
    "Gradient Boosting": (wf(GradientBoostingClassifier(random_state=RS)),
        {"clf__n_estimators": [100, 200, 300], "clf__learning_rate": [.03, .07, .15],
         "clf__max_depth": [2, 3, 4], "clf__subsample": [.7, .9, 1.0]}),
    XGB_NAME: (wf(xgb_est), xgb_grid),
    "AdaBoost": (wf(AdaBoostClassifier(random_state=RS)),
        {"clf__n_estimators": [100, 200, 400], "clf__learning_rate": [.03, .1, .3, 1.0]}),
}

# %% 7. Tuning des hyperparamètres (RandomizedSearchCV) ------------------------
best = {}; tune_rows = []
for name, (pipe, grid_) in search_space.items():
    t0 = time.time()
    rs = RandomizedSearchCV(pipe, grid_, n_iter=10, scoring="roc_auc", cv=cv_tune,
                            random_state=RS, n_jobs=1, refit=True)
    rs.fit(X_tr, y_tr)
    best[name] = rs.best_estimator_
    tune_rows.append({"Modèle": name, "AUC CV (tuning)": round(rs.best_score_, 4),
                      "Meilleurs hyperparamètres": {k.replace("clf__", ""): v for k, v in rs.best_params_.items()},
                      "Durée (s)": round(time.time()-t0)})
    log(f"[{name}] AUC CV={rs.best_score_:.4f} | {rs.best_params_} | {time.time()-t0:.0f}s")
tune_df = pd.DataFrame(tune_rows)

# %% 8. Voting & Stacking construits à partir des modèles tunés ---------------
def tuned_clf(name):   # on réutilise le classifieur tuné, avec son propre prétraitement
    return best[name]
estimators = [(n.split(" ")[0].lower() + str(i), tuned_clf(n)) for i, n in enumerate(best)]
voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
stacking = StackingClassifier(estimators=estimators,
                              final_estimator=LogisticRegression(max_iter=1000),
                              cv=5, stack_method="predict_proba", n_jobs=1)
log("Entraînement Voting & Stacking ...")
voting.fit(X_tr, y_tr); stacking.fit(X_tr, y_tr)
best["Voting (soft)"] = voting; best["Stacking (LR)"] = stacking

# %% 9. Évaluation sur le test -------------------------------------------------
log("\n=== 9. ÉVALUATION (jeu de test) ===")
res = []; probas = {}; preds = {}
for name, model in best.items():
    p = model.predict_proba(X_te)[:, 1]; yhat = (p >= .5).astype(int)
    probas[name], preds[name] = p, yhat
    res.append({"Modèle": name, "Accuracy": accuracy_score(y_te, yhat),
                "Precision": precision_score(y_te, yhat), "Recall": recall_score(y_te, yhat),
                "F1": f1_score(y_te, yhat), "AUC": roc_auc_score(y_te, p)})
    log(f"\n--- {name} ---")
    log(classification_report(y_te, yhat, target_names=["No_Failure", "Failure"], digits=3))
res_df = pd.DataFrame(res).set_index("Modèle").round(4).sort_values("AUC", ascending=False)
log("\nTABLEAU COMPARATIF :\n", res_df.to_string())

# Courbes ROC
plt.figure(figsize=(8, 6.5))
for name in res_df.index:
    fpr, tpr, _ = roc_curve(y_te, probas[name])
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={res_df.loc[name,'AUC']:.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("Taux de faux positifs"); plt.ylabel("Taux de vrais positifs (rappel)")
plt.title("Courbes ROC – jeu de test"); plt.legend(loc="lower right", fontsize=9)
save("12_roc_curves.png")
plt.figure(figsize=(8, 6.5))
for name in res_df.index:
    fpr, tpr, _ = roc_curve(y_te, probas[name]); plt.plot(fpr, tpr, lw=2, label=name)
plt.xlim(0, .15); plt.ylim(.85, 1.003); plt.plot([0, 1], [0, 1], "k--"); plt.legend(loc="lower right", fontsize=9)
plt.title("Courbes ROC – zoom"); plt.xlabel("FPR"); plt.ylabel("TPR"); save("13_roc_zoom.png")

# Matrices de confusion
fig, axes = plt.subplots(2, 3, figsize=(14, 8)); axes = axes.ravel()
for ax, name in zip(axes, res_df.index):
    sns.heatmap(confusion_matrix(y_te, preds[name]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No_F", "Fail"], yticklabels=["No_F", "Fail"])
    ax.set_title(name, fontsize=10); ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
save("14_confusion_matrices.png")

# Importance des variables (meilleur modèle simple basé sur arbres)
tree_names = [n for n in best if n in ("Random Forest", "Gradient Boosting", XGB_NAME)]
ref = max(tree_names, key=lambda n: res_df.loc[n, "AUC"])
mdl = best[ref]
feat_names = list(features_num + ["Month", "DayOfWeek"]) + list(
    mdl.named_steps["prep"].named_transformers_["cat"].named_steps["oh"].get_feature_names_out(features_cat))
if hasattr(mdl.named_steps["clf"], "feature_importances_"):
    imp = pd.Series(mdl.named_steps["clf"].feature_importances_, index=feat_names).sort_values()
else:
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(mdl, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=RS)
    imp = pd.Series(pi.importances_mean, index=list(X_te.columns)).sort_values()
plt.figure(figsize=(8, 6)); imp.plot.barh(color="#457b9d"); plt.title(f"Importance des variables – {ref}")
save("15_feature_importance.png")
log(f"\nImportance des variables ({ref}) :\n", imp.sort_values(ascending=False).round(4).head(8).to_string())

# %% 10. Analyse économique ----------------------------------------------------
log("\n=== 10. ANALYSE ÉCONOMIQUE ===")
# HYPOTHÈSES (à remplacer par les coûts réels de l'entreprise) :
C_FN = 15_000   # € : panne NON détectée -> arrêt non planifié (perte de production + réparation d'urgence)
C_TP = 3_000    # € : panne détectée -> maintenance préventive planifiée (arrêt court, pièces)
C_FP = 800      # € : fausse alerte -> inspection inutile (main-d'œuvre + arrêt court)
C_TN = 0
def cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn*C_FN + tp*C_TP + fp*C_FP + tn*C_TN, (tn, fp, fn, tp)
n_fail = int(y_te.sum())
base_cost = n_fail * C_FN   # sans modèle : toutes les pannes sont subies
log(f"Hypothèses : C_FN={C_FN}€, C_TP={C_TP}€, C_FP={C_FP}€ | pannes dans le test = {n_fail}")
log(f"Coût sans modèle (maintenance corrective) : {base_cost:,.0f} €")
eco = []
for name in res_df.index:
    c, (tn, fp, fn, tp) = cost(y_te, preds[name])
    # seuil optimal (min coût) sur le test, à titre indicatif
    ths = np.linspace(.02, .98, 97)
    cs = [cost(y_te, (probas[name] >= t).astype(int))[0] for t in ths]
    t_opt = ths[int(np.argmin(cs))]; c_opt = min(cs)
    eco.append({"Modèle": name, "FN": fn, "FP": fp, "Coût (seuil 0.5) €": c, "Économie vs sans modèle €": base_cost - c,
                "Économie %": 100*(base_cost - c)/base_cost, "Seuil optimal": round(t_opt, 2),
                "Coût (seuil opt.) €": c_opt, "Économie (seuil opt.) €": base_cost - c_opt})
eco_df = pd.DataFrame(eco).set_index("Modèle")
log(eco_df.round(1).to_string())

bm = eco_df["Économie vs sans modèle €"].idxmax()
# graphique coût
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
eco_df["Coût (seuil 0.5) €"].plot.barh(ax=ax[0], color="#457b9d")
ax[0].axvline(base_cost, color="red", ls="--", label=f"Sans modèle : {base_cost:,.0f} €"); ax[0].legend()
ax[0].set_title("Coût total sur le jeu de test"); ax[0].set_xlabel("€")
ths = np.linspace(.02, .98, 97)
for name in [bm]:
    ax[1].plot(ths, [cost(y_te, (probas[name] >= t).astype(int))[0] for t in ths], lw=2, label=name)
ax[1].axhline(base_cost, color="red", ls="--", label="Sans modèle")
ax[1].set_xlabel("Seuil de décision"); ax[1].set_ylabel("Coût (€)"); ax[1].set_title(f"Coût selon le seuil – {bm}"); ax[1].legend()
save("16_economic.png")

# Sensibilité au ratio coût FN / coût FP (modèle retenu)
sens = []
for cfn in [5_000, 10_000, 15_000, 30_000, 60_000]:
    for cfp in [400, 800, 1_600]:
        tn, fp, fn, tp = confusion_matrix(y_te, preds[bm]).ravel()
        c_mod = fn*cfn + tp*C_TP + fp*cfp
        sens.append((cfn, cfp, 100*(n_fail*cfn - c_mod)/(n_fail*cfn)))
sens_df = pd.DataFrame(sens, columns=["C_FN", "C_FP", "eco"]).pivot(index="C_FN", columns="C_FP", values="eco").round(1)
log(f"\nSensibilité de l'économie (%) – {bm} :\n", sens_df.to_string())

# Extrapolation annuelle
days = (df.Date.max() - df.Date.min()).days + 1
n_obs_year = len(df) / days * 365
scale = n_obs_year / len(y_te)
log(f"\nPériode des données : {days} jours, {len(df)} relevés -> ~{n_obs_year:,.0f} relevés/an")
log(f"Économie annuelle extrapolée ({bm}, seuil 0.5) : {eco_df.loc[bm,'Économie vs sans modèle €']*scale:,.0f} €")
log(f"Économie annuelle extrapolée ({bm}, seuil optimal) : {eco_df.loc[bm,'Économie (seuil opt.) €']*scale:,.0f} €")

# %% 11. Export ----------------------------------------------------------------
res_df.to_csv(os.path.join(OUT_DIR, "tableau_comparatif.csv"))
eco_df.round(1).to_csv(os.path.join(OUT_DIR, "analyse_economique.csv"))
tune_df.to_csv(os.path.join(OUT_DIR, "hyperparametres.csv"), index=False)
d.drop(columns=["Pressure_Band", "Cutting_Band"]).to_csv(os.path.join(OUT_DIR, "Machine_Downtime_clean.csv"), index=False)
log("\nFichiers exportés dans", OUT_DIR)





