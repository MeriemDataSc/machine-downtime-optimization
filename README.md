# Optimization of Machine Downtime

Projet d'analyse exploratoire et de prédiction binaire des pannes machine à partir de relevés de capteurs. Le travail compare plusieurs classifieurs et explore des scénarios économiques associés aux alertes de panne.

> **Résultats exploratoires :** le run local obtient une AUC de 1,0000 pour Voting (soft), sur une séparation aléatoire de 75/25. Ce score exceptionnel doit être vérifié par une validation chronologique avant toute conclusion opérationnelle.

## Structure du dépôt

```text
.
├── data/
│   ├── raw/                  # CSV source local; exclu de Git
│   └── processed/            # CSV nettoyé; exclu de Git
├── notebooks/
│   └── 01_machine_downtime_analysis.ipynb
├── src/
│   └── analyse_downtime.py   # Script de référence
├── reports/
│   ├── latest_run/           # Résultats générés par le script corrigé
│   └── previous_run/         # Sorties fournies initialement
├── outputs/                  # Sorties de travail locales; exclues de Git
└── requirements.txt
```

## Résultats du dernier run

L'exécution porte sur 2 500 relevés. Au seuil 0,5, le meilleur résultat du tableau est Voting (soft) : accuracy 0,9984, F1 0,9984 et AUC 1,0000. Les figures et tableaux complets sont dans [`reports/latest_run/`](reports/latest_run/).

L'analyse économique utilise des hypothèses fixées dans le script (15 000 € par panne manquée, 3 000 € par maintenance préventive et 800 € par fausse alerte). Le seuil « optimal » du tableau a été choisi sur le jeu de test : son coût ne constitue donc pas une évaluation indépendante. Les coûts et extrapolations annuelles sont des scénarios, pas des économies démontrées.

## Données

Place le fichier source autorisé sous `data/raw/Machine_Downtime.csv`. Le CSV nettoyé est régénérable et reste dans `data/processed/`. Les CSV de données sont exclus du dépôt. Vérifie la provenance et les droits de redistribution avant toute publication.

## Installation

Python 3.10 ou plus récent est recommandé.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux : source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Exécution

Depuis la racine du dépôt :

```bash
python src/analyse_downtime.py
```

Le script lit par défaut `data/raw/Machine_Downtime.csv` et écrit les figures et tableaux régénérés dans `outputs/downtime_analysis/`. Les chemins peuvent être précisés :

```bash
python src/analyse_downtime.py --data "chemin/vers/Machine_Downtime.csv" --out-dir outputs/downtime_analysis
```

## Limites et prochaines étapes

- Comparer le découpage aléatoire à une validation chronologique et réserver les relevés les plus récents au test final.
- Choisir le seuil sur un jeu de validation, pas sur le test final.
- Le notebook conserve l'analyse d'origine; le script est la version à privilégier et les deux doivent être alignés.
- Le projet prédit `Failure` / `No_Failure`; il ne prédit pas la durée d'arrêt ni un calendrier de maintenance optimal.
- Aucun fichier de licence n'est fourni, conformément à la demande.
