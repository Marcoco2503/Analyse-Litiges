Gestionnaire de Litiges Intelligent — Concentrix Payment Services

Plateforme d'intelligence artificielle souveraine pour l'analyse, l'anonymisation RGPD, la synthèse et l'évaluation de sévérité des réclamations financières sur architecture CPU et base de données relationnelle optimisée.

Certification visée : Chef de projet Data et Intelligence Artificielle
Auteur : Marco CABROL  
Entreprise d'accueil : Concentrix Payment Services  
Date : Octobre 2025  

---

1.Description du projet

Ce projet répond aux exigences opérationnelles et réglementaires de Concentrix Payment Services dans le traitement des réclamations financières des consommateurs (corpus CFPB). 

Fonctionnalités principales :
* Anonymisation RGPD native (Privacy by Design) : Masquage déterministe automatique par expressions régulières des informations personnellement identifiables (cartes bancaires à 16 chiffres, adresses e-mails, numéros de téléphone et identifiants de compte).
* Double évaluation NLP de la sévérité (BERT) : Comparaison entre un modèle pré-entraîné généraliste (`nlptown/bert-base-multilingual-uncased-sentiment`) et une règle métier spécialisée intégrant les facteurs aggravants de fraude bancaire.
* Résumé automatique abstrait (T5) : Génération de synthèses textuelles courtes ( 40 tokens) via le modèle Seq2Seq `t5-small`.
* Persistance relationnelle optimisée (SQLite) : Base de données structurée (`litiges.db`) avec index B-Tree garantissant des temps de requêtage < 1 ms et respect du droit à l'effacement.
* Benchmarking NLP et métriques réelles : Calcul en direct des latences CPU, scores ROUGE, Précision, Rappel et score $F_2$.
* Exposition souveraine sécurisée : Tunnel chiffré TLS 1.3 via `pyngrok` sans transfert de données sur des serveurs tiers.

---

2.Structure de l'archive et livrables

Projet de fin d'étude/
│
├── app.py                  # Application principale Streamlit (Interface, NLP & CRUD)
├── init_db.py              # Script d'initialisation de la BDD SQLite et indexation
├── csv_creation.py         # Pipeline de traitement batch, anonymisation et génération
├── tunnel.py               # Script d'exposition distante sécurisée via Ngrok
│
├── litiges.db              # Base de données relationnelle SQLite opérationnelle
├── export_dump.sql         # Script complet d'export dump DDL/DML (Livrable RNCP)
├── requirements.txt        # Dépendances logicielles et versions figées
├── README.md               # Guide d'installation et d'exploitation du projet
└── csv/
    │
    ├── complaints.csv      # CSV de base
    └── resultats_analyse_ia.csv # CSV sorti du scrypt csv-creation.py

---

3.Prérequis système

Système d'exploitation : Windows 10/11, macOS ou Linux (Ubuntu 20.04+)

Langage : Python 3.10 ou version supérieure

Matériel requis : Processeur standard (CPU x86_64, 4 cœurs recommandés), 4 Go de RAM minimum.

---

4.Procédure d'installation pas-à-pas

Étape 1 : Cloner ou extraire le projet

cd Projet de fin d'étude

Étape 2 : Créer et activer un environnement virtuel

Sous Windows :

python -m venv env
.\env\Scripts\activate

Sous Linux / macOS :

python3 -m venv env
source env/bin/activate

Étape 3 : Installer les dépendances logicielles

pip install --upgrade pip
pip install -r requirements.txt

Étape 4 : Initialiser la base de données (si nécessaire)

python init_db.py

---

5. Guide d'exécution et utilisation

Dans un premier terminal, lancez Streamlit :

streamlit run app.py

Dans un second terminal, lancez le tunnel :

python tunnel.py

Cela donne l'url publique : https://alyse-raptorial-repressively.ngrok-free.dev

6. Restauration de la base de données

# Suppression de l'ancienne base
rm litiges.db

# Restauration complète depuis le dump SQL
sqlite3 litiges.db < export_dump.sql
