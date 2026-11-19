import sqlite3
import pandas as pd
import os

DB_FILE = "litiges.db"
CSV_SOURCE = "resultats_analyse_ia.csv"

# Connexion à la base SQLite
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Création de la table principale
cursor.execute("""
CREATE TABLE IF NOT EXISTS reclamations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date_received TEXT,
    product TEXT,
    sub_product TEXT,
    issue TEXT,
    consumer_narrative TEXT,
    company TEXT,
    note_pretrained REAL,
    note_finetuned REAL,
    resume TEXT
);
""")

# Création d'index
cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON reclamations (date_received);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_company ON reclamations (company);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_notes ON reclamations (note_finetuned);")

conn.commit()

# Migration des données existantes du CSV vers SQLite
if os.path.exists(CSV_SOURCE):
    df = pd.read_csv(CSV_SOURCE, low_memory=False)
    # Renommage des colonnes pour correspondre au schéma SQL
    rename_mapping = {
        "Date received": "date_received",
        "Product": "product",
        "Sub-product": "sub_product",
        "Issue": "issue",
        "Consumer complaint narrative": "consumer_narrative",
        "Company": "company",
        "Note_Pretrained": "note_pretrained",
        "Note_Finetuned": "note_finetuned",
        "Resume": "resume"
    }
    df = df.rename(columns=rename_mapping)
    valid_cols = [c for c in rename_mapping.values() if c in df.columns]
    df[valid_cols].to_sql("reclamations", conn, if_exists="append", index=False)

# Génération automatique du fichier SQL d'export
with open("export_dump.sql", "w", encoding="utf-8") as f:
    for line in conn.iterdump():
        f.write(f"{line}\n")

conn.close()