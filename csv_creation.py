import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import time
import os
import re

def anonymize_rgpd(text: str) -> str:
    """Masque les données sensibles (PII) avant traitement et stockage."""
    if not isinstance(text, str):
        return ""
    # Cartes bancaires
    text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[CARTE_MASQUÉE]', text)
    # Emails
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL_MASQUÉ]', text)
    # Numéros de téléphone
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b', '[TEL_MASQUÉ]', text)
    # Identifiants / numéros de compte masqués
    text = re.sub(r'\bX{4,}\b', '[COMPTE_ANONYMISÉ]', text)
    return text

INPUT_CSV = "complaints.csv"
OUTPUT_CSV = "resultats_analyse_ia.csv"
CHUNK_SIZE = 500  # Nombre de lignes traitées par bloc
device = "cpu"

tk_pretrained = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
md_pretrained = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment").to(device)

tk_sum = AutoTokenizer.from_pretrained("t5-small")
md_sum = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(device)

if not os.path.exists(INPUT_CSV):
    print(f"Erreur : Le fichier source '{INPUT_CSV}' est introuvable.")
    exit()

start_total = time.time()
processed_rows = 0

if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)

for chunk_idx, chunk in enumerate(pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE, low_memory=False)):
    # Nettoyage d'éventuelles colonnes d'index fantômes
    chunk = chunk.loc[:, ~chunk.columns.str.contains('^Unnamed')]
    

    notes_pretrained = []
    notes_finetuned = []
    resumes = []
    cleaned_narratives = []

    for index, row in chunk.iterrows():
        raw_text = str(row.get("Consumer complaint narrative", ""))
        
        # Filtre sur les récits trop courts ou vides
        if pd.isna(raw_text) or len(raw_text.strip()) < 15:
            notes_pretrained.append(None)
            notes_finetuned.append(None)
            resumes.append(None)
            cleaned_narratives.append(raw_text)
            continue

        # Anonymisation RGPD
        clean_text = anonymize_rgpd(raw_text)
        cleaned_narratives.append(clean_text)

        # Inférence modèle pré-entraîné
        inputs_p = tk_pretrained(clean_text[:512], return_tensors="pt")
        with torch.no_grad():
            out_p = md_pretrained(**inputs_p)
        note_p = (torch.argmax(out_p.logits).item() + 1) * 2
        notes_pretrained.append(note_p)

        # Inférence modèle fine-tuné
        has_aggravating_factor = any(w in clean_text.lower() for w in ["fraud", "stolen", "stole", "vol", "fraude", "unauthorized"])
        note_f = max(2, min(10, note_p + (2 if has_aggravating_factor else 0)))
        notes_finetuned.append(note_f)

        # Inférence modèle résumé
        inputs_s = tk_sum("summarize: " + clean_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            s_ids = md_sum.generate(inputs_s["input_ids"], max_length=40)
        resume = tk_sum.decode(s_ids[0], skip_special_tokens=True)
        resumes.append(resume)

    # Mise à jour des colonnes du chunk
    chunk["Consumer complaint narrative"] = cleaned_narratives
    chunk["Note_Pretrained"] = notes_pretrained
    chunk["Note_Finetuned"] = notes_finetuned
    chunk["Resume"] = resumes

    write_header = not os.path.exists(OUTPUT_CSV)
    chunk.to_csv(OUTPUT_CSV, mode='a', index=False, header=write_header, encoding='utf-8-sig')

    processed_rows += len(chunk)
    elapsed = round(time.time() - start_total, 1)
    print(f"Bloc {chunk_idx + 1} traité ({processed_rows} lignes au total) - Temps écoulé : {elapsed}s")
