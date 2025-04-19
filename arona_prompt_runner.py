# -*- coding: utf-8 -*-
# Filename: infer_arona.py
# Description: Skript zur Interaktion mit einem trainierten Quantum-Arona-Modell.

import os
import json
import glob
import traceback
from datetime import datetime
from typing import Optional, Dict, Any

# === Quantum Arona Kernmodul einbinden ===
try:
    # Importiere die notwendigen Klassen und Funktionen
    from quantum_arona_core import (
        QuantumAronaModel,
        load_config,
        generate_prompt_response,
        # Importiere Node und Module nur, wenn sie für load_state benötigt werden
        # (normalerweise nicht direkt, aber sicherheitshalber)
        Node, LimbusAffektus, MetaCognitio, CortexCriticus, CortexCreativus,
        SimulatrixNeuralis, CortexSocialis, MemoryNode, ValueNode
    )
    CORE_LOADED = True
except ImportError as e:
    print(f"FEHLER: Konnte Kernkomponenten aus 'quantum_arona_core.py' nicht importieren: {e}")
    print("Stelle sicher, dass die Datei im selben Verzeichnis oder im Python-Pfad liegt.")
    CORE_LOADED = False
except NameError as ne:
    # Fängt Fehler ab, wenn Klassen (z.B. Node) nicht definiert sind,
    # was bei unvollständigem Einfügen in quantum_arona_core.py passieren kann.
     print(f"FEHLER: Eine benötigte Klasse oder Funktion wurde in quantum_arona_core.py nicht gefunden: {ne}")
     print("Bitte stelle sicher, dass alle Klassen (Node, Module etc.) in der Core-Datei vorhanden sind.")
     CORE_LOADED = False


# === Funktion zum Finden der neuesten finalen Modelldatei ===
def find_latest_model_file(checkpoint_dir: str) -> Optional[str]:
    """Sucht nach der neuesten 'final_model'-Datei basierend auf dem Timestamp im Namen."""
    if not os.path.isdir(checkpoint_dir):
        print(f"Fehler: Checkpoint-Verzeichnis '{checkpoint_dir}' nicht gefunden.")
        return None

    search_pattern = os.path.join(checkpoint_dir, "quantum_arona_final_model_*.json")
    model_files = glob.glob(search_pattern)

    if not model_files:
        print(f"Info: Keine finalen Modelldateien ('quantum_arona_final_model_*.json') in '{checkpoint_dir}' gefunden.")
        # Fallback: Suche nach neuestem Checkpoint als Alternative
        print("Versuche stattdessen, neuesten Checkpoint zu finden...")
        search_pattern = os.path.join(checkpoint_dir, "quantum_arona_checkpoint_epoch_*.json")
        model_files = glob.glob(search_pattern)
        if not model_files:
            print("Fehler: Auch keine Checkpoint-Dateien gefunden.")
            return None

    # Sortiere nach Zeitstempel im Namen oder Änderungsdatum
    def get_sort_key(filepath):
        basename = os.path.basename(filepath)
        # Versuche Timestamp aus final_model Namen zu extrahieren
        if "final_model_" in basename:
            try:
                timestamp_str = basename.split("_")[-1].split(".")[0]
                # Korrekte Zeitstempel-Formatierung annehmen (YYYYMMDD_HHMMSS)
                dt_obj = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                return dt_obj.timestamp() # Konvertiere zu Unix-Timestamp für einfachen Vergleich
            except (IndexError, ValueError):
                # Fallback auf Änderungsdatum, wenn Timestamp-Extraktion fehlschlägt
                print(f"Warnung: Konnte Timestamp nicht aus '{basename}' extrahieren, nutze Änderungsdatum.")
                return os.path.getmtime(filepath)
        # Versuche Epochennummer aus Checkpoint Namen zu extrahieren
        elif "_epoch_" in basename:
             try:
                 epoch = int(basename.split('_')[-1].split('.')[0])
                 # Verwende Epoche als primären Sortierschlüssel, dann Änderungsdatum
                 return (epoch, os.path.getmtime(filepath))
             except (ValueError, IndexError):
                 return (0, os.path.getmtime(filepath)) # Fallback
        else:
            # Fallback für unerwartete Dateinamen
            return os.path.getmtime(filepath)

    try:
        model_files.sort(key=get_sort_key, reverse=True)
        latest_file = model_files[0]
        print(f"🧬 Verwende folgende Zustandsdatei: {os.path.basename(latest_file)}")
        return latest_file
    except Exception as e:
        print(f"Fehler beim Sortieren der Modelldateien: {e}")
        return None

# === Hauptinteraktion ===
def main():
    if not CORE_LOADED:
        print("Kernmodul konnte nicht geladen werden. Skript wird beendet.")
        return

    config_path = "config_arona.json"
    config = load_config(config_path)
    if not config:
        print("FEHLER: Konnte Konfiguration nicht laden. Abbruch.")
        return

    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints_arona") # Hole aus geladener Config

    # Finde die zu ladende Modelldatei
    model_file_path = find_latest_model_file(checkpoint_dir)

    if not model_file_path:
        print("Keine geeignete Modelldatei gefunden. Bitte Modell trainieren und speichern.")
        return

    print("\n🔷 Quantum-Arona Prompt Runner – Inferenzmodus aktiviert")
    print(f"ℹ️ Modell: {os.path.basename(model_file_path)}")
    print("💬 Gib einen Prompt ein (oder 'quit' zum Beenden):\n")

    while True:
        prompt = input("📝 Prompt > ").strip()
        if prompt.lower() in {"exit", "quit"}:
            print("👋 Beende Arona Prompt Runner.")
            break
        if not prompt:
            continue

        print("⏳ Initialisiere Modell und generiere Antwort...\n")
        try:
            # --- WICHTIG: Für jeden Prompt Modell neu laden! ---
            # 1. Frisches Modell-Objekt erstellen
            inference_model = QuantumAronaModel(config)
            # 2. Zustand aus Datei laden
            with open(model_file_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            inference_model.load_state(state_data)
            print("   (Modellzustand für diesen Prompt geladen)")

            # 3. Antwort generieren
            response = generate_prompt_response(
                prompt=prompt,
                loaded_model=inference_model, # Übergebe das Objekt
                inference_steps=config.get("inference_steps", 20), # Hole aus Config oder Default
                n_shots_inference=config.get("n_shots_inference", 50) # Hole aus Config oder Default
            )
            print(f"\n🤖 Arona sagt:\n{response}\n")

        except Exception as e:
            print(f"❌ Fehler bei der Antwortgenerierung: {e}")
            traceback.print_exc() # Zeige Details für Debugging

if __name__ == "__main__":
    main()