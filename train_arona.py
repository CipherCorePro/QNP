# -*- coding: utf-8 -*-
# Filename: train_arona.py
# Description: Hauptskript zum Trainieren des Quantum-Arona Q-LLM.
# Aufrufbeispiel: python train_arona.py --config config_arona.json --data ./data/ethik_spezial.txt

import argparse
import os
import time
import traceback
import json
import glob
import re
from typing import Optional, List, Dict, Tuple, Any # Import für Type Hints

# Stelle sicher, dass der Importpfad korrekt ist
try:
    from quantum_arona_core import (
        load_config,
        QuantumAronaModel,
        DatasetLoader,
        Contextualizer,
        StateEmbedder,
        QuantumTrainer,
        PersistenceManager # Importiere PersistenceManager
    )
except ImportError as e:
    print(f"FEHLER: Konnte Kernkomponenten nicht importieren: {e}")
    print("Stelle sicher, dass 'quantum_arona_core.py' im selben Verzeichnis oder im Python-Pfad liegt.")
    exit(1) # Beende das Skript bei Importfehlern


def find_latest_checkpoint(checkpoint_dir: str, pattern: str = "quantum_arona_checkpoint_epoch_*.json") -> Optional[str]:
    """Sucht nach dem neuesten Checkpoint-File basierend auf der Epochennummer im Namen."""
    if not os.path.isdir(checkpoint_dir):
        print(f"Info: Checkpoint-Verzeichnis '{checkpoint_dir}' nicht gefunden.")
        return None

    search_pattern = os.path.join(checkpoint_dir, pattern)
    checkpoint_files = glob.glob(search_pattern)

    if not checkpoint_files:
        # print(f"Info: Keine Checkpoint-Dateien mit Muster '{pattern}' in '{checkpoint_dir}' gefunden.")
        return None

    latest_epoch = -1
    latest_file = None
    epoch_pattern = re.compile(r"_epoch_(\d+)\.json$")

    for file in checkpoint_files:
        match = epoch_pattern.search(os.path.basename(file)) # Suche nur im Dateinamen
        if match:
            try:
                epoch = int(match.group(1))
                # Aktualisiere, wenn eine höhere Epoche gefunden wird
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_file = file
                # Optional: Bei gleicher Epoche, nimm die alphabetisch letzte (könnte neuer sein)
                elif epoch == latest_epoch and latest_file is not None and file > latest_file:
                     latest_file = file
            except (ValueError, IndexError):
                print(f"Warnung: Konnte Epochennummer nicht aus '{os.path.basename(file)}' extrahieren.")
                continue

    # Fallback, falls kein Checkpoint mit "_epoch_" gefunden wurde, aber Dateien vorhanden sind
    if latest_file is None and checkpoint_files:
         print("Warnung: Keine Epochennummer in Dateinamen gefunden. Sortiere nach Änderungsdatum.")
         try:
             # Sortiere nach letztem Änderungsdatum, neueste zuerst
             checkpoint_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
             latest_file = checkpoint_files[0]
             print(f"Fallback: Wähle neuesten Checkpoint nach Datum: {os.path.basename(latest_file)}")
         except OSError as e:
             print(f"Fehler beim Zugriff auf Datei-Metadaten für Sortierung: {e}")
             # Letzter Fallback: nehme den ersten gefundenen
             try:
                 latest_file = checkpoint_files[0]
                 print(f"Letzter Fallback: Wähle ersten gefundenen Checkpoint: {os.path.basename(latest_file)}")
             except IndexError: # Falls checkpoint_files doch leer wurde
                 latest_file = None


    return latest_file

def main():
    parser = argparse.ArgumentParser(description="Trainiert das Quantum-Arona Q-LLM.")
    parser.add_argument("-c", "--config", type=str, default="config_arona.json", help="Pfad zur Konfigurationsdatei.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional: Pfad zu einem Checkpoint zum expliziten Laden.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional: Überschreibt Epochen aus Config.")
    # KORRIGIERT: nargs='+' erlaubt mehrere Dateien, hier nur eine erwartet, daher entfernt.
    parser.add_argument("--data", type=str, default=None,
                        help="Optional: Pfad zu EINER alternativen Trainingsdatei (überschreibt 'dataset_files' in der Config).")
    args = parser.parse_args()

    # 1. Konfiguration laden
    print(f"Lade Konfiguration aus: {args.config}")
    config = load_config(args.config)

    # NEU: Datensatz überschreiben, falls --data Argument gegeben und gültig
    if args.data:
        data_path = args.data
        if os.path.exists(data_path) and os.path.isfile(data_path):
            config['dataset_files'] = [data_path] # Ersetze Liste in Config durch diesen einen Pfad
            print(f"Info: Verwende spezifischen Datensatz aus Kommandozeile: {data_path}")
        else:
            print(f"WARNUNG: Angegebene Datei unter '--data' wurde nicht gefunden oder ist keine Datei: {data_path}")
            print("Verwende Datensätze aus der Konfigurationsdatei.")
            # Hier nicht abbrechen, sondern Fallback auf Config

    # Überschreibe andere Config-Werte mit Argumenten, falls gegeben
    if args.epochs: config['training_epochs'] = args.epochs
    # ... weitere Überschreibungen ...

    # Stelle sicher, dass Checkpoint-Verzeichnis existiert
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints_arona")
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints werden in '{checkpoint_dir}' gespeichert/gesucht.")
    except OSError as e:
        print(f"FEHLER: Konnte Checkpoint-Verzeichnis '{checkpoint_dir}' nicht erstellen: {e}")
        return

    # 2. Komponenten initialisieren
    print("Initialisiere Komponenten...")
    try:
        model = QuantumAronaModel(config)
        # Stelle sicher, dass dataset_files eine Liste ist
        dataset_files_list = config.get("dataset_files", [])
        if not isinstance(dataset_files_list, list):
             print(f"WARNUNG: 'dataset_files' in Config ist keine Liste. Versuche Konvertierung.")
             dataset_files_list = [str(dataset_files_list)] if dataset_files_list else []

        loader = DatasetLoader(
            file_paths=dataset_files_list, # Korrigiert: Immer Liste übergeben
            chunk_size=config.get("chunk_size", 500),
            overlap=config.get("chunk_overlap", 50)
        )
        contextualizer = Contextualizer()
        embedder = StateEmbedder(embedding_dim=config.get("embedding_dim", 128))

        # PersistenceManager initialisieren
        db_path = config.get("log_db_path")
        persistence_manager = None
        if db_path:
            try:
                db_dir = os.path.dirname(db_path)
                if db_dir and not os.path.exists(db_dir): os.makedirs(db_dir, exist_ok=True)
                persistence_manager = PersistenceManager(db_path)
                print(f"Persistence Manager initialisiert (DB: {db_path})")
            except Exception as db_err:
                 print(f"WARNUNG: Konnte Persistence Manager nicht initialisieren: {db_err}")
                 persistence_manager = None
        else: print("Info: Kein DB-Pfad in Config, Persistence Manager nicht aktiv.")

        trainer = QuantumTrainer(model, loader, contextualizer, embedder, config, persistence_manager)
    except Exception as e:
        print(f"FATAL ERROR during initialization: {e}")
        traceback.print_exc()
        return

    # 3. Checkpoint laden (Explizit oder Automatisch)
    checkpoint_to_load = args.checkpoint
    if not checkpoint_to_load and config.get("auto_load_latest_checkpoint", False):
        print("Suche nach neuestem Checkpoint zum automatischen Laden...")
        latest_checkpoint_file = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint_file:
            print(f"Neuester Checkpoint gefunden: {os.path.basename(latest_checkpoint_file)}")
            checkpoint_to_load = latest_checkpoint_file
        else: print("Kein vorheriger Checkpoint im Verzeichnis gefunden. Starte neues Training.")

    if checkpoint_to_load:
        if os.path.exists(checkpoint_to_load):
            print(f"Versuche Checkpoint zu laden: {checkpoint_to_load}")
            load_successful = trainer.load_checkpoint(checkpoint_to_load)
            if not load_successful: print("FEHLER beim Laden des Checkpoints. Training wird mit Initialzustand gestartet.")
        else: print(f"FEHLER: Angegebene Checkpoint-Datei nicht gefunden: {checkpoint_to_load}. Training wird mit Initialzustand gestartet.")

    # 4. Training starten
    print("\n--- Starte Quantum-Arona Training ---")
    training_start_time = time.time()
    try:
        trainer.train()
        training_duration = time.time() - training_start_time
        print(f"--- Training abgeschlossen (Dauer: {training_duration:.2f}s) ---")
    except KeyboardInterrupt:
        training_duration = time.time() - training_start_time
        print(f"\n--- Training durch Benutzer abgebrochen (Dauer: {training_duration:.2f}s) ---")
    except Exception as e:
        training_duration = time.time() - training_start_time
        print(f"\nFATAL ERROR during training (Dauer bis Fehler: {training_duration:.2f}s): {e}")
        traceback.print_exc()
    finally:
        # 5. Finalen Zustand speichern (Immer speichern)
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            final_checkpoint_path = os.path.join(checkpoint_dir, f"quantum_arona_final_state_{timestamp}.json")
            print(f"Speichere finalen Modellzustand nach: {final_checkpoint_path}")
            if 'trainer' in locals() and trainer is not None:
                 trainer.save_checkpoint(final_checkpoint_path)
            else: print("Fehler: Trainer-Objekt nicht verfügbar zum Speichern.")
        except Exception as save_err:
             print(f"FEHLER beim Speichern des finalen Checkpoints: {save_err}")
        finally:
             if persistence_manager: print("Schließe Persistence Manager DB..."); persistence_manager.close()
             print("Quantum-Arona Trainingsskript beendet.")

if __name__ == "__main__":
    main()