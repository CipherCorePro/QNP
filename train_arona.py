# -*- coding: utf-8 -*-
# Filename: train_arona.py
# Description: Hauptskript zum Trainieren des Quantum-Arona Q-LLM.

import argparse
import os
from quantum_arona_core import (
    load_config,
    QuantumAronaModel,
    DatasetLoader,
    Contextualizer,
    StateEmbedder,
    QuantumTrainer,
    # Eventuell PersistenceManager importieren
)

def main():
    parser = argparse.ArgumentParser(description="Trainiert das Quantum-Arona Q-LLM.")
    parser.add_argument("-c", "--config", type=str, default="config_arona.json", help="Pfad zur Konfigurationsdatei.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional: Pfad zu einem Checkpoint zum Laden.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional: Überschreibt Epochen aus Config.")
    # Weitere Argumente zum Überschreiben von Config-Werten hinzufügen...
    args = parser.parse_args()

    # 1. Konfiguration laden
    print(f"Lade Konfiguration aus: {args.config}")
    config = load_config(args.config)
    if args.epochs: config['training_epochs'] = args.epochs # Überschreibe Epochen

    # Stelle sicher, dass Checkpoint-Verzeichnis existiert
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints_arona")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints werden in '{checkpoint_dir}' gespeichert.")

    # 2. Komponenten initialisieren
    print("Initialisiere Komponenten...")
    try:
        model = QuantumAronaModel(config)
        loader = DatasetLoader(
            file_paths=config.get("dataset_files", []),
            chunk_size=config.get("chunk_size", 500),
            overlap=config.get("chunk_overlap", 50)
        )
        contextualizer = Contextualizer() # Mit Default-Werten oder aus Config
        embedder = StateEmbedder(embedding_dim=config.get("embedding_dim", 128))
        # TODO: PersistenceManager initialisieren
        trainer = QuantumTrainer(model, loader, contextualizer, embedder, config)
    except Exception as e:
        print(f"FATAL ERROR during initialization: {e}")
        traceback.print_exc()
        return

    # 3. Optional: Checkpoint laden
    if args.checkpoint:
        print(f"Versuche Checkpoint zu laden: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    elif config.get("auto_load_latest_checkpoint", False):
        # TODO: Logik zum Finden und Laden des neuesten Checkpoints implementieren
        pass

    # 4. Training starten
    print("\n--- Starte Quantum-Arona Training ---")
    try:
        trainer.train()
        print("--- Training abgeschlossen ---")
    except KeyboardInterrupt:
        print("\n--- Training durch Benutzer abgebrochen ---")
    except Exception as e:
        print(f"\nFATAL ERROR during training: {e}")
        traceback.print_exc()
    finally:
        # 5. Finalen Zustand speichern (optional oder immer?)
        final_checkpoint_path = os.path.join(checkpoint_dir, f"quantum_arona_final_state_{time.strftime('%Y%m%d_%H%M%S')}.json")
        print(f"Speichere finalen Modellzustand nach: {final_checkpoint_path}")
        trainer.save_checkpoint(final_checkpoint_path)
        print("Quantum-Arona Trainingsskript beendet.")

if __name__ == "__main__":
    main()
