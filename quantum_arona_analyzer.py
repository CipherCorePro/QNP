# quantum_arona_analyzer.py

import json
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import os
from datetime import datetime


def lade_checkpoint(pfad: str) -> dict:
    with open(pfad, "r", encoding="utf-8") as f:
        return json.load(f)


def top_aktivierte_knoten(nodes: list[dict], n: int = 5) -> list[tuple[str, float]]:
    """Extrahiert die n Knoten mit dem höchsten Aktivierungslevel."""
    gültige = [
        (node["label"], node["activation"])
        for node in nodes
        if "label" in node and isinstance(node.get("activation"), (int, float))
    ]
    if not gültige:
        print("⚠️ Keine gültigen Knoten mit Aktivierungswerten gefunden.")
    return sorted(gültige, key=lambda x: x[1], reverse=True)[:n]


def top_hebb_verbindungen(conns: list[dict], n: int = 5) -> list[tuple[str, str, float]]:
    """Gibt die n Verbindungen mit dem höchsten Gewicht zurück."""
    return sorted(
        (
            (conn["source"], conn["target"], conn["weight"])
            for conn in conns
            if "weight" in conn
        ),
        key=lambda x: x[2],
        reverse=True
    )[:n]


def speichere_plot(fig, dateiname: str):
    os.makedirs("analyzer", exist_ok=True)
    pfad = os.path.join("analyzer", dateiname)
    fig.savefig(pfad)
    plt.close(fig)


def plot_top_knoten(knoten_liste: list[tuple[str, float]]):
    if not knoten_liste:
        print("⚠️ Kein Plot möglich – keine gültigen Aktivierungen.")
        return
    ids, werte = zip(*knoten_liste)
    fig = plt.figure(figsize=(8, 4))
    plt.barh(ids, werte)
    plt.xlabel("Aktivierungslevel")
    plt.title("Top aktivierte Knoten")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    speichere_plot(fig, "top_knoten.png")


def plot_top_verbindungen(verbindungen: list[tuple[str, str, float]]):
    if not verbindungen:
        print("⚠️ Kein Plot möglich – keine gültigen Gewichte.")
        return
    beschriftung = [f"{s}→{t}" for s, t, _ in verbindungen]
    werte = [gewicht for _, _, gewicht in verbindungen]
    fig = plt.figure(figsize=(8, 4))
    plt.barh(beschriftung, werte)
    plt.xlabel("Gewicht")
    plt.title("Top Hebb-Verbindungen")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    speichere_plot(fig, "top_verbindungen.png")


def visualisiere_netzwerk(nodes: list[dict], conns: list[dict]):
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node["label"])
    for conn in conns:
        if "source" in conn and "target" in conn:
            G.add_edge(conn["source"], conn["target"], weight=conn.get("weight", 0.1))
    pos = nx.spring_layout(G, seed=42)
    fig = plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold")
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights)
    plt.title("Netzwerkstruktur")
    speichere_plot(fig, "netzwerkstruktur.png")


def plot_emotion_state(daten: dict):
    emotion = daten.get("emotion_state", {})
    if not emotion:
        print("⚠️ Keine Emotionen vorhanden.")
        return
    labels = list(emotion.keys())
    werte = list(emotion.values())
    fig = plt.figure(figsize=(6, 4))
    plt.bar(labels, werte, color="salmon")
    plt.ylim(0, 1)
    plt.title("Emotion State (PAD-Modell)")
    plt.ylabel("Wert")
    plt.tight_layout()
    speichere_plot(fig, "emotion_state.png")


def epochen_verlauf_anzeigen(verzeichnis: str):
    verlauf = []
    for datei in os.listdir(verzeichnis):
        if datei.startswith("quantum_arona_checkpoint_epoch_") and datei.endswith(".json"):
            pfad = os.path.join(verzeichnis, datei)
            daten = lade_checkpoint(pfad)
            epoch = int(datei.split("_epoch_")[1].split(".")[0])
            loss = daten.get("trainer_state", {}).get("last_epoch_avg_loss")
            if loss is not None:
                verlauf.append((epoch, loss))

    if verlauf:
        verlauf.sort()
        epochen, verluste = zip(*verlauf)
        fig = plt.figure(figsize=(6, 4))
        plt.plot(epochen, verluste, marker="o")
        plt.title("Verlustverlauf über Epochen")
        plt.xlabel("Epoche")
        plt.ylabel("Durchschnittlicher Verlust")
        plt.grid(True)
        plt.tight_layout()
        speichere_plot(fig, "epochen_verlauf.png")
    else:
        print("⚠️ Keine Verlaufsdaten gefunden.")


def analysiere_checkpoint(pfad: str):
    daten = lade_checkpoint(pfad)
    nodes = daten.get("nodes", [])
    conns = daten.get("connections", [])

    log_pfadeintrag = os.path.join("analyzer", f"analyse_log.txt")
    os.makedirs("analyzer", exist_ok=True)

    with open(log_pfadeintrag, "a", encoding="utf-8") as log:
        log.write(f"\n===== Analyse vom {datetime.now().isoformat(timespec='seconds')} =====\n")
        log.write(f"Checkpoint: {os.path.basename(pfad)}\n")

        print("\n🔎 Top 5 aktivierte Knoten:")
        log.write("\n🔎 Top 5 aktivierte Knoten:\n")
        top_knoten = top_aktivierte_knoten(nodes)
        for k_id, wert in top_knoten:
            print(f"  {k_id}: {wert:.4f}")
            log.write(f"  {k_id}: {wert:.4f}\n")

        print("\n🔗 Top 5 Hebb-Verbindungen:")
        log.write("\n🔗 Top 5 Hebb-Verbindungen:\n")
        top_conns = top_hebb_verbindungen(conns)
        for src, tgt, gewicht in top_conns:
            print(f"  {src} -> {tgt}: {gewicht:.4f}")
            log.write(f"  {src} -> {tgt}: {gewicht:.4f}\n")

        trainer = daten.get("trainer_state", {})
        if trainer:
            print("\n🧠 Trainer-State:")
            log.write("\n🧠 Trainer-State:\n")
            for key, val in trainer.items():
                print(f"  {key}: {val}")
                log.write(f"  {key}: {val}\n")

    plot_top_knoten(top_knoten)
    plot_top_verbindungen(top_conns)
    visualisiere_netzwerk(nodes, conns)
    plot_emotion_state(daten)

    ordner = os.path.dirname(pfad)
    epochen_verlauf_anzeigen(ordner)


def cli():
    parser = argparse.ArgumentParser(description="Quantum Arona Analyzer")
    parser.add_argument("--checkpoint", type=str, help="Pfad zur Checkpoint-Datei")
    parser.add_argument("--all_epochs", type=str, help="Ordner mit mehreren Checkpoints")
    args = parser.parse_args()

    if args.all_epochs:
        for i in range(1, 11):
            dateiname = f"quantum_arona_checkpoint_epoch_{i}.json"
            pfad = os.path.join(args.all_epochs, dateiname)
            if os.path.exists(pfad):
                analysiere_checkpoint(pfad)
            else:
                print(f"⚠️ Datei nicht gefunden: {pfad}")
    elif args.checkpoint:
        analysiere_checkpoint(args.checkpoint)
    else:
        print("⚠️ Bitte gib --checkpoint oder --all_epochs an.")


if __name__ == "__main__":
    cli()
