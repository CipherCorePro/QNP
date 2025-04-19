# qllm_gemini_app.py

# -*- coding: utf-8 -*-
# Filename: arona_gemini_app.py
# Description: Streamlit App zur Interaktion mit Quantum-Arona,
#              verstärkt durch Gemini für die finale Textausgabe.

import streamlit as st
import os
import json
import glob
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# Sicherstellen, dass das Kernmodul geladen werden kann
try:
    from quantum_arona_core import (
        QuantumAronaModel,
        load_config,
        run_arona_inference,  # Die neue Inferenzfunktion
        # Benötigte Node/Modul-Klassen für das Laden des Modells
        Node, LimbusAffektus, MetaCognitio, CortexCriticus, CortexCreativus,
        SimulatrixNeuralis, CortexSocialis, MemoryNode, ValueNode
    )
    CORE_LOADED = True
except ImportError as e:
    st.error(f"FEHLER: Konnte Kernkomponenten aus 'quantum_arona_core.py' nicht importieren: {e}\nStelle sicher, dass die Datei im selben Verzeichnis liegt.")
    CORE_LOADED = False
except NameError as ne:
     st.error(f"FEHLER: Eine benötigte Klasse/Funktion wurde in quantum_arona_core.py nicht gefunden: {ne}\nBitte Core-Datei prüfen.")
     CORE_LOADED = False

# Gemini API Import
try:
    import google.generativeai as genai
    GEMINI_LOADED = True
except ImportError:
    st.warning("Google Generative AI SDK nicht gefunden (`pip install google-generativeai`). Gemini-Integration ist deaktiviert.")
    GEMINI_LOADED = False

# --- Hilfsfunktionen (aus infer_arona.py übernommen/angepasst) ---

def find_latest_model_file(checkpoint_dir: str) -> Optional[str]:
    """Sucht nach der neuesten 'final_model'-Datei oder dem neuesten Checkpoint."""
    if not os.path.isdir(checkpoint_dir):
        st.error(f"Fehler: Checkpoint-Verzeichnis '{checkpoint_dir}' nicht gefunden.")
        return None

    # Priorisiere finale Modelldateien
    search_pattern_final = os.path.join(checkpoint_dir, "quantum_arona_final_state_*.json")
    model_files = glob.glob(search_pattern_final)

    if not model_files:
        st.info(f"Keine finalen Modelldateien gefunden. Suche nach neuestem Checkpoint...")
        search_pattern_ckpt = os.path.join(checkpoint_dir, "quantum_arona_checkpoint_epoch_*.json")
        model_files = glob.glob(search_pattern_ckpt)
        if not model_files:
            st.error("Fehler: Weder finale Modelldateien noch Checkpoints gefunden.")
            return None

    def get_sort_key(filepath):
        basename = os.path.basename(filepath)
        if "final_model_" in basename:
            try:
                timestamp_str = basename.split("_")[-1].split(".")[0]
                dt_obj = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                return dt_obj.timestamp()
            except: return os.path.getmtime(filepath) # Fallback
        elif "_epoch_" in basename:
             try:
                 epoch = int(basename.split('_')[-1].split('.')[0])
                 return (epoch, os.path.getmtime(filepath)) # Sortiere primär nach Epoche
             except: return (0, os.path.getmtime(filepath))
        else: return os.path.getmtime(filepath)

    try:
        model_files.sort(key=get_sort_key, reverse=True)
        latest_file = model_files[0]
        st.success(f"🧬 Verwende Zustandsdatei: {os.path.basename(latest_file)}")
        return latest_file
    except Exception as e:
        st.error(f"Fehler beim Sortieren der Modelldateien: {e}")
        return None

# --- Caching für das Arona-Modell ---
# Verhindert das Neuladen bei jeder Interaktion
@st.cache_resource(show_spinner="Lade Quantum Arona Modell...")
def load_arona_model(config: Dict, model_file_path: str) -> Optional[QuantumAronaModel]:
    """Lädt das QuantumAronaModel-Objekt aus der Datei."""
    if not model_file_path or not os.path.exists(model_file_path):
        st.error(f"Modelldatei '{model_file_path}' nicht gefunden oder ungültig.")
        return None
    try:
        model = QuantumAronaModel(config)
        with open(model_file_path, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        model.load_state(state_data)
        print("Arona Modell im Cache geladen.") # Für Konsolen-Debugging
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des Arona-Modellzustands: {e}")
        traceback.print_exc()
        return None

# --- Gemini API Interaktion ---
def generate_gemini_response(api_key: str, user_prompt: str, arona_context: str) -> str:
    """Sendet den Prompt und Aronas Kontext an Gemini und gibt die Antwort zurück."""
    if not GEMINI_LOADED:
        return "[Gemini SDK nicht geladen]"
    if not api_key:
        return "[Bitte Gemini API Schlüssel eingeben]"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash') # Oder ein anderes Modell

        # Konstruiere einen detaillierten Prompt für Gemini
        gemini_prompt = f"""
        Du bist ein fortgeschrittener KI-Assistent, der die Ausgabe eines experimentellen, quanten-inspirierten KI-Modells namens Quantum-Arona interpretiert und darauf aufbaut.

        Der Benutzer hat folgenden ursprünglichen Prompt eingegeben:
        "{user_prompt}"

        Quantum-Arona hat diesen Prompt intern verarbeitet und liefert folgenden Kontext über seinen "Denkprozess":
        {arona_context}

        Deine Aufgabe ist es:
        1.  Antworte direkt und präzise auf den *ursprünglichen Prompt* des Benutzers.
        2.  Nutze den von Quantum-Arona gelieferten Kontext (Gedankenkette, Emotion, Fokus), um die *Perspektive, den Ton oder die thematische Gewichtung* deiner Antwort zu beeinflussen. Interpretiere Aronas Kontext kreativ.
        3.  Formuliere eine kohärente, gut lesbare und hilfreiche Antwort in natürlicher Sprache. Erkläre Aronas internen Prozess NICHT, sondern nutze ihn als Inspiration.

        Antworte jetzt auf den ursprünglichen Prompt unter Berücksichtigung des Arona-Kontexts:
        """

        response = model.generate_content(gemini_prompt)
        return response.text

    except Exception as e:
        st.error(f"Fehler bei der Gemini API-Anfrage: {e}")
        traceback.print_exc()
        return "[Fehler bei der Kommunikation mit Gemini]"

# --- Streamlit App Hauptteil ---
st.set_page_config(layout="wide")
st.title("🌌 Quantum Arona + Gemini Interface v1.1")
st.caption("Ein Experiment von CipherCore & Gemini")

if not CORE_LOADED:
    st.stop() # Beende App, wenn Kernmodul fehlt

# --- Konfiguration und Modell laden ---
config_path = "config_arona.json"
config = load_config(config_path)
checkpoint_dir = config.get("checkpoint_dir", "./checkpoints_arona")
model_file_path = find_latest_model_file(checkpoint_dir)

# Lade das Arona-Modell einmal pro Session mit Caching
arona_model_instance = None
if model_file_path:
    arona_model_instance = load_arona_model(config, model_file_path)
else:
    st.error("Keine Modelldatei zum Laden gefunden. Training erforderlich.")
    st.stop()

if not arona_model_instance:
     st.error("Fehler beim Laden des Arona Modells. App kann nicht fortfahren.")
     st.stop()


# --- API Key Eingabe (in der Sidebar) ---
st.sidebar.header("Konfiguration")
# WICHTIG: Für echte Anwendungen Streamlit Secrets verwenden!
api_key = st.sidebar.text_input("Google Gemini API Key:", type="password", help="Gib deinen API-Schlüssel für Gemini ein.")
st.sidebar.caption("Hinweis: Dein API-Schlüssel wird nur für diese Sitzung verwendet und nicht dauerhaft gespeichert (im Code oder in Streamlit Secrets empfohlen).")


# --- Benutzer-Interaktion ---
st.header("Prompt Eingabe")
user_prompt = st.text_area("Gib deinen Prompt für Quantum Arona ein:", height=100)

if st.button("🚀 Antwort generieren (Arona + Gemini)"):
    if not user_prompt:
        st.warning("Bitte gib einen Prompt ein.")
    elif not api_key and GEMINI_LOADED:
         st.warning("Bitte gib deinen Gemini API Schlüssel in der Sidebar ein.")
    else:
        # 1. Arona Inferenz starten
        with st.spinner("Quantum Arona denkt nach... (führt Inferenzschritte aus)"):
            try:
                # Erstelle *Kopie* des geladenen Modells für diese Inferenz
                # (optional, aber sicherer, um Zustandsänderungen zu isolieren)
                # Da load_arona_model gecached ist, erstellen wir hier KEINE Kopie,
                # sondern nutzen die gecachte Instanz. Wichtig: Zustand wird verändert!
                # Alternative: Bei jedem Klick neu laden (langsamer)
                # Alternative 2: Modell kopieren (kann Speicherintensiv sein)
                # Wir gehen hier das Risiko ein, dass sich der gecachte Zustand ändert.
                # Für echte Isolation: load_arona_model ohne Cache hier aufrufen.
                
                # Korrektur: Wir müssen das Modell für jeden Prompt zurücksetzen!
                # Daher laden wir den State neu in die gecachte Instanz.
                print("Reloading state into cached model for new prompt...")
                with open(model_file_path, 'r', encoding='utf-8') as f:
                     state_data = json.load(f)
                arona_model_instance.load_state(state_data) # Setzt Zustand zurück!
                
                arona_text, thought_chain, final_emotion = run_arona_inference(
                    prompt=user_prompt,
                    loaded_model=arona_model_instance,
                    inference_steps=config.get("inference_steps", 20),
                    n_shots_inference=config.get("n_shots_inference", 50)
                )
                st.success("Arona Inferenz abgeschlossen.")

                # Zeige Aronas interne Ergebnisse
                with st.expander("🧠 Aronas interner Prozess (Details)", expanded=False):
                    st.write("**Gedankenkette (Dominante Konzepte pro Schritt):**")
                    st.write(thought_chain)
                    st.write("**Finaler Emotionszustand (PAD):**")
                    st.json(final_emotion)
                    st.write("**Aronas Textbeschreibung des Prozesses:**")
                    st.write(arona_text)

                # 2. Gemini Antwort generieren (wenn SDK geladen und Key vorhanden)
                if GEMINI_LOADED and api_key:
                     with st.spinner("Gemini formuliert die finale Antwort..."):
                        # Bereite Kontext für Gemini vor
                        arona_context_for_gemini = f"""
                        Interner Fokus (Gedankenkette): {thought_chain}
                        Finaler emotionaler Zustand (Pleasure, Arousal, Dominance): {final_emotion}
                        Aronas eigene Zusammenfassung: {arona_text}
                        """
                        gemini_response = generate_gemini_response(api_key, user_prompt, arona_context_for_gemini)
                        st.success("Finale Antwort von Gemini generiert.")
                        st.header("💬 Finale Antwort (Gemini, inspiriert von Arona)")
                        st.markdown(gemini_response) # Markdown für bessere Formatierung
                elif not GEMINI_LOADED:
                     st.warning("Gemini SDK nicht geladen. Zeige nur Aronas Beschreibung.")
                     st.header("🤖 Aronas Beschreibung")
                     st.write(arona_text)
                else: # Gemini SDK geladen, aber kein Key
                     st.warning("Kein Gemini API Key eingegeben. Zeige nur Aronas Beschreibung.")
                     st.header("🤖 Aronas Beschreibung")
                     st.write(arona_text)


            except Exception as e:
                st.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
                traceback.print_exc()

# Optional: Zeige Modellpfad im Footer
st.sidebar.markdown("---")
if model_file_path:
    st.sidebar.caption(f"Geladenes Modell: `{os.path.basename(model_file_path)}`")