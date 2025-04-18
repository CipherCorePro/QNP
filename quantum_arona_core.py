# -*- coding: utf-8 -*-
# Filename: quantum_arona_core.py
# Description: Kernarchitektur für Quantum-Arona - Zustandsbasiertes Q-LLM v1.0
#              Integriert QNP-H v2.1 Logik und Q-LLM Trainingskomponenten.
# Author: [CipherCore Technology] & Gemini
# Status: Hoch experimenteller Prototyp - Vollständige Struktur

import numpy as np
import pandas as pd # Für potenzielle Datenstrukturen im Training
import random
from collections import deque, Counter
import json
import sqlite3
import os
import time
import traceback
from typing import Optional, Callable, List, Tuple, Dict, Any, Generator
import uuid
from datetime import datetime
import math

# Optional: Für Netzwerk-Visualisierung beim Debugging
try: import networkx as nx; NETWORKX_AVAILABLE = True
except ImportError: NETWORKX_AVAILABLE = False

# Optional: Für Fortschrittsbalken
try: from tqdm import tqdm; TQDM_AVAILABLE = True
except ImportError: def tqdm(iterable, *args, **kwargs): return iterable

# ########################################################################
# # 1. Quanten-Engine (Qubit Simulation, PQC, Messung)
# ########################################################################

# === Basis-Gates & Helfer (aus QNP-H) ===
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
P1 = np.array([[0, 0], [0, 1]], dtype=complex)

def _ry(theta: float) -> np.ndarray:
    cos_t = np.cos(theta / 2); sin_t = np.sin(theta / 2)
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=complex)

def _rz(phi: float) -> np.ndarray:
    exp_m = np.exp(-1j * phi / 2); exp_p = np.exp(1j * phi / 2)
    return np.array([[exp_m, 0], [0, exp_p]], dtype=complex)

def _apply_gate(state_vector: np.ndarray, gate: np.ndarray, target_qubit: int, num_qubits: int) -> np.ndarray:
    if gate.shape != (2, 2): raise ValueError("Gate must be 2x2.")
    if not (0 <= target_qubit < num_qubits): raise ValueError(f"Target qubit {target_qubit} out of range [0, {num_qubits-1}].")
    expected_len = 2**num_qubits
    current_len = len(state_vector)
    if current_len != expected_len: raise ValueError(f"State vector length {current_len} != expected {expected_len} for {num_qubits} qubits.")
    op_list = [I] * num_qubits
    op_list[target_qubit] = gate
    full_matrix = op_list[0]
    for i in range(1, num_qubits): full_matrix = np.kron(full_matrix, op_list[i])
    return np.dot(full_matrix, state_vector)

def _apply_cnot(state_vector: np.ndarray, control_qubit: int, target_qubit: int, num_qubits: int) -> np.ndarray:
    if not (0 <= control_qubit < num_qubits and 0 <= target_qubit < num_qubits): raise ValueError("Qubit index out of range.")
    if control_qubit == target_qubit: raise ValueError("Control and target must be different.")
    expected_len = 2**num_qubits
    current_len = len(state_vector)
    if current_len != expected_len: raise ValueError(f"State vector length {current_len} != expected {expected_len} for {num_qubits} qubits.")
    op_list_p0 = [I] * num_qubits; op_list_p1 = [I] * num_qubits
    op_list_p0[control_qubit] = P0; op_list_p1[control_qubit] = P1
    op_list_p1[target_qubit] = X
    term0_matrix = op_list_p0[0]; term1_matrix = op_list_p1[0]
    for i in range(1, num_qubits):
        term0_matrix = np.kron(term0_matrix, op_list_p0[i])
        term1_matrix = np.kron(term1_matrix, op_list_p1[i])
    cnot_matrix = term0_matrix + term1_matrix
    return np.dot(cnot_matrix, state_vector)

# === Quantum Node System (QNS) ===
class QuantumNodeSystem:
    """Simuliert das quantenbasierte Verhalten eines Knotens via PQC."""
    def __init__(self, num_qubits: int, initial_params: Optional[np.ndarray] = None):
        if num_qubits <= 0: raise ValueError("num_qubits must be positive.")
        self.num_qubits = num_qubits
        self.num_params = num_qubits * 2 # 2 Parameter (RY, RZ) pro Qubit
        self.state_vector_size = 2**self.num_qubits

        if initial_params is None:
            self.params = np.random.rand(self.num_params) * np.pi
        elif isinstance(initial_params, np.ndarray) and initial_params.shape == (self.num_params,):
            if not np.all(np.isfinite(initial_params)): raise ValueError("Initial params non-finite.")
            self.params = np.array(initial_params, dtype=float)
        else:
            raise ValueError(f"Shape mismatch for initial_params ({initial_params.shape if isinstance(initial_params, np.ndarray) else type(initial_params)} vs {(self.num_params,)})")

        self.state_vector = np.zeros(self.state_vector_size, dtype=complex)
        self.state_vector[0] = 1.0 + 0j
        self.last_measurement_results: List[Dict] = [] # Für Sprunganalyse

    def _build_pqc_ops(self, input_strength: float) -> List[Tuple]:
        """Definiert die PQC-Gate-Sequenz (H -> RY -> RZ -> CNOT Kette)."""
        ops = []
        scaled_input = np.tanh(input_strength) # Skaliert klassischen Input
        for i in range(self.num_qubits): ops.append(('H', i))
        for i in range(self.num_qubits):
            theta = scaled_input * self.params[2 * i]
            ops.append(('RY', i, theta if np.isfinite(theta) else 0.0))
        for i in range(self.num_qubits):
            phi = self.params[2 * i + 1]
            ops.append(('RZ', i, phi if np.isfinite(phi) else 0.0))
        if self.num_qubits > 1:
            for i in range(self.num_qubits): ops.append(('CNOT', i, (i + 1) % self.num_qubits))
        return ops

    def activate(self, input_strength: float, n_shots: int) -> Tuple[float, np.ndarray, List[Dict]]:
        """Führt PQC aus, misst und gibt norm. Hamming-Gewicht, finalen State-Vector und Mess-Log zurück."""
        if not np.isfinite(input_strength): input_strength = 0.0
        pqc_ops = self._build_pqc_ops(input_strength)
        total_hamming_weight = 0
        final_state_vector = np.zeros(self.state_vector_size, dtype=complex); final_state_vector[0]=1.0 # Init
        measurement_log = []

        for shot in range(n_shots):
            current_state = np.zeros(self.state_vector_size, dtype=complex); current_state[0] = 1.0
            gate_failed = False
            for op_index, op in enumerate(pqc_ops):
                try:
                    op_type = op[0]
                    if op_type == 'H': current_state = _apply_gate(current_state, H, op[1], self.num_qubits)
                    elif op_type == 'RY': current_state = _apply_gate(current_state, _ry(op[2]), op[1], self.num_qubits)
                    elif op_type == 'RZ': current_state = _apply_gate(current_state, _rz(op[2]), op[1], self.num_qubits)
                    elif op_type == 'CNOT': current_state = _apply_cnot(current_state, op[1], op[2], self.num_qubits)
                    if not np.all(np.isfinite(current_state)): raise ValueError("Non-finite state vector detected")
                except Exception as e:
                    print(f"WARNUNG: Gate-Fehler in QNS {op} Shot {shot+1}: {e}. Resetting state.")
                    #traceback.print_exc() # Kann sehr verbose sein
                    current_state.fill(0.0); current_state[0] = 1.0
                    gate_failed = True; break
            if gate_failed: continue
            final_state_vector = current_state # Letzter erfolgreicher State vor Messung

            probabilities = np.abs(current_state)**2
            probabilities = np.maximum(0, probabilities) # Numerische Stabilität
            prob_sum = np.sum(probabilities)
            if not np.isclose(prob_sum, 1.0, atol=1e-7):
                 if prob_sum < 1e-9: probabilities.fill(0.0); probabilities[0] = 1.0 # Fallback auf |0>
                 else: probabilities /= prob_sum # Normalisieren
            probabilities = np.maximum(0, probabilities) # Erneute Sicherstellung
            probabilities /= np.sum(probabilities) # Finale Normalisierung

            try:
                 measured_index = np.random.choice(self.state_vector_size, p=probabilities)
                 state_idx_int = int(measured_index)
                 binary_repr = format(state_idx_int, f'0{self.num_qubits}b')
                 hamming_weight = binary_repr.count('1')
                 total_hamming_weight += hamming_weight
                 measurement_log.append({"shot": shot, "index": state_idx_int, "binary": binary_repr, "hamming": hamming_weight})
            except Exception as e:
                 print(f"WARNUNG: Messungs-Fehler Shot {shot+1}: {e}. Using argmax fallback.")
                 measured_index = np.argmax(probabilities)
                 state_idx_int = int(measured_index); binary_repr = format(state_idx_int, f'0{self.num_qubits}b')
                 hamming_weight = binary_repr.count('1'); total_hamming_weight += hamming_weight
                 measurement_log.append({"shot": shot, "index": state_idx_int, "binary": binary_repr, "hamming": hamming_weight, "error": str(e)})

        activation_prob = 0.0
        if n_shots > 0 and self.num_qubits > 0:
            activation_prob = float(np.clip(total_hamming_weight / (n_shots * self.num_qubits), 0.0, 1.0))
        if not np.isfinite(activation_prob): activation_prob = 0.0

        self.last_measurement_results = measurement_log
        return activation_prob, final_state_vector, measurement_log

    def get_params(self) -> np.ndarray:
        """Gibt eine Kopie der aktuellen Quantenparameter zurück (sicher gegen NaN/Inf)."""
        safe_params = self.params.copy()
        safe_params = np.nan_to_num(safe_params, nan=np.pi, posinf=2*np.pi, neginf=0.0)
        return safe_params

    def set_params(self, params: np.ndarray):
        """Setzt die Quantenparameter (sicher gegen NaN/Inf und Clipping)."""
        if isinstance(params, np.ndarray) and params.shape == self.params.shape:
            safe_params = np.nan_to_num(params, nan=np.pi, posinf=2*np.pi, neginf=0.0)
            self.params = np.clip(safe_params, 0, 2 * np.pi)
        else:
            print(f"WARNING: Shape/Type mismatch QParam set ({params.shape if isinstance(params, np.ndarray) else type(params)} vs {self.params.shape}). Not set.")

    def update_internal_params(self, delta_params: np.ndarray):
        """Wendet Parameteränderungen an und clippt (zentral für Training)."""
        if not isinstance(delta_params, np.ndarray) or delta_params.shape != self.params.shape:
             print(f"WARNING: Invalid delta_params for QParam update ({delta_params.shape if isinstance(delta_params, np.ndarray) else type(delta_params)} vs {self.params.shape}). Skipped.")
             return
        if not np.all(np.isfinite(delta_params)):
             # Ersetze nicht-finite Deltas mit 0, um den Rest zu retten
             delta_params = np.nan_to_num(delta_params, nan=0.0, posinf=0.0, neginf=0.0)
             print(f"WARNING: Non-finite values in delta_params clamped to 0.")

        new_params = self.params + delta_params
        new_params_safe = np.nan_to_num(new_params, nan=np.pi, posinf=2*np.pi, neginf=0.0)
        self.params = np.clip(new_params_safe, 0, 2 * np.pi)

# ########################################################################
# # 2. Netzwerk-Struktur (Knoten, Verbindungen, Module)
# ########################################################################

class Connection:
    """Repräsentiert eine gerichtete, gewichtete Verbindung."""
    def __init__(self, target_node: 'Node', weight: Optional[float] = None):
        self.target_node: Optional['Node'] = target_node
        raw_weight = weight if weight is not None else random.uniform(0.05, 0.3)
        self.weight: float = float(np.clip(raw_weight, 0.0, 1.0))
    def __repr__(self) -> str:
        target_label = getattr(self.target_node, 'label', 'None')
        return f"<Conn to:{target_label} W:{self.weight:.3f}>"

class Node:
    """Basisklasse für alle Knoten im Quantum-Arona Netzwerk."""
    # Default Qubit Anzahl für neue Knoten
    DEFAULT_NUM_QUBITS = 4 # Beispielwert, kann konfiguriert werden

    def __init__(self, label: str, num_qubits: Optional[int] = None, is_quantum: bool = True, neuron_type: str = "excitatory"):
        self.label: str = label
        self.neuron_type: str = neuron_type
        self.is_quantum = is_quantum
        self.connections: List[Connection] = []
        self.activation: float = 0.0 # Letzte berechnete Aktivierung
        self.activation_sum: float = 0.0 # Klassischer Input für nächste Aktivierung
        self.num_qubits = num_qubits if num_qubits is not None else self.DEFAULT_NUM_QUBITS

        self.q_system: Optional[QuantumNodeSystem] = None
        if self.is_quantum and self.num_qubits > 0:
            try: self.q_system = QuantumNodeSystem(num_qubits=self.num_qubits)
            except Exception as e: print(f"ERROR init QNS for {self.label}: {e}"); self.q_system = None
        elif self.is_quantum: print(f"WARNING: Node {label} is quantum but num_qubits={self.num_qubits}. No QSystem.")

        self.last_measurement_log: List[Dict] = [] # Speichert die Messergebnisse des letzten activate-Aufrufs
        self.last_state_vector: Optional[np.ndarray] = None # Speichert den letzten Zustand vor der Messung

    def add_connection(self, target_node: 'Node', weight: Optional[float] = None):
        if target_node is self or target_node is None: return
        if not any(conn.target_node == target_node for conn in self.connections):
            self.connections.append(Connection(target_node, weight))

    def calculate_activation(self, n_shots: int):
        """Berechnet die nächste Aktivierung basierend auf activation_sum."""
        if self.is_quantum and self.q_system:
            # Quanten-Aktivierung durchführen
            try:
                self.activation, self.last_state_vector, self.last_measurement_log = self.q_system.activate(
                    self.activation_sum, n_shots
                )
            except Exception as e:
                print(f"ERROR during quantum activation for {self.label}: {e}")
                #traceback.print_exc() # Optional für Debugging
                self.activation = 0.0
                self.last_state_vector = None
                self.last_measurement_log = []
        else:
            # Klassische Aktivierung (z.B. für ValueNodes oder als Fallback)
            # Einfaches Beispiel: Sigmoid der Summe
            self.activation = 1 / (1 + np.exp(-float(self.activation_sum)))
            self.last_state_vector = None # Keine Quantenzustände für klassische Knoten
            self.last_measurement_log = []

    def get_state_representation(self) -> Dict[str, Any]:
        """Gibt eine repräsentative Momentaufnahme des Knotenzustands zurück."""
        state = {
            "label": self.label,
            "activation": self.activation,
            "type": type(self).__name__,
            "is_quantum": self.is_quantum,
        }
        if self.is_quantum and self.q_system:
            state["num_qubits"] = self.num_qubits
            # Optional: Komprimierte Repräsentation des Zustandsvektors oder der Parameter hinzufügen
            # state["q_params_hash"] = hash(self.q_system.get_params().tobytes()) # Beispiel: Hash der Parameter
            # state["state_vector_norm"] = np.linalg.norm(self.last_state_vector) if self.last_state_vector is not None else None
            # Füge Sprung-Analyse hinzu (Berechnung muss woanders erfolgen)
            state["last_jump_info"] = self.analyze_jumps(self.last_measurement_log)

        # Füge spezifische Infos für Module/Werte hinzu
        if isinstance(self, LimbusAffektus): state["emotion_state"] = self.emotion_state.copy()
        if isinstance(self, MetaCognitio): state["strategy_state"] = self.strategy_state.copy()
        # ... weitere Modulzustände ...
        return state

    def analyze_jumps(self, measurement_log: List[Dict]) -> Dict[str, Any]:
        """Analysiert Sprungverhalten im letzten Mess-Log."""
        if not measurement_log or len(measurement_log) < 2:
            return {"jump_detected": False, "max_jump": 0, "avg_jump": 0.0}

        jumps = []
        indices = [m['index'] for m in measurement_log if 'index' in m]
        for i in range(len(indices) - 1):
            jump_size = abs(indices[i+1] - indices[i])
            jumps.append(jump_size)

        if not jumps:
            return {"jump_detected": False, "max_jump": 0, "avg_jump": 0.0}

        max_jump = max(jumps)
        avg_jump = np.mean(jumps)
        # Definition eines "signifikanten" Sprungs (Beispiel: > 1/4 des Zustandsraums)
        significant_threshold = (2**self.num_qubits) / 4 if self.is_quantum and self.num_qubits > 0 else 1
        jump_detected = max_jump > significant_threshold

        return {
            "jump_detected": jump_detected,
            "max_jump": max_jump,
            "avg_jump": round(avg_jump, 3),
            "significant_threshold": round(significant_threshold, 1)
        }

    def __repr__(self) -> str:
        act_str = f"{self.activation:.3f}"
        q_info = ""
        if self.is_quantum and self.q_system: q_info = f" Q:{self.num_qubits}"
        elif not self.is_quantum: q_info = " (Cls)"
        return f"<{type(self).__name__} {self.label} Act:{act_str}{q_info} Conns:{len(self.connections)}>"


# === Kognitive Module (Vollständige Logik aus QNP-H v2.1 übernommen) ===

# --- Emotion (PAD) Model Parameters (Globale Konstanten) ---
EMOTION_DIMENSIONS = ["pleasure", "arousal", "dominance"]
INITIAL_EMOTION_STATE = {dim: 0.0 for dim in EMOTION_DIMENSIONS}
EMOTION_UPDATE_RATE = 0.03
EMOTION_VOLATILITY = 0.02
EMOTION_DECAY_TO_NEUTRAL = 0.05
CURRENT_EMOTION_STATE = INITIAL_EMOTION_STATE.copy() # Aktueller globaler Zustand

class LimbusAffektus(Node):
    """Models the emotional state (PAD) based on network activity."""
    def __init__(self, label: str = "Limbus Affektus", num_qubits: int = Node.DEFAULT_NUM_QUBITS, neuron_type: str = "interneuron"):
        super().__init__(label, num_qubits=num_qubits, is_quantum=True, neuron_type=neuron_type)
        self.emotion_state = INITIAL_EMOTION_STATE.copy()

    def update_emotion_state(self, all_nodes: List['Node'], module_outputs: Dict[str, deque]):
        """Updates the PAD emotional state based on network signals. (Logik von QNP-H)"""
        global CURRENT_EMOTION_STATE # Globalen Zustand aktualisieren

        pleasure_signal = 0.0; arousal_signal = 0.0; dominance_signal = 0.0
        relevant_nodes = [n for n in all_nodes if hasattr(n, 'activation') and isinstance(n.activation, (float, np.number)) and n.activation > 0.1 and not np.isnan(n.activation) and hasattr(n, 'label')]
        activations = [float(n.activation) for n in relevant_nodes]
        avg_act = np.mean(activations) if activations else 0.0
        std_act = np.std(activations) if len(activations) > 1 else 0.0

        # --- Pleasure Calculation ---
        pos_triggers = ["chance", "positiv", "erfolg", "gut", "ja", "innovation", "lösung"]
        neg_triggers = ["risiko", "problem", "negativ", "fehler", "schlecht", "nein", "kritik"]
        for node in relevant_nodes:
            if not isinstance(node.label, str): continue
            label_lower = node.label.lower()
            is_pos = any(trigger in label_lower for trigger in pos_triggers)
            is_neg = any(trigger in label_lower for trigger in neg_triggers)
            if is_pos and not is_neg: pleasure_signal += node.activation * 0.7
            elif is_neg and not is_pos: pleasure_signal -= node.activation * 0.9
        # Critic influence
        critic_evals_deque = module_outputs.get("Cortex Criticus")
        if critic_evals_deque and isinstance(critic_evals_deque[-1], list):
            scores = [e.get('score', 0.5) for e in critic_evals_deque[-1] if isinstance(e, dict) and isinstance(e.get('score'), (float, np.number))]
            if scores: pleasure_signal += (np.mean(scores) - 0.5) * 1.5

        # --- Arousal Calculation ---
        # Simplified: Use avg_act and std_act
        arousal_signal = float(np.clip(avg_act * 0.4 + std_act * 0.3, 0, 1)) # Removed activation change for simplicity here

        # --- Dominance Calculation ---
        meta_cog_node = next((n for n in all_nodes if isinstance(n, MetaCognitio)), None)
        meta_cog_act = float(meta_cog_node.activation) if meta_cog_node and hasattr(meta_cog_node, 'activation') else 0.0
        control_proxy = 1.0 - std_act
        dominance_signal = float(np.clip(meta_cog_act * 0.5 + control_proxy * 0.5, 0, 1))

        # --- Update PAD State ---
        for dim, signal in [("pleasure", pleasure_signal), ("arousal", arousal_signal), ("dominance", dominance_signal)]:
            current_val = self.emotion_state.get(dim, 0.0)
            target_val = np.clip(float(signal), -1.0, 1.0) if dim == "pleasure" else np.clip(float(signal) * 2.0 - 1.0, -1.0, 1.0)
            decayed_val = current_val * (1.0 - EMOTION_DECAY_TO_NEUTRAL)
            change_emo = (target_val - decayed_val) * EMOTION_UPDATE_RATE
            noise = np.random.normal(0, EMOTION_VOLATILITY)
            self.emotion_state[dim] = float(np.clip(decayed_val + change_emo + noise, -1.0, 1.0))

        CURRENT_EMOTION_STATE = self.emotion_state.copy()
        # Eigene Aktivierung des Moduls spiegelt emotionale Intensität wider
        self.activation = float(np.mean(np.abs(list(self.emotion_state.values()))))
        # Aktivierung muss hier gesetzt werden, da sie nicht durch calculate_activation erfolgt
        # Wir überschreiben den Quanten-Output mit der emotionalen Intensität
        # -> Alternative: Emotion beeinflusst den input_strength des QNS stärker?
        return self.emotion_state

    def get_emotion_influence_factors(self) -> Dict[str, float]:
        """Gibt Faktoren zur Modulation anderer Prozesse zurück. (Logik von QNP-H)"""
        p = float(self.emotion_state.get("pleasure", 0.0)); a = float(self.emotion_state.get("arousal", 0.0)); d = float(self.emotion_state.get("dominance", 0.0))
        return {
            "signal_modulation": 1.0 + p * 0.15,
            "learning_rate_factor": float(np.clip(1.0 + a * 0.25 + p * 0.10, 0.6, 1.8)),
            "exploration_factor": float(np.clip(1.0 + a * 0.35 - d * 0.20, 0.5, 1.7)),
            "criticism_weight_factor": float(np.clip(1.0 - p * 0.25 + d * 0.10, 0.6, 1.4)),
            "creativity_weight_factor": float(np.clip(1.0 + p * 0.20 + a * 0.10, 0.6, 1.7)),
        }

# Meta-Cognition Parameters
REFLECTION_LOG_MAX_LEN = 150
STAGNATION_DETECTION_WINDOW = 6
STAGNATION_THRESHOLD = 0.008
OSCILLATION_DETECTION_WINDOW = 8
OSCILLATION_THRESHOLD_STD = 0.3

class MetaCognitio(Node):
    """Monitors network state, logs reflections, and adapts learning strategies."""
    def __init__(self, label: str = "Meta Cognitio", num_qubits: int = Node.DEFAULT_NUM_QUBITS, neuron_type: str = "interneuron"):
        super().__init__(label, num_qubits=num_qubits, is_quantum=True, neuron_type=neuron_type)
        self.reflection_log: deque = deque(maxlen=REFLECTION_LOG_MAX_LEN)
        self.strategy_state: Dict[str, Any] = {"lr_boost": 1.0, "last_avg_activation": 0.5, "stagnation_counter": 0, "oscillation_detected": False}

    def log_reflection(self, message: str, epoch: int, data: Optional[Dict] = None):
        """Logs a meta-cognitive event."""
        log_entry = {"epoch": epoch, "timestamp": time.time(), "message": message, "data": data or {}}
        self.reflection_log.append(log_entry)
        # TODO: Optional persistent logging via PersistenceManager

    def adapt_strategy(self, condition: str):
        """Adjusts learning rate boost. (Logik von QNP-H)"""
        lr_boost_before = float(self.strategy_state.get("lr_boost", 1.0)); new_lr_boost = lr_boost_before
        if condition == "stagnation": new_lr_boost = min(lr_boost_before * 1.25, 2.5)
        elif condition == "oscillation": new_lr_boost = max(lr_boost_before * 0.75, 0.5)
        elif condition in ["stagnation_resolved", "oscillation_resolved"]: new_lr_boost = lr_boost_before * 0.9 + 1.0 * 0.1
        else: new_lr_boost = lr_boost_before * 0.98 + 1.0 * 0.02
        self.strategy_state["lr_boost"] = float(np.clip(new_lr_boost, 0.5, 2.5))
        if abs(self.strategy_state["lr_boost"] - lr_boost_before) > 0.01: print(f"[Meta] Strategy Update: {condition} -> LR Boost: {self.strategy_state['lr_boost']:.2f}")

    def get_meta_cognitive_state(self) -> Dict[str, Any]: return self.strategy_state.copy()

    def analyze_network_state(self, all_nodes: List['Node'], activation_history: Dict[str, deque], weights_history: Dict[str, deque], epoch: int):
        """Analyzes network for stagnation/oscillation. (Logik von QNP-H)"""
        nodes_with_history = [n for n in all_nodes if hasattr(n, 'activation') and hasattr(n, 'label') and isinstance(n.activation, (float, np.number)) and not np.isnan(n.activation) and n.label in activation_history and activation_history[n.label]]
        if not nodes_with_history: return
        activations = [float(n.activation) for n in nodes_with_history]
        avg_activation = np.mean(activations) if activations else 0.0

        # --- Stagnation Detection ---
        last_avg_activation_float = float(self.strategy_state.get("last_avg_activation", 0.5))
        activation_change = abs(avg_activation - last_avg_activation_float)
        if activation_change < STAGNATION_THRESHOLD and avg_activation < 0.7: self.strategy_state["stagnation_counter"] += 1
        else:
            if self.strategy_state["stagnation_counter"] >= STAGNATION_DETECTION_WINDOW: self.log_reflection(f"Stagnation resolved", epoch); self.adapt_strategy("stagnation_resolved")
            self.strategy_state["stagnation_counter"] = 0
        if self.strategy_state["stagnation_counter"] >= STAGNATION_DETECTION_WINDOW:
            if self.strategy_state["stagnation_counter"] == STAGNATION_DETECTION_WINDOW: self.log_reflection(f"Stagnation suspected", epoch, data={"avg_act": avg_activation, "change": activation_change}); self.adapt_strategy("stagnation")
        self.strategy_state["last_avg_activation"] = avg_activation

        # --- Oscillation Detection ---
        oscillating_nodes = []
        for label, history in activation_history.items():
            if len(history) >= OSCILLATION_DETECTION_WINDOW:
                window_hist = [h for h in list(history)[-OSCILLATION_DETECTION_WINDOW:] if isinstance(h, (float, np.number)) and not np.isnan(h)]
                if len(window_hist) >= OSCILLATION_DETECTION_WINDOW // 2:
                     if np.std(window_hist) > OSCILLATION_THRESHOLD_STD: oscillating_nodes.append(label)
        currently_oscillating = len(oscillating_nodes) > 0; was_oscillating = self.strategy_state.get("oscillation_detected", False)
        if currently_oscillating and not was_oscillating: self.log_reflection(f"Oscillations detected: {oscillating_nodes[:3]}...", epoch); self.adapt_strategy("oscillation"); self.strategy_state["oscillation_detected"] = True
        elif not currently_oscillating and was_oscillating: self.log_reflection("Oscillation resolved.", epoch); self.adapt_strategy("oscillation_resolved"); self.strategy_state["oscillation_detected"] = False


class CortexCreativus(Node):
    """Generates new ideas by combining or focusing on active concepts."""
    def __init__(self, label: str = "Cortex Creativus", num_qubits: int = Node.DEFAULT_NUM_QUBITS, neuron_type: str = "excitatory"):
        super().__init__(label, num_qubits=num_qubits, is_quantum=True, neuron_type=neuron_type or random.choice(["excitatory", "interneuron"]))

    def generate_new_ideas(self, active_nodes: List['Node'], creativity_factor: float = 1.0) -> List[str]:
        """Generates potential new ideas. (Logik von QNP-H)"""
        ideas = []; threshold = max(0.1, 0.5 / max(float(creativity_factor), 0.1))
        relevant_nodes = [n for n in active_nodes if hasattr(n, 'activation') and isinstance(n.activation, (float, np.number)) and not np.isnan(n.activation) and n.activation > threshold and hasattr(n, 'label')]
        relevant_nodes.sort(key=lambda n: float(n.activation), reverse=True)
        num_ideas_to_generate = int(1 + float(creativity_factor) * 1.2 + self.activation * 2.0)
        if len(relevant_nodes) >= 2:
            for i in range(min(num_ideas_to_generate // 2, len(relevant_nodes) - 1)):
                ideas.append(f"Idea_comb_{relevant_nodes[i].label[:8]}_and_{relevant_nodes[i+1].label[:8]}")
        if len(relevant_nodes) >= 1: ideas.append(f"Idea_focus_on_{relevant_nodes[0].label}")
        if float(creativity_factor) > 1.1 or (len(ideas) < num_ideas_to_generate and active_nodes):
             try:
                 random_node1 = random.choice(active_nodes)
                 potential_partners = [n for n in active_nodes if n != random_node1]
                 wild_idea = f"Wild_focus_{getattr(random_node1, 'label', '?')}"
                 if potential_partners: wild_idea = f"Wild_link_{getattr(random_node1, 'label', '?')[:8]}_{getattr(random.choice(potential_partners), 'label', '?')[:8]}"
                 if wild_idea not in ideas: ideas.append(wild_idea)
             except IndexError: pass
        return ideas[:num_ideas_to_generate]

class SimulatrixNeuralis(Node):
    """Simulates potential future scenarios."""
    def __init__(self, label: str = "Simulatrix Neuralis", num_qubits: int = Node.DEFAULT_NUM_QUBITS, neuron_type: str = "excitatory"):
        super().__init__(label, num_qubits=num_qubits, is_quantum=True, neuron_type=neuron_type or random.choice(["excitatory", "interneuron"]))

    def simulate_scenarios(self, active_nodes: List['Node']) -> List[str]:
        """Generates hypothetical scenarios. (Logik von QNP-H)"""
        scenarios = []
        pleasure = CURRENT_EMOTION_STATE.get("pleasure", 0.0)
        mood_modifier = "Optimistic" if pleasure > 0.25 else ("Pessimistic" if pleasure < -0.25 else "Neutral")
        scenario_nodes = [n for n in active_nodes if hasattr(n, 'activation') and isinstance(n.activation, (float, np.number)) and not np.isnan(n.activation) and n.activation > 0.65 and hasattr(n, 'label')]
        scenario_nodes.sort(key=lambda n: float(n.activation), reverse=True)
        value_nodes_dict = {n.label: float(n.activation) for n in active_nodes if isinstance(n, ValueNode) and hasattr(n,'label')}

        for node in scenario_nodes[:3]:
            scenarios.append(f"{mood_modifier}Scenario_if_{node.label}_dominates(Act:{node.activation:.2f})")
            node_label_lower = node.label.lower() if isinstance(node.label, str) else ""
            if value_nodes_dict.get("Sicherheit", 0.0) > 0.6 and "risiko" not in node_label_lower: scenarios.append(f"CautiousVar_of_{node.label}")
            if value_nodes_dict.get("Innovation", 0.0) > 0.6 and "chance" not in node_label_lower: scenarios.append(f"InnovativeVar_of_{node.label}")
        return scenarios

class CortexCriticus(Node):
    """Evaluates ideas and scenarios."""
    def __init__(self, label: str = "Cortex Criticus", num_qubits: int = Node.DEFAULT_NUM_QUBITS, neuron_type: str = "inhibitory"):
        super().__init__(label, num_qubits=num_qubits, is_quantum=True, neuron_type=neuron_type or "inhibitory")

    def evaluate_items(self, items_to_evaluate: List[str], current_network_state_nodes: List['Node'], criticism_factor: float = 1.0) -> List[Dict]:
        """Assigns a critique score. (Logik von QNP-H)"""
        evaluated = []
        if not items_to_evaluate: return evaluated
        value_nodes = {n.label: float(n.activation) for n in current_network_state_nodes if isinstance(n, ValueNode) and hasattr(n,'label')}
        sicherheit_val = value_nodes.get("Sicherheit", 0.5); ethik_val = value_nodes.get("Ethik", 0.5)
        pleasure = CURRENT_EMOTION_STATE.get("pleasure", 0.0)
        base_criticism = 0.4 + (self.activation * 0.3) + (float(criticism_factor) - 1.0) * 0.15 - pleasure * 0.2

        for item in items_to_evaluate:
            score_adjustment = 0.0; item_lower = item.lower() if isinstance(item, str) else ""
            if "risiko" in item_lower or "problem" in item_lower or "pessimistic" in item_lower: score_adjustment -= 0.25 * sicherheit_val
            if "chance" in item_lower or "potential" in item_lower or "optimistic" in item_lower: score_adjustment += 0.15 * (1.0 - sicherheit_val)
            if "ethik" in item_lower or "moral" in item_lower: score_adjustment += 0.2 * ethik_val
            if "wild" in item_lower: score_adjustment -= 0.1 * float(criticism_factor)
            if "cautious" in item_lower: score_adjustment += 0.05 * sicherheit_val
            if "innovative" in item_lower: score_adjustment += 0.08 * (1.0 - sicherheit_val)
            raw_score = base_criticism + score_adjustment + random.uniform(-0.04, 0.04)
            final_score = float(np.clip(raw_score, 0.0, 1.0))
            evaluated.append({"item": item, "score": round(final_score, 3)})
        return evaluated

class CortexSocialis(Node):
    """Models social context awareness."""
    def __init__(self, label: str = "Cortex Socialis", num_qubits: int = Node.DEFAULT_NUM_QUBITS, neuron_type: str = "excitatory"):
        super().__init__(label, num_qubits=num_qubits, is_quantum=True, neuron_type=neuron_type or random.choice(["excitatory", "interneuron"]))
        self.social_context_factors: Dict[str, float] = {} # Hält z.B. wahrgenommene Relevanz anderer Knoten/Konzepte

    def update_social_factors(self, all_nodes: List['Node']) -> Dict[str, float]:
        """Updates perceived social relevance based on network state. (Logik von QNP-H)"""
        dominance = CURRENT_EMOTION_STATE.get("dominance", 0.0)
        global_influence_mod = 1.0 + dominance * 0.05
        # Initialisiere Faktoren, wenn leer (Beispiel)
        if not self.social_context_factors:
            self.social_context_factors = {n.label: random.uniform(0.3, 0.7) for n in all_nodes if isinstance(n, MemoryNode) and hasattr(n,'label')}

        target_nodes = [n for n in all_nodes if isinstance(n, MemoryNode) and hasattr(n, 'activation') and hasattr(n, 'label')]
        for node in target_nodes:
            if node.label not in self.social_context_factors: continue
            activation = float(node.activation); current_factor = self.social_context_factors[node.label]; change_factor = 0.0
            if activation > 0.75: change_factor = 0.04
            elif activation < 0.35: change_factor = -0.02
            new_factor = current_factor + change_factor * global_influence_mod
            self.social_context_factors[node.label] = float(np.clip(new_factor, 0.05, 0.95))
        return self.social_context_factors.copy()

class MemoryNode(Node):
     """Repräsentiert ein Konzept/Thema (aus Trainingsdaten extrahiert)."""
     def __init__(self, label: str, num_qubits: int = Node.DEFAULT_NUM_QUBITS, neuron_type: str = "excitatory"):
        super().__init__(label, num_qubits=num_qubits, is_quantum=True, neuron_type=neuron_type)
        # Memory type ist im Training weniger relevant, könnte aber Metadaten speichern

class ValueNode(Node):
     """Repräsentiert Trainingsziele oder Kontext-Bias."""
     def __init__(self, label: str, initial_value: float = 0.5):
        super().__init__(label, num_qubits=0, is_quantum=False, neuron_type="excitatory")
        self.activation = float(np.clip(initial_value, 0.0, 1.0)) # Direkte klassische Aktivierung
     def update_value(self, adjustment: float):
         """Werte werden im Training ggf. extern gesetzt, statt intern angepasst."""
         self.activation = float(np.clip(self.activation + adjustment, 0.0, 1.0))
         # Alternative: self.activation = ziel_wert # Direktes Setzen durch Trainer


# ########################################################################
# # 3. Lernen & Parameter-Updates (Q-LLM spezifisch)
# ########################################################################

DEFAULT_CLASSICAL_LR = 0.01
DEFAULT_QUANTUM_LR = 0.005
DEFAULT_DECAY_RATE = 0.001

def calculate_dynamic_learning_rates(base_lr_classical: float, base_lr_quantum: float,
                                      emotion_state: Dict[str, float], meta_cognitive_state: Dict[str, Any]
                                      ) -> Tuple[float, float]:
    """Berechnet dynamische Lernraten für klassische Gewichte und Quantenparameter."""
    p = emotion_state.get("pleasure", 0.0); a = emotion_state.get("arousal", 0.0)
    lr_boost = meta_cognitive_state.get("lr_boost", 1.0)

    # Faktor basierend auf Emotion und Metakognition
    mod_factor = float(np.clip((1.0 + a * 0.2 + p * 0.1) * lr_boost, 0.3, 2.0))

    dynamic_lr_classical = float(np.clip(base_lr_classical * mod_factor, 0.0001, 0.1))
    dynamic_lr_quantum = float(np.clip(base_lr_quantum * mod_factor, 0.0001, 0.05)) # Quanten-LR oft kleiner

    return dynamic_lr_classical, dynamic_lr_quantum

def update_classical_weight(connection: Connection, delta_weight: float):
    """Aktualisiert ein klassisches Verbindungsgewicht."""
    if not np.isfinite(delta_weight): delta_weight = 0.0
    connection.weight = float(np.clip(connection.weight + delta_weight, 0.0, 1.0))

def update_quantum_params(node: 'Node', delta_params: np.ndarray):
    """Ruft die Update-Funktion des QNS auf."""
    if node.is_quantum and node.q_system:
        node.q_system.update_internal_params(delta_params)

def calculate_parameter_updates(
    network_state: Dict[str, Any], # Aktueller Zustand (Aktivierungen, Emotionen etc.)
    target_state: Dict[str, Any], # Gewünschter Zustand (aus Trainingsdaten)
    feedback_signal: float, # Externer Reward/Loss (z.B. 1.0 für gut, -1.0 für schlecht)
    lr_classical: float,
    lr_quantum: float
) -> Dict[str, Any]:
    """
    Berechnet die notwendigen Updates für klassische Gewichte und Quantenparameter.
    -> Dies ist die **KERN-HERAUSFORDERUNG** des Q-LLM Trainings!
    Muss die Differenz zwischen Ist- und Soll-Zustand in konkrete Delta-Werte übersetzen.
    """
    weight_updates: Dict[Tuple[str, str], float] = {} # (source_label, target_label) -> delta_weight
    q_param_updates: Dict[str, np.ndarray] = {} # node_label -> delta_params array

    # --- Beispielhafte, vereinfachte Logik ---
    # 1. Berechne "Fehler" oder "Abweichung" pro Knoten/Modul
    state_diff = {}
    current_activations = network_state.get("activations", {})
    target_activations = target_state.get("activations", {}) # Zielaktivierungen
    for label, current_act in current_activations.items():
        target_act = target_activations.get(label) # Was sollte dieser Knoten tun?
        if target_act is not None:
             # Einfacher Fehler: Differenz mal externes Feedback
             # Positives Feedback verstärkt Richtung Ziel, negatives kehrt um
             error = (target_act - current_act) * feedback_signal
             state_diff[label] = error

    # 2. Übersetze Fehler in Updates (stark vereinfacht!)
    nodes_map = network_state.get("nodes_map", {}) # Brauchen Zugriff auf Knotenobjekte
    for source_label, source_node in nodes_map.items():
        # Quantenparameter-Update (Beispiel: Verschiebe Parameter in Richtung Fehler)
        if source_node.is_quantum and source_node.q_system and source_label in state_diff:
            node_error = state_diff[source_label]
            # SEHR simple Annahme: Fehler beeinflusst alle Parameter leicht
            delta_q = np.ones(source_node.q_system.num_params) * node_error * lr_quantum * 0.1 # Kleiner Skalierungsfaktor
            q_param_updates[source_label] = delta_q

        # Klassische Gewichts-Updates (Beispiel: Hebbisch + Fehler)
        if hasattr(source_node, 'connections'):
             source_act = current_activations.get(source_label, 0.0)
             for conn in source_node.connections:
                 target_node = conn.target_node
                 if target_node and hasattr(target_node, 'label'):
                     target_label = target_node.label
                     target_act = current_activations.get(target_label, 0.0)
                     target_error = state_diff.get(target_label, 0.0) # Fehler des Zielknotens

                     # Modifizierte Hebbsche Regel mit Fehlerterm
                     # Verstärke, wenn beide aktiv UND Ziel-Fehler positiv ist (oder Quell-Fehler?)
                     # Schwäche, wenn aktiv, aber Ziel-Fehler negativ
                     # -> Braucht sorgfältiges Design!
                     delta_w = lr_classical * source_act * target_act * target_error # Nur als Platzhalteridee
                     weight_updates[(source_label, target_label)] = delta_w

    return {"weight_updates": weight_updates, "q_param_updates": q_param_updates}


# ########################################################################
# # 4. Datenverarbeitung & Kontext (Q-LLM spezifisch)
# ########################################################################

class DatasetLoader:
    """Lädt und zerlegt Trainingsdateien in verarbeitbare Chunks."""
    def __init__(self, file_paths: List[str], chunk_size: int = 500, overlap: int = 50):
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _read_file(self, file_path: str) -> Optional[str]:
        """Liest eine Textdatei."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"ERROR reading file {file_path}: {e}")
            return None

    def generate_chunks(self) -> Generator[Dict[str, Any], None, None]:
        """Erzeugt Chunks aus den Dateien als Dictionarys."""
        for file_path in self.file_paths:
            content = self._read_file(file_path)
            if content:
                print(f"Processing file: {file_path} (Length: {len(content)})")
                start = 0
                chunk_id_counter = 0
                while start < len(content):
                    end = start + self.chunk_size
                    chunk_text = content[start:end]
                    yield {
                        "file_id": os.path.basename(file_path),
                        "chunk_id": f"{os.path.basename(file_path)}_{chunk_id_counter}",
                        "text": chunk_text,
                        "offset": start
                    }
                    start += self.chunk_size - self.overlap # Nächster Chunk mit Überlappung
                    chunk_id_counter += 1
            else:
                 print(f"Skipping empty or unreadable file: {file_path}")


class Contextualizer:
    """Fügt Trainingszielen (Emotion, Fokus etc.) zu Text-Chunks hinzu."""
    def __init__(self, default_emotion: Dict = None, default_target: Dict = None):
        self.default_emotion = default_emotion or {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}
        self.default_target = default_target or {"target_category": None, "expected_jump_freq": "mittel"}

    def add_context(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fügt Kontext basierend auf Regeln oder Metadaten hinzu."""
        # Beispiel: Einfache Regel basierend auf Dateinamen oder Inhalt
        chunk_data['context'] = {
            "emotion": self.default_emotion.copy(),
            "target": self.default_target.copy()
        }
        if "ethik" in chunk_data.get('file_id', '').lower():
            chunk_data['context']['target']['target_category'] = "Ethik"
            chunk_data['context']['emotion'] = {"pleasure": 0.1, "arousal": 0.3, "dominance": 0.2} # Beispiel
        elif "technik" in chunk_data.get('file_id', '').lower():
             chunk_data['context']['target']['target_category'] = "Technologie"
             chunk_data['context']['emotion'] = {"pleasure": 0.2, "arousal": 0.5, "dominance": 0.4} # Beispiel
        # TODO: Ausgereiftere Logik zur Kontextbestimmung (z.B. Keyword-Analyse, Metadaten aus YAML)
        return chunk_data


# ########################################################################
# # 5. Zustandsextraktion & Embedding (Q-LLM spezifisch)
# ########################################################################

class StateExtractor:
    """Extrahiert relevante Zustandsinformationen aus dem Netzwerk."""
    def __init__(self, network_nodes: List[Node]):
        self.nodes = network_nodes
        self.node_map = {n.label: n for n in network_nodes if hasattr(n, 'label')}

    def extract_current_state(self) -> Dict[str, Any]:
        """Extrahiert den aktuellen Gesamtnetzwerkzustand."""
        activations = {n.label: n.activation for n in self.nodes if hasattr(n, 'label')}
        module_states = {n.label: n.get_state_representation() for n in self.nodes if not isinstance(n, MemoryNode) and not isinstance(n, ValueNode)} # Oder spezifische Module abfragen
        emotion = CURRENT_EMOTION_STATE.copy()
        # Extrahiere Sprunginformationen von allen Quantenknoten
        jump_summary = {n.label: n.analyze_jumps(n.last_measurement_log) for n in self.nodes if n.is_quantum and n.q_system}

        return {
            "timestamp": time.time(),
            "activations": activations,
            "module_specific_states": module_states, # Enthält z.B. PAD-Werte von Limbus
            "current_emotion": emotion,
            "jump_summary": jump_summary,
            "nodes_map": self.node_map # Referenz auf die Knotenobjekte für Updater
            # TODO: Optional: Klassische Gewichte, Q-Parameter (können groß sein!)
        }

class StateEmbedder:
    """(Platzhalter) Wandelt den extrahierten Zustand in einen Vektor um."""
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        # TODO: Hier könnte ein vortrainiertes Modell oder eine Dimensionsreduktionstechnik initialisiert werden

    def embed_state(self, network_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Erzeugt einen Vektor-Embedding aus dem Netzwerkzustand."""
        # --- SEHR VEREINFACHTE BEISPIEL-IMPLEMENTIERUNG ---
        # Sammle numerische Werte: Aktivierungen, Emotionen, Sprung-Metriken
        features = []
        activations = network_state.get("activations", {})
        features.extend(list(activations.values()))

        emotion = network_state.get("current_emotion", {})
        features.extend([emotion.get("pleasure", 0), emotion.get("arousal", 0), emotion.get("dominance", 0)])

        jumps = network_state.get("jump_summary", {})
        for jump_info in jumps.values():
            features.append(jump_info.get("max_jump", 0))
            features.append(jump_info.get("avg_jump", 0))

        if not features: return None

        # Konvertiere zu festem Vektor (Padding/Hashing/PCA/Autoencoder nötig für echte Implementierung)
        feature_vector = np.array(features, dtype=float)
        # Einfaches Hashing/Padding zur festen Dimension (NICHT SEMANTISCH SINNVOLL!)
        hashed_vector = np.zeros(self.embedding_dim)
        indices = np.abs(np.int64(feature_vector * 1000)) % self.embedding_dim # Beispielhafte Hash-Logik
        magnitudes = np.tanh(feature_vector) # Beispielhafte Skalierung
        np.add.at(hashed_vector, indices[:len(magnitudes)], magnitudes) # Addiere Werte an Hash-Positionen

        # Normalisieren
        norm = np.linalg.norm(hashed_vector)
        return hashed_vector / norm if norm > 0 else hashed_vector


# ########################################################################
# # 6. Trainings-Loop & Modell-Management (Q-LLM spezifisch)
# ########################################################################

class QuantumAronaModel:
    """Kapselt das gesamte Netzwerk und seine Zustände."""
    def __init__(self, config: Dict):
        self.config = config
        self.num_qubits_per_node = config.get("num_qubits_per_node", Node.DEFAULT_NUM_QUBITS)
        self.nodes: List[Node] = []
        self.node_map: Dict[str, Node] = {}
        self.global_emotion_state = INITIAL_EMOTION_STATE.copy() # Interner Zustand
        self.history: Dict[str, deque] = {} # Für Aktivierungs-, Gewichts-, etc. History

        self._initialize_network(config.get("network_structure", {}))

    def _initialize_network(self, structure_config: Dict):
        """Initialisiert die Knoten und Verbindungen."""
        print("Initializing Quantum Arona Network...")
        node_configs = structure_config.get("nodes", [])
        connection_configs = structure_config.get("connections", [])

        # 1. Knoten erstellen
        created_nodes = {}
        for node_conf in node_configs:
            label = node_conf.get("label")
            node_class_name = node_conf.get("class", "Node")
            if not label: continue
            try:
                # Finde die Klasse dynamisch
                node_class = globals().get(node_class_name, Node)
                # Übergib spezifische Parameter (num_qubits, initial_value etc.)
                params = {k: v for k, v in node_conf.items() if k not in ['label', 'class']}
                if 'num_qubits' not in params and issubclass(node_class, Node) and node_class != ValueNode:
                    params['num_qubits'] = self.num_qubits_per_node
                # Erstelle Instanz
                node_instance = node_class(label=label, **params)
                self.nodes.append(node_instance)
                created_nodes[label] = node_instance
            except Exception as e:
                print(f"ERROR creating node {label} ({node_class_name}): {e}")
        self.node_map = created_nodes
        print(f"Created {len(self.nodes)} nodes.")

        # 2. Verbindungen erstellen
        connections_added = 0
        for conn_conf in connection_configs:
            source_label = conn_conf.get("source")
            target_label = conn_conf.get("target")
            weight = conn_conf.get("weight")
            if source_label in self.node_map and target_label in self.node_map:
                self.node_map[source_label].add_connection(self.node_map[target_label], weight=weight)
                connections_added += 1
            else:
                print(f"Warning: Skipping connection {source_label} -> {target_label} (node not found).")
        print(f"Added {connections_added} connections.")

    def step(self, input_chunk: Optional[str] = None, context: Optional[Dict] = None, n_shots: int = 5):
        """Führt einen Simulationsschritt durch (Verarbeitung eines Chunks)."""
        # 1. Kontext anwenden (z.B. Emotionen setzen, ValueNodes (Ziele) anpassen)
        if context:
            emotion_context = context.get('emotion')
            if emotion_context:
                self.global_emotion_state.update(emotion_context)
                # Update Limbus Affektus state directly? Or let it influence dynamics?
                limbus = self.node_map.get("Limbus Affektus")
                if isinstance(limbus, LimbusAffektus): limbus.emotion_state = self.global_emotion_state.copy()

            target_values = context.get('target_values', {})
            for label, target_val in target_values.items():
                 if label in self.node_map and isinstance(self.node_map[label], ValueNode):
                     self.node_map[label].activation = float(target_val) # Zielwert setzen

            # TODO: Input Chunk verarbeiten und in Netzwerk-Input umwandeln
            #       -> Z.B. Aktivierung von MemoryNodes, die Keywords aus dem Chunk enthalten
            #       -> Dieser Teil braucht eine eigene, komplexere Logik (Embedding, Mapping)
            if input_chunk:
                self.apply_text_input(input_chunk)

        # 2. Klassischen Input berechnen (basierend auf Aktivierungen vom *letzten* Schritt)
        limbus = self.node_map.get("Limbus Affektus")
        emotion_factors = limbus.get_emotion_influence_factors() if isinstance(limbus, LimbusAffektus) else {}
        calculate_classic_input_sum(self.nodes, emotion_factors)

        # 3. Aktivierungen berechnen (Quanten & Klassisch)
        for node in self.nodes:
            # ValueNodes und andere klassische Knoten haben keine Quanten-Aktivierung
            if not node.is_quantum:
                 # Klassische Aktivierung für ValueNode wird extern gesetzt oder bleibt
                 if not isinstance(node, ValueNode):
                     node.calculate_activation(n_shots=0) # Verwende Standard-Sigmoid etc.
            else:
                 # Quanten-Aktivierung
                 node.calculate_activation(n_shots=n_shots)

        # 4. Module-Logik ausführen (Emotionen aktualisieren, etc.)
        # Wir brauchen eine definierte Reihenfolge oder parallele Ausführung
        module_outputs = {} # Sammelt Ausgaben für andere Module
        if isinstance(limbus, LimbusAffektus):
            self.global_emotion_state = limbus.update_emotion_state(self.nodes, module_outputs)
            # Optional: Logge Limbus Output
        # ... andere Module hier aufrufen (Criticus, Creativus etc.) ...
        # Ihre Logik kann den Zustand für den *nächsten* Schritt beeinflussen

        # 5. Zustand extrahieren
        extractor = StateExtractor(self.nodes)
        current_network_state = extractor.extract_current_state()
        return current_network_state

    def apply_text_input(self, text_chunk: str):
        """Wandelt Text-Input in initiale Netzwerkaktivierung um (SEHR VEREINFACHT)."""
        # Beispiel: Aktiviere MemoryNodes basierend auf simplen Keywords
        words = set(text_chunk.lower().split())
        for node in self.nodes:
            if isinstance(node, MemoryNode) and hasattr(node, 'label'):
                # Wenn der Knoten-Label (als Keyword) im Text vorkommt -> erhöhe Input
                if node.label.lower() in words:
                     node.activation_sum += 1.5 # Starker initialer Input
                # Komplexere Methode: Embeddings vergleichen o.ä.

    def apply_updates(self, updates: Dict[str, Any]):
        """Wendet berechnete Gewichts- und Parameter-Updates an."""
        weight_updates = updates.get("weight_updates", {})
        q_param_updates = updates.get("q_param_updates", {})

        # Klassische Gewichte aktualisieren
        for (source_label, target_label), delta_w in weight_updates.items():
            if source_label in self.node_map:
                source_node = self.node_map[source_label]
                if hasattr(source_node, 'connections'):
                    for conn in source_node.connections:
                        if conn.target_node and getattr(conn.target_node, 'label', None) == target_label:
                             update_classical_weight(conn, delta_w)
                             break # Angenommen, nur eine Verbindung pro Paar

        # Quantenparameter aktualisieren
        for node_label, delta_q in q_param_updates.items():
            if node_label in self.node_map:
                update_quantum_params(self.node_map[node_label], delta_q)

    def get_state(self) -> Dict[str, Any]:
        """Gibt den serialisierbaren Zustand des Modells zurück (für Checkpoints)."""
        nodes_state = []
        connections_state = []
        for node in self.nodes:
            n_state = {
                "label": node.label, "class": type(node).__name__,
                "activation": node.activation, "neuron_type": node.neuron_type,
                "is_quantum": node.is_quantum, "num_qubits": node.num_qubits
            }
            if node.is_quantum and node.q_system:
                n_state["q_params"] = node.q_system.get_params().tolist() # Als Liste speichern
            nodes_state.append(n_state)

            if hasattr(node, 'connections'):
                 for conn in node.connections:
                     if conn.target_node and hasattr(conn.target_node, 'label'):
                         connections_state.append({
                             "source": node.label, "target": conn.target_node.label, "weight": conn.weight
                         })
        return {
            "version": "quantum_arona_v1_checkpoint",
            "config": self.config,
            "nodes": nodes_state,
            "connections": connections_state,
            "emotion_state": self.global_emotion_state,
            # TODO: Speicher auch relevante History oder Modul-interne Zustände
        }

    def load_state(self, state_data: Dict[str, Any]):
        """Lädt einen gespeicherten Modellzustand."""
        print("Loading model state...")
        if state_data.get("version") != "quantum_arona_v1_checkpoint":
             print("Warning: Checkpoint version mismatch.")
        # Config überschreiben? Oder nur Netzwerk? Hier nur Netzwerk:
        loaded_nodes = {n['label']: n for n in state_data.get('nodes', [])}
        loaded_connections = {(c['source'], c['target']): c['weight'] for c in state_data.get('connections', [])}

        for node in self.nodes:
            if node.label in loaded_nodes:
                n_data = loaded_nodes[node.label]
                node.activation = n_data.get('activation', 0.0)
                if node.is_quantum and node.q_system and 'q_params' in n_data:
                    try:
                        params_list = n_data['q_params']
                        if params_list: node.q_system.set_params(np.array(params_list, dtype=float))
                    except Exception as e: print(f"Error loading QParams for {node.label}: {e}")

                if hasattr(node, 'connections'):
                     for conn in node.connections:
                         if conn.target_node and hasattr(conn.target_node, 'label'):
                             conn_key = (node.label, conn.target_node.label)
                             if conn_key in loaded_connections: conn.weight = loaded_connections[conn_key]
            else:
                 print(f"Warning: Node {node.label} found in current model but not in checkpoint.")

        self.global_emotion_state = state_data.get('emotion_state', INITIAL_EMOTION_STATE.copy())
        limbus = self.node_map.get("Limbus Affektus")
        if isinstance(limbus, LimbusAffektus): limbus.emotion_state = self.global_emotion_state.copy()
        print("Model state loaded.")


class QuantumTrainer:
    """Orchestriert den Trainingsprozess für Quantum-Arona."""
    def __init__(self, model: QuantumAronaModel, dataset_loader: DatasetLoader,
                 contextualizer: Contextualizer, state_embedder: StateEmbedder,
                 config: Dict):
        self.model = model
        self.loader = dataset_loader
        self.contextualizer = contextualizer
        self.embedder = state_embedder
        self.config = config
        self.lr_classical = config.get("learning_rate_classical", DEFAULT_CLASSICAL_LR)
        self.lr_quantum = config.get("learning_rate_quantum", DEFAULT_QUANTUM_LR)
        self.n_shots = config.get("simulation_shots", 5)
        self.epochs = config.get("training_epochs", 1)
        self.persistence_manager = None # TODO: PersistenceManager für Logs/Checkpoints initialisieren

    def _get_target_state(self, context: Dict) -> Dict[str, Any]:
        """Definiert den Zielzustand basierend auf dem Kontext."""
        # TODO: Implementiere Logik, um aus context['target'] einen Zielzustand abzuleiten
        # (z.B. hohe Aktivierung für target_category, spezifisches Emotionsprofil)
        return {"activations": {context.get("target",{}).get("target_category"): 1.0}} # Simples Beispiel

    def _calculate_feedback(self, current_state: Dict[str, Any], target_state: Dict[str, Any]) -> float:
        """Berechnet ein Feedback-Signal basierend auf Ist- und Soll-Zustand."""
        # TODO: Implementiere eine Metrik (z.B. Cosinus-Ähnlichkeit der Embeddings,
        #       Differenz der Aktivierungen, Erfüllung von Modul-Zielen)
        # Beispiel: Einfache Aktivierungsdifferenz für Zielkategorie
        target_cat = list(target_state.get("activations",{}).keys())[0] # Annahme: nur eine Zielkat.
        if target_cat:
            current_act = current_state.get("activations",{}).get(target_cat, 0.0)
            target_act = target_state.get("activations",{}).get(target_cat, 1.0)
            # Belohnung, wenn nah dran, Strafe wenn weit weg
            feedback = 1.0 - abs(target_act - current_act) # Wert zwischen 0 und 1
            return feedback * 2.0 - 1.0 # Skaliert auf [-1, 1] (Reward/Penalty)
        return 0.0 # Kein Feedback, wenn kein Ziel

    def train_epoch(self, epoch_num: int):
        """Führt eine Trainingsepoche durch."""
        print(f"\n--- Starting Training Epoch {epoch_num}/{self.epochs} ---")
        chunk_generator = self.loader.generate_chunks()
        total_loss = 0.0
        processed_chunks = 0

        # Verwende tqdm für Fortschrittsanzeige, falls verfügbar
        iterator = chunk_generator
        if TQDM_AVAILABLE: iterator = tqdm(iterator, desc=f"Epoch {epoch_num}", unit="chunk")

        for chunk_data in iterator:
            try:
                # 1. Kontext hinzufügen
                contextualized_chunk = self.contextualizer.add_context(chunk_data)
                context = contextualized_chunk.get("context", {})
                text_input = contextualized_chunk.get("text")

                # 2. Modell-Schritt ausführen -> aktuellen Zustand erhalten
                current_state = self.model.step(input_chunk=text_input, context=context, n_shots=self.n_shots)

                # 3. Zielzustand definieren
                target_state = self._get_target_state(context) # Beispiel

                # 4. Feedback berechnen
                feedback = self._calculate_feedback(current_state, target_state)
                # Einfacher "Loss" für Reporting (höher ist schlechter)
                loss = 1.0 - (feedback + 1.0) / 2.0 # Skaliert Feedback [-1,1] auf Loss [1,0]
                total_loss += loss
                processed_chunks += 1

                # 5. Dynamische Lernraten berechnen
                meta_cog = self.model.node_map.get("Meta Cognitio")
                meta_state = meta_cog.get_meta_cognitive_state() if isinstance(meta_cog, MetaCognitio) else {}
                dyn_lr_c, dyn_lr_q = calculate_dynamic_learning_rates(
                    self.lr_classical, self.lr_quantum, self.model.global_emotion_state, meta_state
                )

                # 6. Parameter-Updates berechnen
                updates = calculate_parameter_updates(
                    current_state, target_state, feedback, dyn_lr_c, dyn_lr_q
                )

                # 7. Updates anwenden
                self.model.apply_updates(updates)

                # TODO: Zustand/Embedding/Updates loggen (via PersistenceManager)
                # TODO: Optional: Plastizität anwenden? (Decay, Pruning, Sprouting)

                if TQDM_AVAILABLE and processed_chunks % 10 == 0: # Update tqdm Beschreibung
                   iterator.set_postfix({"avg_loss": f"{total_loss/processed_chunks:.4f}", "lr_q": f"{dyn_lr_q:.5f}"})

            except Exception as e:
                print(f"\nERROR processing chunk {chunk_data.get('chunk_id', 'N/A')}: {e}")
                traceback.print_exc() # Zeige detaillierten Fehler
                # Breche den Chunk ab, aber mache mit dem nächsten weiter? Oder Epoche abbrechen?
                # continue

        avg_epoch_loss = total_loss / processed_chunks if processed_chunks > 0 else 0
        print(f"--- Epoch {epoch_num} finished. Average Loss: {avg_epoch_loss:.5f} ---")
        return avg_epoch_loss

    def train(self):
        """Startet den gesamten Trainingsprozess."""
        for i in range(self.epochs):
            epoch_num = i + 1
            avg_loss = self.train_epoch(epoch_num)
            # TODO: Checkpoint speichern nach jeder Epoche (oder alle N Epochen)
            # self.save_checkpoint(f"quantum_arona_checkpoint_epoch_{epoch_num}.json")
            # TODO: Evtl. Evaluation auf Validierungsset nach jeder Epoche

    def save_checkpoint(self, filename: str):
        """Speichert den aktuellen Modellzustand."""
        try:
            state_data = self.model.get_state()
            # TODO: Speicher auch Optimierer-Zustände, Lernraten etc.
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2)
            print(f"Checkpoint saved: {filename}")
        except Exception as e:
            print(f"ERROR saving checkpoint {filename}: {e}")

    def load_checkpoint(self, filename: str):
        """Lädt einen Modellzustand aus einem Checkpoint."""
        if not os.path.exists(filename):
             print(f"ERROR: Checkpoint file not found: {filename}")
             return
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                 state_data = json.load(f)
            self.model.load_state(state_data)
            # TODO: Lade auch Optimierer-Zustände etc.
            print(f"Checkpoint loaded: {filename}")
        except Exception as e:
            print(f"ERROR loading checkpoint {filename}: {e}")


# ########################################################################
# # 7. Persistence Manager (Adaptiert für Training)
# ########################################################################
# TODO: PersistenceManager Klasse hier einfügen und anpassen für:
#       - Speichern von Trainingslogs pro Chunk/Epoch (Zustandsvektor, Loss, Feedback, Metadaten)
#       - Verwalten von Checkpoints
class PersistenceManager:
    """Verwaltet die Speicherung von Logs und Checkpoints."""
    # ... (Implementierung basierend auf QNP-H PersistenceManager, aber mit neuen Tabellen/Logik) ...
    pass


# ########################################################################
# # 8. Hilfsfunktionen & Konfiguration
# ########################################################################

def load_config(config_file: str = "config_arona.json") -> Dict:
    """Lädt die Konfigurationsdatei."""
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"ERROR loading config {config_file}: {e}. Using defaults.")
    # Default-Konfiguration als Fallback
    return {
        "num_qubits_per_node": 4,
        "simulation_shots": 10,
        "training_epochs": 3,
        "learning_rate_classical": 0.01,
        "learning_rate_quantum": 0.005,
        "dataset_files": ["./training_data/sample1.txt", "./training_data/sample2.md"], # Beispiel
        "chunk_size": 400,
        "chunk_overlap": 40,
        "embedding_dim": 64,
        "checkpoint_dir": "./checkpoints_arona",
        "log_db_path": "training_arona.db",
        "network_structure": { # Beispielhafte minimale Struktur
            "nodes": [
                {"label": "Limbus Affektus", "class": "LimbusAffektus"},
                {"label": "Meta Cognitio", "class": "MetaCognitio"},
                {"label": "Cortex Criticus", "class": "CortexCriticus"},
                {"label": "Cortex Creativus", "class": "CortexCreativus"},
                {"label": "Simulatrix Neuralis", "class": "SimulatrixNeuralis"},
                {"label": "Cortex Socialis", "class": "CortexSocialis"},
                {"label": "Concept_A", "class": "MemoryNode"}, # Beispiel-Konzept
                {"label": "Concept_B", "class": "MemoryNode"},
                {"label": "Ziel_Ethik", "class": "ValueNode", "initial_value": 0.5}, # Beispiel-Zielwert
            ],
            "connections": [ # Beispielhafte wenige Verbindungen
                {"source": "Concept_A", "target": "Limbus Affektus", "weight": 0.2},
                {"source": "Limbus Affektus", "target": "Meta Cognitio", "weight": 0.3},
                {"source": "Meta Cognitio", "target": "Cortex Criticus", "weight": 0.4},
                {"source": "Ziel_Ethik", "target": "Cortex Criticus", "weight": 0.5},
            ]
        }
    }

# --- Globale Variablen ---
# CURRENT_EMOTION_STATE wird jetzt im QuantumAronaModel verwaltet

# --- Ende quantum_arona_core.py ---
