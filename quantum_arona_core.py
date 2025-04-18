# -*- coding: utf-8 -*-
# Filename: quantum_arona_core.py
# Description: Kernarchitektur für Quantum-Arona - Zustandsbasiertes Q-LLM v1.1
#              Integriert QNP-H v2.1 Logik, Q-LLM Trainingskomponenten,
#              und EXPERIMENTELLE Strategien zur Raumspaltung/Dekohärenz-Triggerung.
# Author: [CipherCore Technology] & Gemini
# Status: Hoch experimenteller Prototyp - Vollständige Struktur mit neuen Lernstrategien

import numpy as np
import pandas as pd
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
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warnung: tqdm nicht gefunden. Fortschrittsbalken nicht verfügbar.")
    def tqdm(iterable, *args, **kwargs): return iterable

# ########################################################################
# # 1. Quanten-Engine (Qubit Simulation, PQC, Messung)
# ########################################################################

# === Basis-Gates & Helfer ===
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
P1 = np.array([[0, 0], [0, 1]], dtype=complex)

def _ry(theta: float) -> np.ndarray:
    if not np.isfinite(theta): theta = 0.0
    cos_t = np.cos(theta / 2); sin_t = np.sin(theta / 2)
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=complex)

def _rz(phi: float) -> np.ndarray:
    if not np.isfinite(phi): phi = 0.0
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
             # print(f"WARNING: Invalid delta_params for QParam update ({delta_params.shape if isinstance(delta_params, np.ndarray) else type(delta_params)} vs {self.params.shape}). Skipped.") # Verbose
             return
        if not np.all(np.isfinite(delta_params)):
             delta_params = np.nan_to_num(delta_params, nan=0.0, posinf=0.0, neginf=0.0)
             # print(f"WARNING: Non-finite values in delta_params clamped to 0.") # Verbose

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
    DEFAULT_NUM_QUBITS = 4
    DEFAULT_ACTIVATION_HISTORY_LEN = 50

    def __init__(self, label: str, num_qubits: Optional[int] = None, is_quantum: bool = True, neuron_type: str = "excitatory"):
        self.label: str = label
        self.neuron_type: str = neuron_type
        self.is_quantum = is_quantum
        self.connections: List[Connection] = []
        self.activation: float = 0.0
        self.activation_sum: float = 0.0
        self.activation_history: deque = deque(maxlen=self.DEFAULT_ACTIVATION_HISTORY_LEN) # Initialisiert

        self.num_qubits = num_qubits if num_qubits is not None else self.DEFAULT_NUM_QUBITS

        self.q_system: Optional[QuantumNodeSystem] = None
        if self.is_quantum and self.num_qubits > 0:
            try: self.q_system = QuantumNodeSystem(num_qubits=self.num_qubits)
            except Exception as e: print(f"ERROR init QNS for {self.label}: {e}"); self.q_system = None
        elif self.is_quantum: print(f"WARNING: Node {label} is quantum but num_qubits={self.num_qubits}.")

        self.last_measurement_log: List[Dict] = []
        self.last_state_vector: Optional[np.ndarray] = None
        self.last_successful_q_delta: Optional[np.ndarray] = None

    def add_connection(self, target_node: 'Node', weight: Optional[float] = None):
        if target_node is self or target_node is None: return
        if not any(conn.target_node == target_node for conn in self.connections):
            self.connections.append(Connection(target_node, weight))

    def calculate_activation(self, n_shots: int):
        """Berechnet die nächste Aktivierung und speichert sie in der History."""
        if self.is_quantum and self.q_system:
            try:
                new_activation, self.last_state_vector, self.last_measurement_log = self.q_system.activate(
                    self.activation_sum, n_shots
                )
                self.activation = new_activation
            except Exception as e:
                print(f"ERROR during quantum activation for {self.label}: {e}")
                self.activation = 0.0; self.last_state_vector = None; self.last_measurement_log = []
        else: # Klassische Aktivierung
            activation_sum_float = float(self.activation_sum) if isinstance(self.activation_sum, (float, np.number)) and np.isfinite(self.activation_sum) else 0.0
            safe_activation_sum = np.clip(activation_sum_float, -700, 700)
            try: self.activation = 1 / (1 + np.exp(-safe_activation_sum))
            except FloatingPointError: self.activation = 1.0 if safe_activation_sum > 0 else 0.0
            self.last_state_vector = None; self.last_measurement_log = []

        if not isinstance(self.activation, (float, np.number)) or not np.isfinite(self.activation): self.activation = 0.0
        self.activation_history.append(self.activation) # Immer zur History hinzufügen

    # *** NEU: Methode für geglättete Aktivierung ***
    def get_smoothed_activation(self, window: int = 3) -> float:
        """Berechnet den gleitenden Durchschnitt der letzten 'window' Aktivierungen."""
        if not self.activation_history:
            return self.activation # Fallback zur aktuellen Aktivierung, wenn History leer

        hist = list(self.activation_history)[-window:]
        valid_hist = [a for a in hist if isinstance(a, (float, np.number)) and np.isfinite(a)]

        if not valid_hist:
             return self.activation # Fallback, wenn keine gültigen Werte in der History
        else:
             return float(np.mean(valid_hist))

    def get_state_representation(self) -> Dict[str, Any]:
        """Gibt eine repräsentative Momentaufnahme des Knotenzustands zurück."""
        state = {"label": self.label, "activation": self.activation, "type": type(self).__name__, "is_quantum": self.is_quantum}
        if self.is_quantum and self.q_system:
            state["num_qubits"] = self.num_qubits
            state["last_jump_info"] = self.analyze_jumps(self.last_measurement_log) # Beinhaltet jetzt Varianz
        if isinstance(self, LimbusAffektus): state["emotion_state"] = self.emotion_state.copy()
        if isinstance(self, MetaCognitio): state["strategy_state"] = self.strategy_state.copy()
        return state

    def analyze_jumps(self, measurement_log: List[Dict]) -> Dict[str, Any]:
        """Analysiert Sprungverhalten UND Zustandsvarianz im letzten Mess-Log."""
        default_jump_info = {"jump_detected": False, "max_jump": 0, "avg_jump": 0.0, "state_variance": 0.0} # Varianz hinzugefügt
        if not measurement_log or len(measurement_log) < 2: return default_jump_info

        valid_indices = [m.get('index') for m in measurement_log]
        valid_indices = [idx for idx in valid_indices if isinstance(idx, (int, float, np.number)) and np.isfinite(idx)]
        if len(valid_indices) < 2: return default_jump_info

        jumps = np.abs(np.diff(np.array(valid_indices, dtype=float)))
        # *** NEU: Berechnung der Zustandsvarianz ***
        state_variance = np.var(valid_indices) if len(valid_indices) > 1 else 0.0 # Varianz nur sinnvoll bei >1 Messung

        if jumps.size == 0: return {"jump_detected": False, "max_jump": 0, "avg_jump": 0.0, "state_variance": round(state_variance, 3)}

        max_jump = np.max(jumps); avg_jump = np.mean(jumps)
        significant_threshold = 1.0
        if self.is_quantum and self.q_system and self.num_qubits > 0: significant_threshold = (2**self.num_qubits) / 4.0
        jump_detected = max_jump > significant_threshold

        return {
            "jump_detected": jump_detected, "max_jump": int(max_jump),
            "avg_jump": round(avg_jump, 3), "significant_threshold": round(significant_threshold, 1),
            "state_variance": round(state_variance, 3) # Varianz hinzugefügt
        }

    def __repr__(self) -> str:
        act_str = f"{self.activation:.3f}"
        q_info = ""
        if self.is_quantum and self.q_system: q_info = f" Q:{self.num_qubits}"
        elif not self.is_quantum: q_info = " (Cls)"
        return f"<{type(self).__name__} {self.label} Act:{act_str}{q_info} Conns:{len(self.connections)}>"


# === Kognitive Module (unverändert in ihrer internen Logik) ===
# (LimbusAffektus, MetaCognitio, CortexCreativus, SimulatrixNeuralis, CortexCriticus, CortexSocialis, MemoryNode, ValueNode)
# ... (Code für die Modulklassen hier einfügen - identisch zur vorherigen Version) ...
# --- Emotion (PAD) Model Parameters (Globale Konstanten) ---
EMOTION_DIMENSIONS = ["pleasure", "arousal", "dominance"]
INITIAL_EMOTION_STATE = {dim: 0.0 for dim in EMOTION_DIMENSIONS}
EMOTION_UPDATE_RATE = 0.03
EMOTION_VOLATILITY = 0.02
EMOTION_DECAY_TO_NEUTRAL = 0.05
# CURRENT_EMOTION_STATE wird jetzt im QuantumAronaModel verwaltet

class LimbusAffektus(Node):
    """Models the emotional state (PAD) based on network activity."""
    def __init__(self, label: str = "Limbus Affektus", num_qubits: int = Node.DEFAULT_NUM_QUBITS, neuron_type: str = "interneuron"):
        super().__init__(label, num_qubits=num_qubits, is_quantum=True, neuron_type=neuron_type)
        self.emotion_state = INITIAL_EMOTION_STATE.copy() # Interner Zustand des Moduls

    def update_emotion_state(self, all_nodes: List['Node'], module_outputs: Dict[str, deque]) -> Dict[str, float]:
        """Updates the PAD emotional state based on network signals. (Logik von QNP-H)"""
        # Diese Methode aktualisiert self.emotion_state
        pleasure_signal = 0.0; arousal_signal = 0.0; dominance_signal = 0.0
        relevant_nodes = [n for n in all_nodes if hasattr(n, 'activation') and isinstance(n.activation, (float, np.number)) and n.activation > 0.1 and not np.isnan(n.activation) and hasattr(n, 'label')]
        activations = [float(n.activation) for n in relevant_nodes]
        avg_act = np.mean(activations) if activations else 0.0
        std_act = np.std(activations) if len(activations) > 1 else 0.0

        # Pleasure Calculation
        pos_triggers = ["chance", "positiv", "erfolg", "gut", "ja", "innovation", "lösung"]
        neg_triggers = ["risiko", "problem", "negativ", "fehler", "schlecht", "nein", "kritik"]
        for node in relevant_nodes:
            if not isinstance(node.label, str): continue
            label_lower = node.label.lower()
            is_pos = any(trigger in label_lower for trigger in pos_triggers)
            is_neg = any(trigger in label_lower for trigger in neg_triggers)
            if is_pos and not is_neg: pleasure_signal += node.activation * 0.7
            elif is_neg and not is_pos: pleasure_signal -= node.activation * 0.9
        critic_evals = []
        if module_outputs and "Cortex Criticus" in module_outputs and module_outputs["Cortex Criticus"]:
             last_critic_output = module_outputs["Cortex Criticus"][-1]
             if isinstance(last_critic_output, list): critic_evals = last_critic_output
        scores = [e.get('score') for e in critic_evals if isinstance(e, dict) and isinstance(e.get('score'), (float, np.number)) and np.isfinite(e.get('score'))]
        if scores: pleasure_signal += (np.mean(scores) - 0.5) * 1.5

        # Arousal Calculation
        arousal_signal = float(np.clip(avg_act * 0.4 + std_act * 0.3, 0, 1))

        # Dominance Calculation
        meta_cog_node = next((n for n in all_nodes if isinstance(n, MetaCognitio)), None)
        meta_cog_act = float(meta_cog_node.activation) if meta_cog_node and hasattr(meta_cog_node, 'activation') and isinstance(meta_cog_node.activation, (float, np.number)) and np.isfinite(meta_cog_node.activation) else 0.0
        control_proxy = 1.0 - std_act
        dominance_signal = float(np.clip(meta_cog_act * 0.5 + control_proxy * 0.5, 0, 1))

        # Update PAD State
        for dim, signal in [("pleasure", pleasure_signal), ("arousal", arousal_signal), ("dominance", dominance_signal)]:
            current_val = self.emotion_state.get(dim, 0.0)
            target_val = np.clip(float(signal), -1.0, 1.0) if dim == "pleasure" else np.clip(float(signal) * 2.0 - 1.0, -1.0, 1.0)
            decayed_val = current_val * (1.0 - EMOTION_DECAY_TO_NEUTRAL)
            change_emo = (target_val - decayed_val) * EMOTION_UPDATE_RATE
            noise = np.random.normal(0, EMOTION_VOLATILITY)
            self.emotion_state[dim] = float(np.clip(decayed_val + change_emo + noise, -1.0, 1.0))

        # Eigene Aktivierung des Moduls (optional, überschreibt Quanten-Aktivierung!)
        self.activation = float(np.mean(np.abs(list(self.emotion_state.values()))))
        # Gibt den NEUEN Zustand zurück, der auch im globalen Modellzustand gespeichert wird
        return self.emotion_state.copy()

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

    def log_reflection(self, message: str, step: int, data: Optional[Dict] = None):
        """Logs a meta-cognitive event, kann jetzt auch Sprungdaten enthalten."""
        log_entry = {
            "step": step, # Schritt (Chunk-Nummer oder Epoche)
            "timestamp": time.time(),
            "message": message,
            "data": data or {} # Hier können z.B. {'jump_node': '...', 'loss_delta': ...} übergeben werden
            }
        self.reflection_log.append(log_entry)

    def adapt_strategy(self, condition: str, current_step: int): # Geändert: current_step
        """Adjusts learning rate boost. (Logik von QNP-H)"""
        lr_boost_before = float(self.strategy_state.get("lr_boost", 1.0)); new_lr_boost = lr_boost_before
        if condition == "stagnation": new_lr_boost = min(lr_boost_before * 1.25, 2.5)
        elif condition == "oscillation": new_lr_boost = max(lr_boost_before * 0.75, 0.5)
        elif condition in ["stagnation_resolved", "oscillation_resolved"]: new_lr_boost = lr_boost_before * 0.9 + 1.0 * 0.1
        else: new_lr_boost = lr_boost_before * 0.98 + 1.0 * 0.02
        self.strategy_state["lr_boost"] = float(np.clip(new_lr_boost, 0.5, 2.5))
        if abs(self.strategy_state["lr_boost"] - lr_boost_before) > 0.01:
            self.log_reflection(f"Strategy Update: {condition} -> LR Boost: {self.strategy_state['lr_boost']:.2f}", current_step)

    def get_meta_cognitive_state(self) -> Dict[str, Any]: return self.strategy_state.copy()

    def analyze_network_state(self, all_nodes: List['Node'], activation_history: Dict[str, deque], current_step: int): # Geändert: current_step
        """Analyzes network for stagnation/oscillation. (Logik von QNP-H)"""
        activations = [float(n.activation) for n in all_nodes if hasattr(n, 'activation') and isinstance(n.activation, (float, np.number)) and not np.isnan(n.activation)]
        avg_activation = np.mean(activations) if activations else 0.0
        if avg_activation < 0.1: # Beispiel: Wenn Netzwerk fast tot ist
            if self.strategy_state["stagnation_counter"] == 0: # Nur einmal loggen
                self.log_reflection("Stagnation suspected (low avg activation)", current_step, data={"avg_act": avg_activation})
                self.adapt_strategy("stagnation", current_step)
            self.strategy_state["stagnation_counter"] += 1
        else:
            if self.strategy_state["stagnation_counter"] > 0: # War vorher stagnierend
                 self.log_reflection("Stagnation resolved", current_step)
                 self.adapt_strategy("stagnation_resolved", current_step)
            self.strategy_state["stagnation_counter"] = 0

class CortexCreativus(Node):
    """Generates new ideas by combining or focusing on active concepts."""
    def __init__(self, label: str = "Cortex Creativus", num_qubits: int = Node.DEFAULT_NUM_QUBITS, neuron_type: str = "excitatory"):
        super().__init__(label, num_qubits=num_qubits, is_quantum=True, neuron_type=neuron_type or random.choice(["excitatory", "interneuron"]))

    def generate_new_ideas(self, active_nodes: List['Node'], creativity_factor: float = 1.0) -> List[str]:
        """Generates potential new ideas. (Logik von QNP-H)"""
        ideas = []; threshold = max(0.1, 0.5 / max(float(creativity_factor), 0.1))
        relevant_nodes = [n for n in active_nodes if hasattr(n, 'activation') and isinstance(n.activation, (float, np.number)) and not np.isnan(n.activation) and n.activation > threshold and hasattr(n, 'label')]
        relevant_nodes.sort(key=lambda n: float(n.activation), reverse=True)
        self_act = float(self.activation) if isinstance(self.activation, (float, np.number)) and np.isfinite(self.activation) else 0.0
        num_ideas_to_generate = int(1 + float(creativity_factor) * 1.2 + self_act * 2.0)
        if len(relevant_nodes) >= 2:
            for i in range(min(num_ideas_to_generate // 2, len(relevant_nodes) - 1)): ideas.append(f"Idea_comb_{relevant_nodes[i].label[:8]}_and_{relevant_nodes[i+1].label[:8]}")
        if len(relevant_nodes) >= 1: ideas.append(f"Idea_focus_on_{relevant_nodes[0].label}")
        if float(creativity_factor) > 1.1 or (len(ideas) < num_ideas_to_generate and active_nodes):
             try:
                 if not active_nodes: return ideas[:num_ideas_to_generate]
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
        value_nodes_dict = {n.label: float(n.activation) for n in active_nodes if isinstance(n, ValueNode) and hasattr(n,'label') and hasattr(n,'activation')}
        for node in scenario_nodes[:3]:
            node_act_f = float(node.activation) if isinstance(node.activation, (float, np.number)) and np.isfinite(node.activation) else 0.0
            scenarios.append(f"{mood_modifier}Scenario_if_{node.label}_dominates(Act:{node_act_f:.2f})")
            node_label_lower = node.label.lower() if isinstance(node.label, str) else ""
            sicherheit_act = float(value_nodes_dict.get("Sicherheit", 0.0))
            innovation_act = float(value_nodes_dict.get("Innovation", 0.0))
            if sicherheit_act > 0.6 and "risiko" not in node_label_lower: scenarios.append(f"CautiousVar_of_{node.label}")
            if innovation_act > 0.6 and "chance" not in node_label_lower: scenarios.append(f"InnovativeVar_of_{node.label}")
        return scenarios

class CortexCriticus(Node):
    """Evaluates ideas and scenarios."""
    def __init__(self, label: str = "Cortex Criticus", num_qubits: int = Node.DEFAULT_NUM_QUBITS, neuron_type: str = "inhibitory"):
        super().__init__(label, num_qubits=num_qubits, is_quantum=True, neuron_type=neuron_type or "inhibitory")

    def evaluate_items(self, items_to_evaluate: List[str], current_network_state_nodes: List['Node'], criticism_factor: float = 1.0) -> List[Dict]:
        """Assigns a critique score. (Logik von QNP-H)"""
        evaluated = []
        if not items_to_evaluate: return evaluated
        value_nodes = {n.label: float(n.activation) for n in current_network_state_nodes if isinstance(n, ValueNode) and hasattr(n,'label') and hasattr(n, 'activation')}
        sicherheit_val = float(value_nodes.get("Sicherheit", 0.5)); ethik_val = float(value_nodes.get("Ethik", 0.5))
        pleasure = CURRENT_EMOTION_STATE.get("pleasure", 0.0)
        self_act_float = float(self.activation) if isinstance(self.activation, (float, np.number)) and np.isfinite(self.activation) else 0.0
        base_criticism = 0.4 + (self_act_float * 0.3) + (float(criticism_factor) - 1.0) * 0.15 - pleasure * 0.2
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
        self.social_context_factors: Dict[str, float] = {} # Hält z.B. wahrgenommene Relevanz

    def update_social_factors(self, all_nodes: List['Node']) -> Dict[str, float]:
        """Updates perceived social relevance based on network state. (Logik von QNP-H)"""
        dominance = CURRENT_EMOTION_STATE.get("dominance", 0.0)
        global_influence_mod = 1.0 + dominance * 0.05
        memory_node_labels = {n.label for n in all_nodes if isinstance(n, MemoryNode) and hasattr(n,'label')}
        if not self.social_context_factors and memory_node_labels: self.social_context_factors = {label: random.uniform(0.3, 0.7) for label in memory_node_labels}
        target_nodes = [n for n in all_nodes if isinstance(n, MemoryNode) and hasattr(n, 'activation') and hasattr(n, 'label')]
        for node in target_nodes:
            if node.label not in self.social_context_factors: continue
            activation = float(node.activation) if isinstance(node.activation, (float, np.number)) and np.isfinite(node.activation) else 0.0
            current_factor = self.social_context_factors[node.label]; change_factor = 0.0
            if activation > 0.75: change_factor = 0.04
            elif activation < 0.35: change_factor = -0.02
            new_factor = current_factor + change_factor * global_influence_mod
            self.social_context_factors[node.label] = float(np.clip(new_factor, 0.05, 0.95))
        return self.social_context_factors.copy()

class MemoryNode(Node):
     """Repräsentiert ein Konzept/Thema (aus Trainingsdaten extrahiert)."""
     def __init__(self, label: str, num_qubits: int = Node.DEFAULT_NUM_QUBITS, neuron_type: str = "excitatory"):
        super().__init__(label, num_qubits=num_qubits, is_quantum=True, neuron_type=neuron_type)

class ValueNode(Node):
     """Repräsentiert Trainingsziele oder Kontext-Bias."""
     def __init__(self, label: str, initial_value: float = 0.5, **kwargs): # Fange zusätzliche kwargs ab
        super().__init__(label, num_qubits=0, is_quantum=False, neuron_type="excitatory")
        init_val_f = 0.5
        if isinstance(initial_value, (float, int, np.number)) and np.isfinite(initial_value): init_val_f = float(initial_value)
        self.activation = float(np.clip(init_val_f, 0.0, 1.0))
     def update_value(self, adjustment: float): pass


# ########################################################################
# # 3. Lernen & Parameter-Updates (Q-LLM spezifisch - Kombinierte Heuristik)
# ########################################################################

DEFAULT_CLASSICAL_LR = 0.01
DEFAULT_QUANTUM_LR = 0.005
DEFAULT_DECAY_RATE = 0.001 # Ggf. für klassische Gewichte nutzen

def calculate_dynamic_learning_rates(
    base_lr_classical: float, base_lr_quantum: float, emotion_state: Dict[str, float],
    meta_cognitive_state: Dict[str, Any], overall_feedback: Optional[float] = None,
    feedback_lr_scaling_factor: float = 0.5
) -> Tuple[float, float]:
    """Berechnet dynamische Lernraten."""
    p=emotion_state.get("pleasure",0.0); a=emotion_state.get("arousal",0.0); lr_boost=meta_cognitive_state.get("lr_boost",1.0)
    mod_factor = float(np.clip((1.0 + a*0.2 + p*0.1) * lr_boost, 0.3, 2.0))
    dynamic_lr_classical = float(np.clip(base_lr_classical * mod_factor, 0.0001, 0.1))
    dynamic_lr_quantum = base_lr_quantum * mod_factor
    if overall_feedback is not None and overall_feedback > 0:
        feedback_boost = 1.0 + (overall_feedback * feedback_lr_scaling_factor)
        dynamic_lr_quantum *= feedback_boost
    dynamic_lr_quantum = float(np.clip(dynamic_lr_quantum, 0.0001, 0.075))
    return dynamic_lr_classical, dynamic_lr_quantum

def update_classical_weight(connection: Connection, delta_weight: float):
    """Aktualisiert ein klassisches Verbindungsgewicht."""
    if not np.isfinite(delta_weight): delta_weight = 0.0
    connection.weight = float(np.clip(connection.weight + delta_weight, 0.0, 1.0))

def update_quantum_params(node: 'Node', delta_params: np.ndarray):
    """Ruft die Update-Funktion des QNS auf."""
    if node.is_quantum and node.q_system:
        node.q_system.update_internal_params(delta_params)

# *** NEUE GEGLÄTTETE HEBB FUNKTION ***
# *** NEUE GEGLÄTTETE HEBB FUNKTION ***
def hebbian_learning_quantum_node_smoothed(
    node_a: 'Node', connection: 'Connection',
    learning_rate_classical: float = 0.1,
    learning_rate_quantum: float = DEFAULT_QUANTUM_LR,
    weight_limit: float = 1.0, reg_factor: float = 0.001,
    history_window: int = 3, # Fenster für gleitenden Durchschnitt
    activation_threshold_high: float = 0.55, # Obere Aktivierungsschwelle
    activation_threshold_low: float = 0.30   # Untere Aktivierungsschwelle
    ):
    """
    MODIFIED Hebbian learning rule using SMOOTHED activations.
    - Updates classical weight based on smoothed pre/post activations.
    - Provides feedback to presynaptic quantum parameters based on smoothed signals.
    - Includes LTD and regularization.
    """
    node_b = connection.target_node
    # Stelle sicher, dass beide Knoten die Methode haben und gültig sind
    if not node_b or not hasattr(node_a, 'get_smoothed_activation') or not hasattr(node_b, 'get_smoothed_activation'):
        # print(f"DEBUG: Skipping Hebbian update. Invalid nodes or missing method. A:{node_a}, B:{node_b}") # Debugging
        return

    act_a_smooth = node_a.get_smoothed_activation(window=history_window)
    act_b_smooth = node_b.get_smoothed_activation(window=history_window)

    # Potentiation
    if act_a_smooth > activation_threshold_high and act_b_smooth > activation_threshold_high:
        delta_weight_classical = learning_rate_classical * act_a_smooth * act_b_smooth
        update_classical_weight(connection, delta_weight_classical)
        if node_a.is_quantum and node_a.q_system:
            q_system_a = node_a.q_system; param_delta = np.zeros_like(q_system_a.get_params())
            if len(param_delta) > 0:
                ry_indices = range(0, q_system_a.num_params, 2)
                # Stärke des Updates basiert auf Koinzidenz der geglätteten Signale
                param_delta[list(ry_indices)] = learning_rate_quantum * act_a_smooth * act_b_smooth * 0.5
            update_quantum_params(node_a, param_delta)
    # LTD
    elif act_a_smooth > activation_threshold_high and act_b_smooth < activation_threshold_low:
        delta_weight_ltd = -0.1 * learning_rate_classical * act_a_smooth
        update_classical_weight(connection, delta_weight_ltd)
        if node_a.is_quantum and node_a.q_system:
            q_system_a = node_a.q_system; param_delta = np.zeros_like(q_system_a.get_params())
            if len(param_delta) > 0:
                ry_indices = range(0, q_system_a.num_params, 2)
                param_delta[list(ry_indices)] = -0.1 * learning_rate_quantum * act_a_smooth * 0.5
            update_quantum_params(node_a, param_delta)

    # Regularization
    connection.weight = float(np.clip(float(connection.weight) - reg_factor * float(connection.weight), 0.0, weight_limit))


def calculate_parameter_updates( # Behalte die Funktion für den Fall, dass nicht-Hebb'sche Updates benötigt werden
    network_state: Dict[str, Any], target_state: Dict[str, Any], feedback_components: Dict[str, float],
    overall_feedback_signal: float, lr_classical: float, lr_quantum: float,
    activation_error_weight: float = 0.6, emotion_error_weight: float = 0.4,
    arousal_error_weight_rz: float = 0.2, dominance_error_weight_rz: float = 0.2
) -> Dict[str, Any]:
    """Berechnet Updates basierend auf Feedback und Zielzustand (NICHT Hebb'sch)."""
    # Unveränderte Logik von oben...
    weight_updates: Dict[Tuple[str, str], float] = {}
    q_param_updates: Dict[str, np.ndarray] = {}
    current_activations = network_state.get("activations", {}); target_activations = target_state.get("activations", {})
    nodes_map = network_state.get("nodes_map", {}); current_emotion = network_state.get("current_emotion", {})
    target_emotion = target_state.get("emotion", {})
    activation_feedback_01 = feedback_components.get("activation", 0.5); emotion_feedback_01 = feedback_components.get("emotion", 0.5)
    activation_error_01 = 1.0 - activation_feedback_01; emotion_error_01 = 1.0 - emotion_feedback_01
    error_arousal, error_dominance = 0.0, 0.0
    if target_emotion and current_emotion:
        target_aro = float(target_emotion.get("arousal", 0.0)); current_aro = float(current_emotion.get("arousal", 0.0))
        if np.isfinite(target_aro) and np.isfinite(current_aro): error_arousal = target_aro - current_aro
        target_dom = float(target_emotion.get("dominance", 0.0)); current_dom = float(current_emotion.get("dominance", 0.0))
        if np.isfinite(target_dom) and np.isfinite(current_dom): error_dominance = target_dom - current_dom
    for node_label, node in nodes_map.items():
        if not isinstance(node, Node) or not node.is_quantum or not node.q_system: continue
        num_params = node.q_system.num_params; delta_q = np.zeros(num_params)
        current_act = float(current_activations.get(node_label, 0.0)); target_act = float(target_activations.get(node_label, -1.0))
        signed_error_act = 0.0
        if target_act != -1.0: signed_error_act = target_act - current_act
        if not np.isfinite(signed_error_act): signed_error_act = 0.0
        error_magnitude = (activation_error_weight * abs(signed_error_act) + emotion_error_weight * emotion_error_01)
        if not np.isfinite(error_magnitude): error_magnitude = 0.0
        delta_theta_magnitude = np.sign(signed_error_act) * error_magnitude * lr_quantum * 0.5
        if not np.isfinite(delta_theta_magnitude): delta_theta_magnitude = 0.0
        for i in range(0, num_params, 2): delta_q[i] = delta_theta_magnitude
        delta_phi_magnitude = (arousal_error_weight_rz * error_arousal + dominance_error_weight_rz * error_dominance) * lr_quantum * 0.3
        if not np.isfinite(delta_phi_magnitude): delta_phi_magnitude = 0.0
        for i in range(1, num_params, 2): delta_q[i] = delta_phi_magnitude
        if np.any(delta_q): delta_q = np.nan_to_num(delta_q, nan=0.0, posinf=0.0, neginf=0.0); q_param_updates[node_label] = delta_q
    safe_overall_feedback = float(np.clip(overall_feedback_signal, -1.0, 1.0))
    for source_label, source_node in nodes_map.items():
        if not isinstance(source_node, Node) or not hasattr(source_node, 'connections'): continue
        source_act = float(current_activations.get(source_label, 0.0))
        if source_act < 0.05: continue
        for conn in source_node.connections:
            target_node = conn.target_node
            if not target_node or not hasattr(target_node, 'label'): continue
            target_label = target_node.label; target_act = float(current_activations.get(target_label, 0.0))
            if target_act < 0.05: continue
            delta_w = 0.0; joint_activity = source_act * target_act
            if safe_overall_feedback > 0: delta_w = lr_classical * joint_activity * safe_overall_feedback
            elif safe_overall_feedback < 0: delta_w = -lr_classical * joint_activity * abs(safe_overall_feedback) * 0.5
            if delta_w != 0.0 and np.isfinite(delta_w): weight_updates[(source_label, target_label)] = delta_w
    return {"weight_updates": weight_updates, "q_param_updates": q_param_updates}


# ########################################################################
# # 4. Datenverarbeitung & Kontext (Q-LLM spezifisch)
# ########################################################################
# DatasetLoader und Contextualizer bleiben unverändert
class DatasetLoader:
    def __init__(self, file_paths: List[str], chunk_size: int = 500, overlap: int = 50):
        self.file_paths = file_paths; self.chunk_size = chunk_size; self.overlap = overlap
        if self.overlap >= self.chunk_size and self.chunk_size > 0: self.overlap = max(0, self.chunk_size // 10)
        elif self.chunk_size <= 0: self.chunk_size = 500; self.overlap = 50
    def _read_file(self, file_path: str) -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: return f.read()
        except Exception as e: print(f"ERROR reading file {file_path}: {e}"); return None
    def generate_chunks(self) -> Generator[Dict[str, Any], None, None]:
        if not self.file_paths: print("WARNUNG: Keine Dateien im DatasetLoader."); return
        total_chunks_generated = 0
        for file_path in self.file_paths:
            content = self._read_file(file_path)
            if content:
                start = 0; chunk_id_counter = 0; file_chunk_count = 0
                while start < len(content):
                    end = start + self.chunk_size; chunk_text = content[start:end]
                    if not chunk_text.strip(): start += max(1, self.chunk_size - self.overlap); continue
                    yield {"file_id": os.path.basename(file_path), "chunk_id": f"{os.path.basename(file_path)}_{chunk_id_counter}", "text": chunk_text, "offset": start}
                    file_chunk_count += 1; total_chunks_generated += 1
                    start += max(1, self.chunk_size - self.overlap); chunk_id_counter += 1
            else: print(f"Skipping empty/unreadable file: {file_path}")
        print(f"Total chunks generated across all files: {total_chunks_generated}")

class Contextualizer:
    def __init__(self, default_emotion: Dict = None, default_target: Dict = None):
        self.default_emotion = default_emotion.copy() if default_emotion else {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}
        self.default_target = default_target.copy() if default_target else {"target_category": None, "expected_jump_freq": "mittel"}
    def add_context(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        context = {"emotion": self.default_emotion.copy(), "target": self.default_target.copy()}
        file_id_lower = chunk_data.get('file_id', '').lower()
        if "ethik" in file_id_lower or "ethics" in file_id_lower: context['target']['target_category'] = "Ethik"; context['emotion'] = {"pleasure": 0.1, "arousal": 0.3, "dominance": 0.2}
        elif "technik" in file_id_lower or "tech" in file_id_lower: context['target']['target_category'] = "Technologie"; context['emotion'] = {"pleasure": 0.2, "arousal": 0.5, "dominance": 0.4}
        elif "philosophie" in file_id_lower or "philosophy" in file_id_lower: context['target']['target_category'] = "Philosophie"; context['emotion'] = {"pleasure": 0.3, "arousal": 0.6, "dominance": 0.1}
        chunk_data['context'] = context
        return chunk_data

# ########################################################################
# # 5. Zustandsextraktion & Embedding (Q-LLM spezifisch)
# ########################################################################
# StateExtractor und StateEmbedder bleiben unverändert
class StateExtractor:
    def __init__(self, network_nodes: List[Node]):
        self.nodes = network_nodes; self.node_map = {n.label: n for n in network_nodes if hasattr(n, 'label')}
    def extract_current_state(self) -> Dict[str, Any]:
        activations = {n.label: float(getattr(n, 'activation', 0.0)) for n in self.nodes if hasattr(n, 'label')}
        module_states = {n.label: n.get_state_representation() for n in self.nodes if hasattr(n, 'label') and not isinstance(n, MemoryNode) and not isinstance(n, ValueNode)}
        emotion = CURRENT_EMOTION_STATE.copy()
        jump_summary = {n.label: n.analyze_jumps(n.last_measurement_log) for n in self.nodes if hasattr(n,'label') and n.is_quantum and hasattr(n, 'last_measurement_log')}
        return {"timestamp": time.time(), "activations": activations, "module_specific_states": module_states, "current_emotion": emotion, "jump_summary": jump_summary, "nodes_map": self.node_map}

class StateEmbedder:
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = max(1, embedding_dim)
    def embed_state(self, network_state: Dict[str, Any]) -> Optional[np.ndarray]:
        features = []
        activations = network_state.get("activations", {}); emotion = network_state.get("current_emotion", {}); jumps = network_state.get("jump_summary", {})
        if isinstance(activations, dict): sorted_labels = sorted(activations.keys()); features.extend([float(activations[label]) for label in sorted_labels if np.isfinite(activations[label])])
        if isinstance(emotion, dict): features.extend([float(emotion.get("pleasure", 0)), float(emotion.get("arousal", 0)), float(emotion.get("dominance", 0))])
        if isinstance(jumps, dict):
             sorted_jump_labels = sorted(jumps.keys())
             for label in sorted_jump_labels:
                 jump_info = jumps[label]
                 if isinstance(jump_info, dict): features.append(float(jump_info.get("max_jump", 0))); features.append(float(jump_info.get("avg_jump", 0)))
        if not features: return np.zeros(self.embedding_dim)
        feature_vector = np.array(features, dtype=float); hashed_vector = np.zeros(self.embedding_dim)
        safe_indices = np.abs(np.int64(np.nan_to_num(feature_vector) * 1000)) % self.embedding_dim
        safe_magnitudes = np.tanh(np.nan_to_num(feature_vector))
        np.add.at(hashed_vector, safe_indices[:len(safe_magnitudes)], safe_magnitudes)
        norm = np.linalg.norm(hashed_vector)
        return hashed_vector / norm if norm > 1e-9 else hashed_vector


# ########################################################################
# # 6. Trainings-Loop & Modell-Management (Q-LLM spezifisch)
# ########################################################################

class QuantumAronaModel:
    """Kapselt das gesamte Netzwerk und seine Zustände."""
    def __init__(self, config: Dict):
        self.config = config
        Node.DEFAULT_NUM_QUBITS = config.get("num_qubits_per_node", 4)
        Node.DEFAULT_ACTIVATION_HISTORY_LEN = config.get("activation_history_len", 50) # Configurable history
        self.num_qubits_per_node = Node.DEFAULT_NUM_QUBITS
        self.nodes: List[Node] = []
        self.node_map: Dict[str, Node] = {}
        self.global_emotion_state = INITIAL_EMOTION_STATE.copy()
        global CURRENT_EMOTION_STATE; CURRENT_EMOTION_STATE = self.global_emotion_state.copy()
        self.history: Dict[str, deque] = {}
        self._initialize_network(config.get("network_structure", {}))
    def _initialize_network(self, structure_config: Dict):
        print("Initializing Quantum Arona Network...")
        node_configs = structure_config.get("nodes", [])
        connection_configs = structure_config.get("connections", [])
        if not node_configs: print("WARNUNG: Keine Knoten in der Konfiguration definiert!")
        created_nodes = {}
        for node_conf in node_configs:
            label = node_conf.get("label"); node_class_name = node_conf.get("class", "Node")
            if not label: print("WARNUNG: Überspringe Knoten ohne Label."); continue
            try:
                node_class = globals().get(node_class_name, Node)
                if not issubclass(node_class, Node): node_class = Node
                params = {k: v for k, v in node_conf.items() if k not in ['label', 'class']}
                if 'num_qubits' not in params and node_class != ValueNode: params['num_qubits'] = self.num_qubits_per_node
                node_instance = node_class(label=label, **params)
                self.nodes.append(node_instance); created_nodes[label] = node_instance
            except Exception as e: print(f"ERROR creating node {label}: {e}"); traceback.print_exc()
        self.node_map = created_nodes; print(f"Created {len(self.nodes)} nodes.")
        connections_added = 0
        for conn_conf in connection_configs:
            source_label = conn_conf.get("source"); target_label = conn_conf.get("target"); weight = conn_conf.get("weight")
            if source_label in self.node_map and target_label in self.node_map:
                try: self.node_map[source_label].add_connection(self.node_map[target_label], weight=weight); connections_added += 1
                except Exception as e: print(f"ERROR adding connection {source_label} -> {target_label}: {e}")
            else:
                 if source_label not in self.node_map: print(f"Warning: Skipping connection from unknown source '{source_label}'.")
                 if target_label not in self.node_map: print(f"Warning: Skipping connection to unknown target '{target_label}'.")
        print(f"Added {connections_added} connections.")
    def calculate_classic_input_sum(self, emotion_factors: Dict[str, float]):
        for node in self.nodes: node.activation_sum = 0.0
        signal_modulation = emotion_factors.get("signal_modulation", 1.0)
        for source_node in self.nodes:
            current_activation = float(source_node.activation) if np.isfinite(source_node.activation) else 0.0
            if not hasattr(source_node, 'connections') or current_activation < 0.01: continue
            is_inhibitory = getattr(source_node, 'neuron_type', '') == "inhibitory"
            base_signal = current_activation * signal_modulation
            for connection in source_node.connections:
                target_node = connection.target_node
                if not target_node or not hasattr(target_node, 'activation_sum'): continue
                conn_weight = float(connection.weight)
                signal_strength = base_signal * conn_weight * (-1.5 if is_inhibitory else 1.0)
                current_target_sum = float(target_node.activation_sum) if np.isfinite(target_node.activation_sum) else 0.0
                target_node.activation_sum = float(np.clip(current_target_sum + signal_strength, -50.0, 50.0))
    def step(self, input_chunk: Optional[str] = None, context: Optional[Dict] = None, n_shots: int = 5):
        global CURRENT_EMOTION_STATE
        if context:
            emotion_context = context.get('emotion')
            if emotion_context and isinstance(emotion_context, dict):
                for dim, value in emotion_context.items():
                    if dim in self.global_emotion_state: self.global_emotion_state[dim] = float(np.clip(value, -1.0, 1.0))
                limbus = self.node_map.get("Limbus Affektus");
                if isinstance(limbus, LimbusAffektus): limbus.emotion_state = self.global_emotion_state.copy()
                CURRENT_EMOTION_STATE = self.global_emotion_state.copy()
            target_values = context.get('target_values', {})
            if isinstance(target_values, dict):
                for label, target_val in target_values.items():
                    if label in self.node_map and isinstance(self.node_map[label], ValueNode):
                        self.node_map[label].activation = float(np.clip(target_val, 0.0, 1.0))
            if input_chunk: self.apply_text_input(input_chunk)
        limbus = self.node_map.get("Limbus Affektus"); emotion_factors = limbus.get_emotion_influence_factors() if isinstance(limbus, LimbusAffektus) else {}
        self.calculate_classic_input_sum(emotion_factors)
        for node in self.nodes: node.calculate_activation(n_shots=n_shots) # Activation history is filled here
        module_outputs: Dict[str, deque] = {}
        if isinstance(limbus, LimbusAffektus):
            new_emotion_state = limbus.update_emotion_state(self.nodes, module_outputs)
            self.global_emotion_state = new_emotion_state; CURRENT_EMOTION_STATE = new_emotion_state
            module_outputs.setdefault("Limbus Affektus", deque(maxlen=10)).append(self.global_emotion_state)
        meta_cog = self.node_map.get("Meta Cognitio")
        if isinstance(meta_cog, MetaCognitio): module_outputs.setdefault("Meta Cognitio", deque(maxlen=10)).append(meta_cog.get_meta_cognitive_state())
        critic = self.node_map.get("Cortex Criticus")
        if isinstance(critic, CortexCriticus):
            items_to_eval = []; evaluations = critic.evaluate_items(items_to_eval, self.nodes, 1.0)
            module_outputs.setdefault("Cortex Criticus", deque(maxlen=10)).append(evaluations)
        extractor = StateExtractor(self.nodes); current_network_state = extractor.extract_current_state()
        return current_network_state
    def apply_text_input(self, text_chunk: str):
        try:
            words = set(text_chunk.lower().split())
            for node in self.nodes:
                if isinstance(node, MemoryNode) and hasattr(node, 'label'):
                    node_label_lower = node.label.lower()
                    if node_label_lower in words or f"{node_label_lower}s" in words:
                        activation_sum_f = float(node.activation_sum) if isinstance(node.activation_sum, (float, np.number)) and np.isfinite(node.activation_sum) else 0.0
                        node.activation_sum = activation_sum_f + 1.5 # Boost input
        except Exception as e: print(f"WARNUNG: Fehler in apply_text_input: {e}")
    def apply_updates(self, updates: Dict[str, Any]):
        weight_updates = updates.get("weight_updates", {}); q_param_updates = updates.get("q_param_updates", {})
        if isinstance(weight_updates, dict):
            for (source_label, target_label), delta_w in weight_updates.items():
                if source_label in self.node_map:
                    source_node = self.node_map[source_label]
                    if hasattr(source_node, 'connections'):
                        for conn in source_node.connections:
                            if conn.target_node and getattr(conn.target_node, 'label', None) == target_label: update_classical_weight(conn, float(delta_w)); break
        if isinstance(q_param_updates, dict):
            for node_label, delta_q in q_param_updates.items():
                if node_label in self.node_map and isinstance(delta_q, np.ndarray): update_quantum_params(self.node_map[node_label], delta_q)
    # *** NEUE METHODE ***
    def apply_hebbian_learning(self, lr_classical: float, lr_quantum: float):
        """Wendet die quantum-modulierte Hebb'sche Lernregel auf alle Verbindungen an."""
        for node in self.nodes:
            if hasattr(node, 'connections'):
                for conn in node.connections:
                    hebbian_learning_quantum_node_smoothed(
                        node_a=node, connection=conn,
                        learning_rate_classical=lr_classical, learning_rate_quantum=lr_quantum,
                        history_window=self.config.get("hebb_history_window", 3),
                        activation_threshold_high=self.config.get("hebb_threshold_high", 0.55),
                        activation_threshold_low=self.config.get("hebb_threshold_low", 0.30),
                        reg_factor=self.config.get("hebb_reg_factor", 0.001)
                    )
    def get_state(self) -> Dict[str, Any]:
        nodes_state, connections_state = [], []
        for node in self.nodes:
            try:
                n_state = {"label": node.label, "class": type(node).__name__, "activation": float(node.activation) if np.isfinite(node.activation) else 0.0, "neuron_type": node.neuron_type, "is_quantum": node.is_quantum, "num_qubits": node.num_qubits}
                if node.is_quantum and node.q_system: n_state["q_params"] = [float(p) for p in node.q_system.get_params()]
                nodes_state.append(n_state)
                if hasattr(node, 'connections'):
                    for conn in node.connections:
                        if conn.target_node and hasattr(conn.target_node, 'label'): connections_state.append({"source": node.label, "target": conn.target_node.label, "weight": float(conn.weight) if np.isfinite(conn.weight) else 0.0})
            except Exception as e: print(f"ERROR getting state for node {getattr(node, 'label', 'UNKNOWN')}: {e}")
        return {"version": "quantum_arona_v1_checkpoint", "config": self.config, "nodes": nodes_state, "connections": connections_state, "emotion_state": self.global_emotion_state}
    def load_state(self, state_data: Dict[str, Any]):
        print("Loading model state...")
        if state_data.get("version") != "quantum_arona_v1_checkpoint": print(f"Warning: Checkpoint version mismatch.")
        loaded_nodes = {n['label']: n for n in state_data.get('nodes', []) if 'label' in n}
        loaded_connections = {(c['source'], c['target']): c.get('weight', 0.0) for c in state_data.get('connections', []) if 'source' in c and 'target' in c}
        self.config = state_data.get("config", self.config)
        Node.DEFAULT_NUM_QUBITS = self.config.get("num_qubits_per_node", 4)
        self.num_qubits_per_node = Node.DEFAULT_NUM_QUBITS
        Node.DEFAULT_ACTIVATION_HISTORY_LEN = self.config.get("activation_history_len", 50)
        for node_label, node in self.node_map.items():
            if node_label in loaded_nodes:
                n_data = loaded_nodes[node_label]
                node.activation = float(n_data.get('activation', 0.0)) if np.isfinite(n_data.get('activation', 0.0)) else 0.0
                if hasattr(node, 'activation_history'): node.activation_history = deque(maxlen=n_data.get('activation_history_len', Node.DEFAULT_ACTIVATION_HISTORY_LEN))
                if node.is_quantum and node.q_system and 'q_params' in n_data:
                    try:
                        params_list = n_data['q_params']
                        if params_list and isinstance(params_list, list):
                             if len(params_list) == node.q_system.num_params: node.q_system.set_params(np.array(params_list, dtype=float))
                             else: print(f"WARNUNG: QParams Anzahl mismatch für {node.label}.")
                        elif params_list is not None: print(f"WARNUNG: q_params für {node.label} keine Liste.")
                    except Exception as e: print(f"Error loading QParams for {node.label}: {e}")
                if hasattr(node, 'connections'):
                     for conn in node.connections:
                         if conn.target_node and hasattr(conn.target_node, 'label'):
                             conn_key = (node.label, conn.target_node.label)
                             if conn_key in loaded_connections: conn.weight = float(loaded_connections[conn_key]) if np.isfinite(loaded_connections[conn_key]) else 0.0
        self.global_emotion_state = state_data.get('emotion_state', INITIAL_EMOTION_STATE.copy())
        limbus = self.node_map.get("Limbus Affektus");
        if isinstance(limbus, LimbusAffektus): limbus.emotion_state = self.global_emotion_state.copy()
        global CURRENT_EMOTION_STATE; CURRENT_EMOTION_STATE = self.global_emotion_state.copy()
        print("Model state loaded.")


class QuantumTrainer:
    """
    Orchestriert den Trainingsprozess für Quantum-Arona.
    Implementiert sprung-induktives Lernen, Hebb'sches Lernen und Peak-Loss-Persistence.
    """
    def __init__(self, model: QuantumAronaModel, dataset_loader: DatasetLoader,
                 contextualizer: Contextualizer, state_embedder: StateEmbedder,
                 config: Dict, persistence_manager: Optional['PersistenceManager'] = None):
        self.model = model
        self.loader = dataset_loader
        self.contextualizer = contextualizer
        self.embedder = state_embedder
        self.config = config
        self.persistence_manager = persistence_manager # Persistence Manager Instanz

        # Basiskonfiguration holen
        self.base_lr_classical = config.get("learning_rate_classical", DEFAULT_CLASSICAL_LR)
        self.base_lr_quantum = config.get("learning_rate_quantum", DEFAULT_QUANTUM_LR)
        self.base_n_shots = config.get("simulation_shots", 5)
        if self.base_n_shots <= 0: print("WARNUNG: simulation_shots <= 0. Setze auf 1."); self.base_n_shots = 1
        self.epochs = config.get("training_epochs", 1)

        # Konfiguration für dynamische Anpassungen & Sprünge
        self.enable_dynamic_shots = config.get("enable_dynamic_shots", True)
        self.stagnation_threshold_loss = config.get("stagnation_threshold_loss", 0.005) # Wenn Loss sich weniger ändert
        self.shots_increase_factor = config.get("shots_increase_factor", 1.5)
        self.shots_decrease_factor = config.get("shots_decrease_factor", 0.95) # Faktor zur Reduzierung (weniger aggressiv)
        self.max_shots = config.get("max_simulation_shots", 50)
        self.min_shots = config.get("min_simulation_shots", 3) # Mindestanzahl Shots
        self.jump_rate_threshold_for_shot_decrease = config.get("jump_rate_threshold", 0.1) # Wenn >10% Chunks Sprünge hatten
        self.loss_improvement_threshold_for_jump = config.get("loss_improvement_threshold_for_jump", -0.01) # Wenn Loss um mehr als 1% sinkt nach Sprung
        self.feedback_lr_scaling = config.get("feedback_lr_scaling_factor", 0.5)

        # Konfiguration für experimentelle Strategien
        self.enable_perturbation = config.get("enable_perturbation", True)             # Strategie 1
        self.perturbation_std_dev = config.get("perturbation_std_dev", 0.03)         # Strategie 1
        self.randomize_shots = config.get("randomize_shots", True)                   # Strategie 2
        self.shot_random_min = config.get("shot_random_min", -1)                     # Strategie 2
        self.shot_random_max = config.get("shot_random_max", 2)                      # Strategie 2
        self.enable_jump_boost = config.get("enable_jump_boost", True)               # Strategie 3 + Bonus
        # Passe jump_threshold_high an die Qubit-Zahl an
        self.jump_threshold_high = config.get("jump_threshold_high", (2**self.model.num_qubits_per_node) / 2) # Bonus
        self.jump_boost_lr_factor_high = config.get("jump_boost_lr_factor_high", 1.4) # Bonus
        self.jump_boost_lr_factor_low = config.get("jump_boost_lr_factor_low", 1.1)   # Strategie 3
        self.jump_lr_dampen_factor = config.get("jump_lr_dampen_factor", 0.98)        # Bonus (Dämpfung)
        self.enable_variance_trigger = config.get("enable_variance_trigger", True)   # Strategie 5
        self.variance_threshold_low = config.get("variance_threshold_low", 0.5)      # Strategie 5
        self.use_hebbian_learning = config.get("use_hebbian_learning", True) # Option Hebb ein/aus
        self.hebb_history_window = config.get("hebb_history_window", 3)      # Hebb Glättung
        self.hebb_threshold_high = config.get("hebb_threshold_high", 0.55)   # Hebb Glättung
        self.hebb_threshold_low = config.get("hebb_threshold_low", 0.30)     # Hebb Glättung
        self.hebb_reg_factor = config.get("hebb_reg_factor", 0.001)          # Hebb Glättung

        # *** HIER: Fehlende Initialisierung für Peak Loss Persistence ***
        self.peak_loss_tracking_enabled = config.get("peak_loss_tracking_enabled", True) # Strategie aktivieren?
        self.highest_loss_recorded = float('-inf') # Höchster Loss bisher
        self.loss_at_last_peak = float('-inf')     # Loss-Wert, der die letzte Persistenz ausgelöst hat
        self.persisting_after_peak = False         # Flag: Sind wir im Halte-Modus?
        # *** Ende fehlende Initialisierung ***

        # Zustandsvariablen (bereits vorhanden)
        self.current_n_shots = self.base_n_shots
        self.last_epoch_avg_loss: Optional[float] = None
        self.current_step_in_epoch = 0
        self.last_chunk_loss: Optional[float] = None
        self.jumps_in_epoch_count = 0

    # _get_target_state und _calculate_feedback bleiben unverändert
    def _get_target_state(self, context: Dict) -> Dict[str, Any]:
        target_state = {"activations": {}, "emotion": None, "jump_profile": None}
        target_config = context.get("target", {}); target_category = target_config.get("target_category")
        if target_category and isinstance(target_category, str): target_state["activations"][target_category] = 0.95
        target_emotion = context.get("emotion")
        if target_emotion and isinstance(target_emotion, dict):
             valid_emotion = {dim: float(np.clip(target_emotion.get(dim, 0.0), -1.0, 1.0)) if np.isfinite(target_emotion.get(dim, 0.0)) else 0.0 for dim in EMOTION_DIMENSIONS}
             target_state["emotion"] = valid_emotion
        else: target_state["emotion"] = self.contextualizer.default_emotion.copy()
        target_jump_freq = target_config.get("expected_jump_freq")
        if target_jump_freq in ["niedrig", "mittel", "hoch"]: target_state["jump_profile"] = {"frequency": target_jump_freq}
        return target_state

    def _calculate_feedback(self, current_state: Dict[str, Any], target_state: Dict[str, Any],
                            weight_activation: float = 0.6, weight_emotion: float = 0.4
                           ) -> Tuple[float, Dict[str, float]]:
        if not target_state: return 0.0, {}
        feedback_components_01 = {}; target_activations = target_state.get("activations", {}); current_activations = current_state.get("activations", {})
        activation_component_feedback = 0.0
        if target_activations and current_activations and isinstance(target_activations, dict) and isinstance(current_activations, dict):
            cat_errors = []
            for target_cat, target_act_val in target_activations.items():
                current_act_val = current_activations.get(target_cat, 0.0); current_act_f = float(current_act_val) if np.isfinite(current_act_val) else 0.0
                target_act_f = float(target_act_val) if np.isfinite(target_act_val) else 1.0; cat_errors.append(abs(target_act_f - current_act_f))
            if cat_errors: activation_component_feedback = 1.0 - np.mean(cat_errors)
        feedback_components_01["activation"] = activation_component_feedback
        target_emotion = target_state.get("emotion"); current_emotion = current_state.get("current_emotion"); emotion_component_feedback = 0.0
        if target_emotion and current_emotion and isinstance(target_emotion, dict) and isinstance(current_emotion, dict):
            error_dist_sq, dims_compared = 0.0, 0
            for dim in EMOTION_DIMENSIONS:
                 target_val_f = float(target_emotion.get(dim, 0.0)) if np.isfinite(target_emotion.get(dim, 0.0)) else 0.0
                 current_val_f = float(current_emotion.get(dim, 0.0)) if np.isfinite(current_emotion.get(dim, 0.0)) else 0.0
                 error_dist_sq += (target_val_f - current_val_f)**2; dims_compared += 1
            if dims_compared > 0:
                 max_possible_error_sq = dims_compared * 4.0; normalized_error = np.sqrt(error_dist_sq / max_possible_error_sq) if max_possible_error_sq > 0 else 0
                 emotion_component_feedback = 1.0 - normalized_error
        feedback_components_01["emotion"] = emotion_component_feedback
        if not feedback_components_01: return 0.0, {}
        total_weight = weight_activation + weight_emotion;
        if total_weight <= 1e-6 : return 0.0, feedback_components_01
        norm_w_act = weight_activation / total_weight; norm_w_emo = weight_emotion / total_weight
        weighted_feedback_01 = (norm_w_act * feedback_components_01.get("activation", 0.0) + norm_w_emo * feedback_components_01.get("emotion", 0.0))
        final_feedback_signal = weighted_feedback_01 * 2.0 - 1.0
        return float(np.clip(final_feedback_signal, -1.0, 1.0)), feedback_components_01

    def train_epoch(self, epoch_num: int):
        """
        Führt eine Trainingsepoche durch, mit experimentellen Strategien.
        """
        # HIER WIRD self.persisting_after_peak verwendet, muss also in __init__ definiert sein!
        print(f"\n--- Starting Training Epoch {epoch_num}/{self.epochs} (Current Shots: {self.current_n_shots}, Persisting: {self.persisting_after_peak}) ---")
        chunk_generator = self.loader.generate_chunks()
        total_loss = 0.0; processed_chunks = 0
        self.jumps_in_epoch_count = 0
        self.last_chunk_loss = None
        epoch_state_variances = []
        iterator = chunk_generator

        if TQDM_AVAILABLE:
            try: iterator = tqdm(chunk_generator, desc=f"Epoch {epoch_num} Shots={self.current_n_shots}", unit="chunk", leave=False)
            except Exception as e: print(f"Warnung: Fehler beim Initialisieren von tqdm: {e}")

        for i, chunk_data in enumerate(iterator):
            self.current_step_in_epoch = i
            current_state, loss, overall_feedback, feedback_comps_01, loss_delta, any_jump_detected, jump_details = None, float('nan'), 0.0, {}, None, False, []
            effective_shots = self.current_n_shots # Startwert für Shots in diesem Chunk

            try:
                # Strategie 2: Randomisierte Shot-Zahl
                if self.randomize_shots and not self.persisting_after_peak:
                    shot_noise = np.random.randint(self.shot_random_min, self.shot_random_max + 1)
                    effective_shots = int(np.clip(self.current_n_shots + shot_noise, self.min_shots, self.max_shots))

                contextualized_chunk = self.contextualizer.add_context(chunk_data)
                context = contextualized_chunk.get("context", {})
                text_input = contextualized_chunk.get("text")

                current_state = self.model.step(input_chunk=text_input, context=context, n_shots=effective_shots)

                target_state = self._get_target_state(context)
                overall_feedback, feedback_comps_01 = self._calculate_feedback(current_state, target_state)
                loss = 1.0 - (overall_feedback + 1.0) / 2.0

                chunk_max_jump = 0; chunk_variances = []
                jump_summary = current_state.get("jump_summary", {})
                if isinstance(jump_summary, dict):
                    for node_label, jump_info in jump_summary.items():
                         if isinstance(jump_info, dict):
                             if jump_info.get("jump_detected", False): any_jump_detected = True
                             chunk_max_jump = max(chunk_max_jump, jump_info.get("max_jump", 0))
                             chunk_variances.append(jump_info.get("state_variance", 0.0))
                if any_jump_detected: self.jumps_in_epoch_count += 1
                if chunk_variances: epoch_state_variances.extend(chunk_variances)

                if np.isfinite(loss) and self.last_chunk_loss is not None and np.isfinite(self.last_chunk_loss): loss_delta = loss - self.last_chunk_loss
                if np.isfinite(loss): self.last_chunk_loss = loss; total_loss += loss; processed_chunks += 1
                else: print(f"WARNUNG: Ungültiger Loss ({loss}) Chunk {chunk_data.get('chunk_id')}. Überspringe Update."); continue

                meta_cog = self.model.node_map.get("Meta Cognitio")
                meta_state = meta_cog.get_meta_cognitive_state() if isinstance(meta_cog, MetaCognitio) else {}
                dyn_lr_c, dyn_lr_q = calculate_dynamic_learning_rates(self.base_lr_classical, self.base_lr_quantum, self.model.global_emotion_state, meta_state, overall_feedback, self.feedback_lr_scaling)

                # Strategie 3 + Bonus: Sprungbasierter LR Boost/Dämpfung
                boosted_lr_q = dyn_lr_q
                if self.enable_jump_boost:
                    boost_factor = 1.0
                    if chunk_max_jump > self.jump_threshold_high: boost_factor = self.jump_boost_lr_factor_high
                    elif chunk_max_jump > 0: boost_factor = self.jump_boost_lr_factor_low
                    else: boost_factor = self.jump_lr_dampen_factor
                    boosted_lr_q = float(np.clip(dyn_lr_q * boost_factor, 0.00005, 0.1))

                # Entscheidung und Anwendung der Lernregel
                if self.use_hebbian_learning:
                    self.model.apply_hebbian_learning(dyn_lr_c, boosted_lr_q)
                else:
                    updates = calculate_parameter_updates(current_state, target_state, feedback_comps_01, overall_feedback, dyn_lr_c, boosted_lr_q)
                    self.model.apply_updates(updates)

                if isinstance(meta_cog, MetaCognitio) and any_jump_detected and loss_delta is not None and loss_delta < self.loss_improvement_threshold_for_jump:
                    meta_cog.log_reflection(f"Jump correlated with loss improvement.", f"{epoch_num}-{processed_chunks}", data={"jump_nodes": jump_details, "loss_delta": round(loss_delta, 5)})

                if self.persistence_manager and processed_chunks % self.config.get("log_interval_chunks", 100) == 0 :
                    state_embedding = self.embedder.embed_state(current_state)
                    log_meta = {"context": context, "feedback_comps": feedback_comps_01, "loss_delta": loss_delta, "jump_details": jump_details if any_jump_detected else None, "learning_rates": {"classical": dyn_lr_c, "quantum": boosted_lr_q}}
                    self.persistence_manager.log_chunk_result(epoch_num, chunk_data, loss, overall_feedback, loss_delta, any_jump_detected, jump_details, state_embedding, meta_data=log_meta)

                if TQDM_AVAILABLE and isinstance(iterator, tqdm) and processed_chunks > 0 and processed_chunks % 50 == 0:
                   iterator.set_postfix({"avg_loss": f"{total_loss/processed_chunks:.4f}", "shots": f"{effective_shots}({self.current_n_shots})", "lr_q": f"{boosted_lr_q:.5f}", "fb": f"{overall_feedback:.2f}", "jmp": any_jump_detected})

            except Exception as e:
                print(f"\nERROR processing chunk {chunk_data.get('chunk_id', 'N/A')}: {e}"); traceback.print_exc(); continue

        # --- Ende der Chunk-Schleife ---
        if TQDM_AVAILABLE and isinstance(iterator, tqdm): iterator.close()

        avg_epoch_loss = total_loss / processed_chunks if processed_chunks > 0 else float('nan')
        jump_rate = self.jumps_in_epoch_count / processed_chunks if processed_chunks > 0 else 0.0
        avg_epoch_variance = float(np.mean(epoch_state_variances)) if epoch_state_variances else 0.0
        print(f"--- Epoch {epoch_num} finished. Avg Loss: {avg_epoch_loss:.5f} ({processed_chunks} chunks). Jump Rate: {jump_rate:.2%}. Avg State Var: {avg_epoch_variance:.3f} ---")

        if self.persistence_manager:
            self.persistence_manager.log_epoch_summary(epoch_num, avg_epoch_loss, processed_chunks, jump_rate)

        # Rückgabe der berechneten Metriken für diese Epoche
        return avg_epoch_loss, jump_rate, avg_epoch_variance

    def train(self):
        """Startet den gesamten Trainingsprozess mit allen Strategien."""
        if not all([self.model, self.loader, self.contextualizer, self.embedder]):
             print("FEHLER: Trainer nicht korrekt initialisiert."); return

        # Lade initialen höchsten Loss, falls Checkpoint geladen wurde
        if self.last_epoch_avg_loss is not None and np.isfinite(self.last_epoch_avg_loss):
             self.highest_loss_recorded = max(self.highest_loss_recorded, self.last_epoch_avg_loss)

        for i in range(self.epochs):
            epoch_num = i + 1
            avg_loss, jump_rate, avg_epoch_variance = self.train_epoch(epoch_num) # Fange Metriken auf

            if avg_loss is None or not np.isfinite(avg_loss):
                print(f"WARNUNG: Epoch {epoch_num}: Ungültiger avg_loss ({avg_loss}). Überspringe Anpassungen.")
                continue

            run_standard_adaptations = True # Standardmäßig laufen Anpassungen
            loss_delta = avg_loss - self.last_epoch_avg_loss if self.last_epoch_avg_loss is not None and np.isfinite(self.last_epoch_avg_loss) else 0.0

            # --- Peak Loss Persistence Logik ---
            if self.peak_loss_tracking_enabled:
                if self.persisting_after_peak:
                    if avg_loss > self.loss_at_last_peak:
                        print(f"INFO: Epoch {epoch_num}: Neuer Loss-Peak ({avg_loss:.5f} > {self.loss_at_last_peak:.5f}) nach Persistenz. Beende Persistenz.")
                        self.persisting_after_peak = False
                        self.highest_loss_recorded = avg_loss
                        self.loss_at_last_peak = avg_loss
                        run_standard_adaptations = True
                    else:
                        print(f"INFO: Epoch {epoch_num}: Persistiere nach Peak ({self.loss_at_last_peak:.5f}). Current Loss {avg_loss:.5f}. Überspringe adaptive Änderungen.")
                        run_standard_adaptations = False
                        # Wichtig: last_epoch_avg_loss trotzdem updaten für nächsten Vergleich!
                        self.last_epoch_avg_loss = avg_loss
                        checkpoint_filename = os.path.join(self.config.get("checkpoint_dir", "."), f"quantum_arona_checkpoint_epoch_{epoch_num}.json")
                        self.save_checkpoint(checkpoint_filename)
                        continue # Gehe direkt zur nächsten Epoche

                elif avg_loss > self.highest_loss_recorded:
                    print(f"INFO: Epoch {epoch_num}: Neuer höchster Loss ({avg_loss:.5f} > {self.highest_loss_recorded:.5f}). Aktiviere Persistenz für nächste Epoche.")
                    self.highest_loss_recorded = avg_loss
                    self.loss_at_last_peak = avg_loss
                    self.persisting_after_peak = True
                    run_standard_adaptations = True
            # --- Ende Peak Loss Persistence Logik ---

            if run_standard_adaptations:
                stagnated = False
                if self.last_epoch_avg_loss is not None and np.isfinite(self.last_epoch_avg_loss):
                    if abs(loss_delta) < self.stagnation_threshold_loss: stagnated = True

                # Strategie 1: Perturbation bei Stagnation
                if self.enable_perturbation and stagnated:
                    print(f"INFO: Epoch {epoch_num}: Stagnation detected. Applying parameter perturbation.")
                    perturb_std = self.perturbation_std_dev; perturbed_nodes = 0
                    for node in self.model.nodes:
                        if node.is_quantum and node.q_system:
                            perturb = np.random.normal(0, perturb_std, size=node.q_system.params.shape)
                            node.q_system.update_internal_params(perturb); perturbed_nodes +=1
                    print(f"INFO: Perturbed {perturbed_nodes} quantum nodes.")

                # Strategie 5: Erhöhe Shots bei niedriger Varianz (nur wenn NICHT stagnierend)
                variance_triggered_shot_increase = False
                if self.enable_variance_trigger and avg_epoch_variance < self.variance_threshold_low and not stagnated:
                    new_shots = min(self.max_shots, int(self.current_n_shots * 1.1))
                    if new_shots > self.current_n_shots:
                        print(f"INFO: Epoch {epoch_num}: Low avg state variance ({avg_epoch_variance:.3f} < {self.variance_threshold_low:.3f}). Increasing shots to {new_shots}.")
                        self.current_n_shots = new_shots
                        variance_triggered_shot_increase = True

                # Dynamische Shot-Anpassung (Stagnation ODER guter Verlauf)
                if self.enable_dynamic_shots and not variance_triggered_shot_increase:
                    if stagnated:
                        new_shots = int(self.current_n_shots * self.shots_increase_factor)
                        new_shots = min(new_shots, self.max_shots)
                        if new_shots > self.current_n_shots:
                            print(f"INFO: Epoch {epoch_num}: Stagnation detected (Loss Delta: {loss_delta:+.5f}). Increasing shots to {new_shots}.")
                            self.current_n_shots = new_shots
                    elif jump_rate <= self.jump_rate_threshold_for_shot_decrease and self.current_n_shots > self.min_shots:
                        new_shots = max(self.min_shots, int(self.current_n_shots * self.shots_decrease_factor))
                        if new_shots < self.current_n_shots:
                            print(f"INFO: Epoch {epoch_num}: Stable, low jump rate. Reducing shots towards minimum: {new_shots}.")
                            self.current_n_shots = new_shots

            # Speichere aktuellen Loss für nächsten Vergleich (immer nötig)
            if np.isfinite(avg_loss):
                 self.last_epoch_avg_loss = avg_loss

            # Checkpoint speichern (immer am Ende der Epoche)
            checkpoint_filename = os.path.join(self.config.get("checkpoint_dir", "."), f"quantum_arona_checkpoint_epoch_{epoch_num}.json")
            self.save_checkpoint(checkpoint_filename) # Speichere auch den Persistenz-Status

        print("Training abgeschlossen.")

    def save_checkpoint(self, filename: str):
        """Speichert den Checkpoint inklusive Persistenz-Status."""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            state_data = self.model.get_state()
            # Füge Trainer-Zustand hinzu
            state_data["trainer_state"] = {
                "current_n_shots": self.current_n_shots,
                "last_epoch_avg_loss": self.last_epoch_avg_loss,
                # --- NEU: Persistenz-Status speichern ---
                "highest_loss_recorded": self.highest_loss_recorded,
                "loss_at_last_peak": self.loss_at_last_peak,
                "persisting_after_peak": self.persisting_after_peak
            }
            with open(filename, 'w', encoding='utf-8') as f: json.dump(state_data, f, indent=2, ensure_ascii=False)
        except Exception as e: print(f"ERROR saving checkpoint {filename}: {e}"); traceback.print_exc()

    def load_checkpoint(self, filename: str):
        """Lädt den Checkpoint inklusive Persistenz-Status."""
        if not os.path.exists(filename): print(f"ERROR: Checkpoint file not found: {filename}"); return False
        try:
            with open(filename, 'r', encoding='utf-8') as f: state_data = json.load(f)
            self.config = state_data.get("config", self.config)
            # Wende geladene Konfig auf Node-Defaults an
            Node.DEFAULT_NUM_QUBITS = self.config.get("num_qubits_per_node", 4)
            Node.DEFAULT_ACTIVATION_HISTORY_LEN = self.config.get("activation_history_len", 50)
            self.model.num_qubits_per_node = Node.DEFAULT_NUM_QUBITS # Modell informieren
            # Lade Modellzustand
            self.model.load_state(state_data)
            # Lade Trainer-Zustand
            trainer_state = state_data.get("trainer_state", {})
            self.current_n_shots = trainer_state.get("current_n_shots", self.base_n_shots)
            self.last_epoch_avg_loss = trainer_state.get("last_epoch_avg_loss", None)
            # --- NEU: Persistenz-Status laden ---
            self.highest_loss_recorded = trainer_state.get("highest_loss_recorded", float('-inf'))
            self.loss_at_last_peak = trainer_state.get("loss_at_last_peak", float('-inf'))
            self.persisting_after_peak = trainer_state.get("persisting_after_peak", False)
            # ---
            # Lade relevante Configs neu (wichtig, falls sie sich im Checkpoint anders sind)
            self.use_hebbian_learning = self.config.get("use_hebbian_learning", True)
            self.hebb_history_window = self.config.get("hebb_history_window", 3)
            self.min_shots = self.config.get("min_simulation_shots", 3)
            self.max_shots = self.config.get("max_simulation_shots", 50)
            # ... (andere Parameter ggf. auch neu laden, falls sie sich ändern könnten) ...

            print(f"Checkpoint loaded: {filename} (Loaded Shots: {self.current_n_shots}, Last Loss: {self.last_epoch_avg_loss}, Persisting: {self.persisting_after_peak}, HighestLoss: {self.highest_loss_recorded})")
            return True
        except Exception as e: print(f"ERROR loading checkpoint {filename}: {e}"); traceback.print_exc(); return False

# ########################################################################
# # 7. Persistence Manager (Adaptiert für Training)
# ########################################################################
class PersistenceManager:
    """Verwaltet die Speicherung von Logs und Checkpoints (Basis-Implementierung)."""
    def __init__(self, db_path: str):
        self.db_path = db_path; self.conn: Optional[sqlite3.Connection] = None; self.cursor: Optional[sqlite3.Cursor] = None
        self._initialize_db()
    def _get_connection(self) -> sqlite3.Cursor:
        if self.conn is None or self.cursor is None:
            try:
                db_dir = os.path.dirname(self.db_path)
                if db_dir and not os.path.exists(db_dir): os.makedirs(db_dir, exist_ok=True)
                self.conn = sqlite3.connect(self.db_path, timeout=15, check_same_thread=False)
                self.conn.execute("PRAGMA journal_mode=WAL;"); self.conn.execute("PRAGMA synchronous=NORMAL;")
                self.cursor = self.conn.cursor()
            except sqlite3.Error as e: print(f"FATAL DB ERROR: Could not connect to {self.db_path}: {e}"); raise
        return self.cursor
    def _initialize_db(self):
        try:
            cursor = self._get_connection()
            cursor.execute('''CREATE TABLE IF NOT EXISTS training_log (
                                log_id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL DEFAULT CURRENT_TIMESTAMP,
                                epoch INTEGER, chunk_id TEXT, file_id TEXT,
                                loss REAL, feedback REAL, loss_delta REAL, jump_detected INTEGER, jump_details TEXT,
                                state_embedding_json TEXT, meta_data_json TEXT
                            )''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS epoch_summary (
                                epoch INTEGER PRIMARY KEY, timestamp REAL DEFAULT CURRENT_TIMESTAMP,
                                avg_loss REAL, num_chunks INTEGER, jump_rate REAL, notes TEXT
                            )''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_log_epoch_chunk ON training_log(epoch, chunk_id)')
            self.conn.commit(); print(f"Training database '{self.db_path}' initialized/checked.")
        except sqlite3.Error as e: print(f"ERROR initializing training database: {e}"); self.conn.rollback()
        finally: self.close()
    def log_chunk_result(self, epoch: int, chunk_data: Dict, loss: float, feedback: float,
                         loss_delta: Optional[float], jump_detected: bool, jump_details: Optional[List[str]],
                         state_embedding: Optional[np.ndarray], meta_data: Optional[Dict] = None):
        try: embedding_json = json.dumps(state_embedding.tolist()) if state_embedding is not None else None
        except Exception as e: embedding_json = None
        try: meta_json = json.dumps(meta_data) if meta_data else None
        except Exception as e: meta_json = None
        try: jump_details_str = ", ".join(jump_details) if jump_details else None
        except Exception: jump_details_str = None
        try:
            cursor = self._get_connection()
            cursor.execute('''INSERT INTO training_log
                              (epoch, file_id, chunk_id, loss, feedback, loss_delta, jump_detected, jump_details, state_embedding_json, meta_data_json)
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                           (epoch, chunk_data.get('file_id'), chunk_data.get('chunk_id'),
                            float(loss) if np.isfinite(loss) else None, float(feedback) if np.isfinite(feedback) else None,
                            float(loss_delta) if loss_delta is not None and np.isfinite(loss_delta) else None,
                            1 if jump_detected else 0, jump_details_str, embedding_json, meta_json))
            # Commit seltener oder am Ende? Hier: alle 100 Chunks
            if self.conn and chunk_data.get('chunk_id','').endswith('_99'): self.conn.commit()
        except sqlite3.Error as e: print(f"ERROR logging chunk DB: {e}")
        except Exception as e: print(f"ERROR logging chunk non-DB: {e}")
    def log_epoch_summary(self, epoch: int, avg_loss: Optional[float], num_chunks: int, jump_rate: float, notes: str = ""):
         try:
             cursor = self._get_connection(); loss_to_store = float(avg_loss) if avg_loss is not None and np.isfinite(avg_loss) else None
             cursor.execute('''INSERT OR REPLACE INTO epoch_summary
                               (epoch, avg_loss, num_chunks, jump_rate, notes) VALUES (?, ?, ?, ?, ?)''',
                            (epoch, loss_to_store, num_chunks, float(jump_rate) if np.isfinite(jump_rate) else None, notes))
             self.conn.commit()
         except sqlite3.Error as e: print(f"ERROR logging epoch summary DB: {e}")
         except Exception as e: print(f"ERROR logging epoch summary non-DB: {e}")
    def close(self):
        if self.conn:
            try: self.conn.commit()
            except sqlite3.Error as e: print(f"WARNUNG: Commit fail on close: {e}")
            finally:
                 try: self.conn.close()
                 except sqlite3.Error as e: print(f"WARNUNG: DB close fail: {e}")
                 finally: self.conn, self.cursor = None, None

# ########################################################################
# # 8. Hilfsfunktionen & Konfiguration
# ########################################################################

def load_config(config_file: str = "config_arona.json") -> Dict:
    """Lädt die Konfigurationsdatei und merged mit Defaults."""
    default_config = {
        "num_qubits_per_node": 2, # Default auf 2 geändert
        "simulation_shots": 30,
        "training_epochs": 10,
        "learning_rate_classical": 0.01,
        "learning_rate_quantum": 0.005,
        "dataset_files": ["./training_data/sample1.txt"],
        "chunk_size": 400,
        "chunk_overlap": 40,
        "embedding_dim": 64,
        "checkpoint_dir": "./checkpoints_arona_nq2", # Angepasst für 2 Qubits
        "log_db_path": "training_arona_nq2.db",       # Angepasst für 2 Qubits
        "auto_load_latest_checkpoint": False,
        "feedback_weight_activation": 0.6,
        "feedback_weight_emotion": 0.4,
        "update_weight_activation_error": 0.7,
        "update_weight_emotion_error": 0.3,
        "arousal_error_weight_rz": 0.2,
        "dominance_error_weight_rz": 0.2,
        "enable_dynamic_shots": True,
        "stagnation_threshold_loss": 0.005,
        "shots_increase_factor": 1.5,
        "shots_decrease_factor": 0.95,
        "max_simulation_shots": 30, # Ggf. anpassen
        "min_simulation_shots": 3,  # Ggf. anpassen
        "jump_rate_threshold_for_shot_decrease": 0.1,
        "loss_improvement_threshold_for_jump": -0.01,
        "feedback_lr_scaling_factor": 0.5,
        "log_interval_chunks": 100,
        "enable_perturbation": True,
        "perturbation_std_dev": 0.03,
        "randomize_shots": True,
        "shot_random_min": -1,
        "shot_random_max": 2,
        "enable_jump_boost": True,
        "jump_threshold_high": 2, # Angepasst: (2**2)/2 = 2
        "jump_boost_lr_factor_high": 1.4,
        "jump_boost_lr_factor_low": 1.1,
        "jump_lr_dampen_factor": 0.98,
        "enable_variance_trigger": True,
        "variance_threshold_low": 0.5, # Muss evtl. für 2 Qubits angepasst werden
        "use_hebbian_learning": True,
        "hebb_history_window": 3,
        "hebb_threshold_high": 0.55,
        "hebb_threshold_low": 0.30,
        "hebb_reg_factor": 0.001,
        "activation_history_len": 50,
        # *** NEU: Parameter für Peak Loss Persistence ***
        "peak_loss_tracking_enabled": True, # Standardmäßig aktiviert
        # *** Ende neue Parameter ***
        "network_structure": {
            "nodes": [
              {"label": "Limbus Affektus", "class": "LimbusAffektus", "num_qubits": 2},
              {"label": "Meta Cognitio", "class": "MetaCognitio", "num_qubits": 2},
              {"label": "Cortex Criticus", "class": "CortexCriticus", "num_qubits": 2},
              {"label": "Cortex Creativus", "class": "CortexCreativus", "num_qubits": 2},
              {"label": "Simulatrix Neuralis", "class": "SimulatrixNeuralis", "num_qubits": 2},
              {"label": "Cortex Socialis", "class": "CortexSocialis", "num_qubits": 2},
              {"label": "Philosophie", "class": "MemoryNode", "num_qubits": 2},
              {"label": "Ethik", "class": "MemoryNode", "num_qubits": 2},
              {"label": "Technologie", "class": "MemoryNode", "num_qubits": 2},
              {"label": "Bewusstsein", "class": "MemoryNode", "num_qubits": 2},
              {"label": "Ziel_Rationalitaet", "class": "ValueNode", "initial_value": 0.6},
              {"label": "Ziel_Empathie", "class": "ValueNode", "initial_value": 0.4}
            ],
            "connections": [
              {"source": "Philosophie", "target": "Meta Cognitio", "weight": 0.3},
              {"source": "Ethik", "target": "Cortex Criticus", "weight": 0.5},
              {"source": "Technologie", "target": "Cortex Creativus", "weight": 0.4},
              {"source": "Limbus Affektus", "target": "Cortex Creativus"},
              {"source": "Limbus Affektus", "target": "Cortex Criticus"},
              {"source": "Meta Cognitio", "target": "Limbus Affektus"},
              {"source": "Ziel_Rationalitaet", "target": "Meta Cognitio", "weight": 0.6},
              {"source": "Ziel_Empathie", "target": "Limbus Affektus", "weight": 0.5}
            ]
        }
    }
    # Rest der Funktion bleibt gleich (Laden & Mergen)
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f: loaded_config = json.load(f)
            merged_config = default_config.copy()
            # Tieferes Mergen für verschachtelte Dictionaries
            for key, value in loaded_config.items():
                if isinstance(value, dict) and isinstance(merged_config.get(key), dict):
                    if key == "network_structure":
                         merged_config[key] = default_config[key].copy()
                         nodes_default = {n['label']: n for n in merged_config[key]['nodes']}
                         nodes_loaded = {n['label']: n for n in value.get('nodes', [])}
                         nodes_default.update(nodes_loaded)
                         merged_config[key]['nodes'] = list(nodes_default.values())
                         merged_config[key]['connections'] = value.get('connections', merged_config[key]['connections'])
                    else: merged_config[key].update(value)
                else: merged_config[key] = value
            print(f"Konfiguration aus '{config_file}' geladen und mit Defaults gemerged.")
            return merged_config
        except Exception as e: print(f"ERROR loading config {config_file}: {e}. Using defaults.")
    else: print(f"WARNUNG: Config '{config_file}' nicht gefunden. Verwende Defaults.")
    return default_config

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting Quantum Arona Core v1.1 (Experimental Q-LLM with Raumspaltung)")
    config = load_config("config_arona.json")

    db_path = config.get("log_db_path", "training_arona.db")
    persistence_manager = None
    try: persistence_manager = PersistenceManager(db_path)
    except Exception as e: print(f"FATAL: Could not initialize PersistenceManager: {e}. Exiting.") ; exit(1)

    try:
        arona_model = QuantumAronaModel(config)
        loader = DatasetLoader(config.get("dataset_files", []), config.get("chunk_size", 500), config.get("chunk_overlap", 50))
        contextualizer = Contextualizer()
        embedder = StateEmbedder(config.get("embedding_dim", 128))
    except Exception as e: print(f"FATAL: Error initializing core components: {e}"); traceback.print_exc(); exit(1)

    trainer = QuantumTrainer(arona_model, loader, contextualizer, embedder, config, persistence_manager)

    if config.get("auto_load_latest_checkpoint", False):
        checkpoint_dir = config.get("checkpoint_dir", ".")
        try:
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("quantum_arona_checkpoint_epoch_") and f.endswith(".json")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f"Attempting to auto-load latest checkpoint: {latest_checkpoint}")
                if trainer.load_checkpoint(latest_checkpoint): print("Successfully loaded latest checkpoint.")
                else: print("Failed to load latest checkpoint, starting fresh.")
            else: print("No checkpoints found for auto-loading.")
        except Exception as e: print(f"Error during auto-load: {e}")

    try: trainer.train()
    except KeyboardInterrupt: print("\nTraining interrupted by user.")
    except Exception as train_error: print(f"\nFATAL ERROR during training: {train_error}"); traceback.print_exc()
    finally:
        if persistence_manager: print("Closing database connection..."); persistence_manager.close()
        # Optional: Finalen Checkpoint speichern
        # final_checkpoint = os.path.join(config.get("checkpoint_dir", "."), "quantum_arona_checkpoint_final.json")
        # trainer.save_checkpoint(final_checkpoint)
        # print(f"Final state saved to {final_checkpoint}")

    print("Quantum Arona Core training process finished.")
