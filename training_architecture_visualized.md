## Diagramm 1: Gesamtüberblick Trainingsprozess (QuantumTrainer.train)
```mermaid
graph TD
    A["Start: train_arona.py"] --> B("Lade Config")
    B --> C("Init: PersistenceManager")
    C --> D("Init: QuantumAronaModel")
    D --> E("Init: DataLoader, Contextualizer, Embedder")
    E --> F("Init: QuantumTrainer")
    F --> G{"Checkpoint laden?"}
    G --|Ja|--> H["trainer.load_checkpoint"]
    G --|Nein|--> I["Start Training Loop"]
    H --> I

    %% Haupt-Trainingsschleife
    subgraph trainer_train["QuantumTrainer.train - Epochenschleife"]
        I --> J{"Epoche < Max Epochen?"}
        J --|Ja|--> K["epoch_num++"]
        K --> L["Führe trainer train_epoch aus"]
        L --> M["Erhalte: avg_loss, jump_rate, avg_variance"]
        M --> N{"avg_loss gültig?"}
        N --|Nein|--> J

        N --|Ja|--> O{"Peak Loss Tracking aktiviert?"}

        %% Peak Loss Persistence (PLP) Logic
        subgraph plp_logic["PLP & Adaptive Strategien am Epochenende"]
            O --|Ja|--> P{"Ist persistierend?"}
            P --|Ja|--> Q{"avg_loss > loss_at_last_peak?"}
            Q --|Ja|--> R["PLP: Beende Persistenz, run_adapt=True"]
            Q --|Nein|--> S["PLP: Weiter persistieren, run_adapt=False"]
            P --|Nein|--> T{"avg_loss > highest_loss_recorded?"}
            T --|Ja|--> U["PLP: Aktiviere Persistenz (nächste Ep.), run_adapt=True"]
            T --|Nein|--> V["PLP: Keine Änderung, run_adapt=True"]
            O --|Nein|--> V 

            R --> W{"Führe adaptive Strategien aus? (run_adapt)"}
            S --> SkipAdapt["Überspringe Anpassungen"]
            U --> W
            V --> W

            W --|Ja|--> Adapt["Wende Stagnation/Varianz/Dynamische Shot-Anpassung an"]
            W --|Nein|--> AdaptSkip["Keine Adaption nötig/möglich"]

            Adapt --> EndAdapt
            AdaptSkip --> EndAdapt
            SkipAdapt --> EndAdapt
        end

        EndAdapt["Ende der Anpassung"] --> II["Update last_epoch_avg_loss (falls nötig)"]
        II --> JJ["Save Checkpoint"]
        JJ --> J 
    end

    %% Ende Training
    J --|Nein|--> KK["Training Ende"]
    KK --> LL["Schließe PersistenceManager"]
    LL --> ZZZ["Ende"]

```

---

---
## Diagramm 2: Ablauf einer einzelnen Epoche (QuantumTrainer.train_epoch)
```mermaid
graph TD
    L_Start["Start train_epoch (aus Diagramm 1: K)"] --> L0["Init: total_loss=0, etc."]
    L0 --> L1{"For each chunk_data in DataLoader.generate_chunks"}
    L1 --|Chunk|--> L1a{"Ist persistierend? (Trainer-Status)"}
    L1a --|Ja|--> L2b["effective_shots = current_n_shots"]
    L1a --|Nein|--> L2a{"Random Shots aktiviert?"}
    L2a --|Ja|--> L2a_1["Berechne random shot_noise"]
    L2a_1 --> L2b["effective_shots = clip(current_n_shots + shot_noise)"]
    L2a --|Nein|--> L2b

    L2b --> L3["contextualizer.add_context"]
    L3 --> L4["Führe model.step aus (siehe Diagramm 3)"]
    L4 --> L5["Erhalte current_state"]
    L5 --> L6["target_state = _get_target_state(context)"]
    L6 --> L7["_calculate_feedback"]
    L7 --> L7a["Erhalte: overall_feedback, feedback_comps"]
    L7a --> L7b["loss = 1 - (overall_feedback + 1)/2"]
    L7b --> L8["Analysiere Sprünge & Varianz aus current_state"]
    L8 --> L8a["Sammle chunk_variances für Epochen-Statistik"]
    L8a --> L9{"Loss gültig?"}
    L9 --|Nein|--> L1 

    L9 --|Ja|--> L9a["Berechne loss_delta"]
    L9a --> L9b["Akkumuliere total_loss, processed_chunks++"]
    L9b --> L10["calculate_dynamic_learning_rates"]
    L10 --> L10a["Erhalte: dyn_lr_c, dyn_lr_q"]
    L10a --> L11{"Jump Boost aktiviert?"}
    L11 --|Ja|--> L12["Berechne boost_factor basierend auf chunk_max_jump"]
    L12 --> L12a["boosted_lr_q = dyn_lr_q * boost_factor"]
    L11 --|Nein|--> L13["boosted_lr_q = dyn_lr_q"]
    L12a --> L13

    L13 --> L14{"Hebb'sches Lernen aktiviert?"}
    L14 --|Ja|--> L15["Wende Hebb'sches Lernen an (siehe Diagramm 6)"] 
    L14 --|Nein|--> L16["Berechne Feedback-basierte Updates (calculate_parameter_updates)"]
    L16 --> L17["Wende Feedback-Updates an (model.apply_updates)"]

    L15 --> L18
    L17 --> L18

    L18 --> L19{"MetaCognitio vorhanden?"}
    L19 --|Ja|--> L20{"Sprung erkannt UND Loss verbessert?"}
    L20 --|Ja|--> L21["meta_cog.log_reflection"]
    L20 --|Nein|--> L22
    L19 --|Nein|--> L22

    L21 --> L22
    L22 --> L23{"Logging Intervall erreicht?"}
    L23 --|Ja|--> L24["embedder.embed_state"]
    L24 --> L25["persistence_manager.log_chunk_result"]
    L25 --> L26
    L23 --|Nein|--> L26

    L26 --> L27{"TQDM aktiv?"}
    L27 --|Ja|--> L28["Aktualisiere TQDM Postfix"]
    L28 --> L1 
    L27 --|Nein|--> L1 

    L1 --|Ende Chunks|--> L_End["Ende train_epoch (zu Diagramm 1: M)"]

```
---

---
## Diagramm 3: Ablauf eines Modellschritts (QuantumAronaModel.step)
```mermaid
graph TD
    step_start["Start model.step (aus Diagramm 2: L3)"] --> step1{"Kontext anwenden?"}
    step1 --|Ja|--> step1a["Emotionen/Ziele aus Kontext setzen"]
    step1 --|Nein|--> step1b["Überspringen"]
    step1a --> step1b
    step1b --> step2{"Input Chunk vorhanden?"}
    step2 --|Ja|--> step2a["model.apply_text_input (boostet MemoryNodes)"]
    step2 --|Nein|--> step3
    step2a --> step3
    step3 --> step4["Hole Emotionsfaktoren von Limbus Affektus"]
    step4 --> step5["model.calculate_classic_input_sum (berechnet activation_sum für alle Nodes)"]
    step5 --> step6{"For each node in model.nodes"}
    step6 --|Node|--> step7["Rufe node.calculate_activation auf"] 
    step7 --> step6 
    step6 --|Ende Nodes|--> step8["Aktualisiere interne Zustände kognitiver Module (Limbus, MetaCog, etc.)"]
    step8 --> step9["state_extractor.extract_current_state"]
    step9 --> step_end["Ende model.step (zu Diagramm 2: L5)"]

```


---

---
## Diagramm 4: Ablauf der Knotenaktivierung (Node.calculate_activation)
```mermaid
graph TD
    nca_start["Start node.calculate_activation (aus Diagramm 3: step6)"] --> nca1{"Ist Quantum Node mit QNS?"}
    nca1 --|Ja|--> nca2["Rufe qns.activate auf (siehe Diagramm 5)"]
    nca2 --> nca3["Setze node.activation, Q-Logs"]
    nca1 --|Nein|--> nca4["Berechne klassische Aktivierung (z.B. Sigmoid)"]
    nca4 --> nca5["Setze node.activation, keine Q-Logs"]
    nca3 --> nca6
    nca5 --> nca6
    nca6 --> nca7["Füge node.activation zu node.activation_history hinzu"]
    nca7 --> nca_end["Ende node.calculate_activation (zurück zu Diagramm 3: step7)"]

```

---

---
## Diagramm 5: Ablauf der Quantensimulation (QuantumNodeSystem.activate)
```mermaid
graph TD
    qnsa_start["Start qns.activate (aus Diagramm 4: nca1)"] --> qnsa1["Build PQC Ops (H, RY, RZ, CNOT)"]
    qnsa1 --> qnsa2["Init: total_hamming=0, measurement_log=[]"]
    qnsa2 --> qnsa3{"For shot in n_shots"}
    qnsa3 --|Shot|--> qnsa4["Init state_vector = |0...0>"]
    qnsa4 --> qnsa5{"For op in PQC Ops"}
    qnsa5 --|Op|--> qnsa6["Wende Gate an (_apply_gate / _apply_cnot)"]
    qnsa6 --> qnsa7{"Gate OK?"}
    qnsa7 --|Nein (Error)|--> qnsa3
    qnsa7 --|Ja|--> qnsa5 
    qnsa5 --|Ende PQC Ops|--> qnsa8["Berechne Wahrscheinlichkeiten (abs(state_vector)**2)"]
    qnsa8 --> qnsa9["Normalisiere Wahrscheinlichkeiten"]
    qnsa9 --> qnsa10["Messe Zustand (np.random.choice)"]
    qnsa10 --> qnsa11["Berechne Hamming-Gewicht"]
    qnsa11 --> qnsa12["Logge Messung (index, binary, hamming)"]
    qnsa12 --> qnsa13["Addiere zu total_hamming_weight"]
    qnsa13 --> qnsa3 
    qnsa3 --|Ende Shots|--> qnsa14["Berechne activation_prob"]
    qnsa14 --> qnsa_end["Ende qns.activate (zu Diagramm 4: nca3)"]
```


---

---
## Diagramm 6: Ablauf der Lernregelanwendung (Fokus auf Hebb)
```mermaid
graph TD
    L13_entry["(aus Diagramm 2: L13)"] --> L14{"Hebb'sches Lernen aktiviert? (use_hebbian_learning)"}
    L14 --|Ja|--> L15["Rufe model.apply_hebbian_learning auf"]

    subgraph hebb_learning["model.apply_hebbian_learning"]
         hl_start["Start apply_hebbian_learning"] --> hl1{"For each node in model.nodes"}
         hl1 --|Node|--> hl2{"For each conn in node.connections"}
         hl2 --|Connection|--> hl3["Rufe hebbian_learning_quantum_node_smoothed auf"]

         subgraph hl_smoothed["hebbian_learning_quantum_node_smoothed"]
              hls_start["Start smoothed hebb"] --> hls1["node_b = conn.target_node"]
              hls1 --> hls2["act_a_smooth = node_a.get_smoothed_activation()"]
              hls2 --> hls3["act_b_smooth = node_b.get_smoothed_activation()"]
              hls3 --> hls4{"LTP? (a > Thr_H & b > Thr_H)"}
              hls4 --|Ja|--> hls5["delta_w = lr_c * a * b"]
              hls5 --> hls6["update_classical_weight(conn, delta_w)"]
              hls6 --> hls7{"Node A Quantum?"}
              hls7 --|Ja|--> hls8["delta_q = lr_q * a * b * 0.5 (RY)"]
              hls8 --> hls9["update_quantum_params(node_a, delta_q)"]
              hls9 --> hls13
              hls7 --|Nein|--> hls13
              hls4 --|Nein|--> hls10{"LTD? (a > Thr_H & b < Thr_L)"}
              hls10 --|Ja|--> hls11["delta_w = -0.1 * lr_c * a"]
              hls11 --> hls12["Berechne & Wende LTD für Q-Params an"]
              hls12 --> hls13
              hls10 --|Nein|--> hls13
              hls13 --> hls14["Regularisierung: conn.weight -= reg * weight"]
              hls14 --> hls_end["Ende smoothed hebb"]
         end

         hl3 --> hl2 
         hl2 --|Ende Connections|--> hl1 
         hl1 --|Ende Nodes|--> hl_end["Ende apply_hebbian_learning"]
    end

    L14 --|Nein|--> L16["Berechne Feedback-Updates (calculate_parameter_updates)"]
    L16 --> L17["Wende Feedback-Updates an (model.apply_updates)"]

    L15 --> L18_exit["(zu Diagramm 2: L18)"]
    hl_end --> L18_exit 
    L17 --> L18_exit 
```
