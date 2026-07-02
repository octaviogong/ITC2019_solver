# -*- coding: utf-8 -*-
"""
=============================================================================
  CONFIGURACIÓN CENTRAL — Solver ITC 2019 (API-Carpio)
  Optimizado para Mitigación Espectral y Cadenas de Kempe
=============================================================================
"""
import numpy as np

# ---------------------------------------------------------------------------
# 1. RUTAS Y SELECCIÓN DE INSTANCIAS
# ---------------------------------------------------------------------------
PATHS = {
    "instances_dir": r"D:\cegnr\OneDrive - Instituto Tecnológico de León\MCC\PATAT\PATAT2019\instancias_itc2019_menores",
    "solutions_dir": r"D:\cegnr\OneDrive - Instituto Tecnológico de León\MCC\PATAT\solver\GNN\soluciones",
    "model_weights": r"D:\cegnr\OneDrive - Instituto Tecnológico de León\MCC\PATAT\solver\GNN\model\gnn_weights.pt",
    "log_file":      r"D:\cegnr\OneDrive - Instituto Tecnológico de León\MCC\PATAT\solver\GNN\logs\solver_run.log",
}

INSTANCES = {
    "run_only": None,
    "exclude": [],
}
 
# ---------------------------------------------------------------------------
# 2. HAMILTONIANO SURROGADO (Energía para el Recocido Simulado)
# ---------------------------------------------------------------------------
SURROGATE_WEIGHTS = {
    "student_conflict": 10000.0,
    "student_travel":    50.0,
    "distribution_soft":  5.0,
    "time_preference":    2.0,
    "room_preference":    1.0
}
 
# ---------------------------------------------------------------------------
# 3. ENTRENAMIENTO Y ARQUITECTURA HÍBRIDA (GCN + LSTM)
# ---------------------------------------------------------------------------
TRAIN = {
    "seed":         1002,
    "train_ratio":  1.0,
    "retrain":      True,
    "epochs":       1000,
    "batch_size":   40,
    "lr":           0.008,
    "weight_decay": 1e-3,
    "device":       "cpu",
}
 
GNN = {
    "hidden_dims":        [128, 128, 64],
    "activation":         "leaky_relu",
    "batch_norm":         True,
    "dropout":            0.1,
    "gaussian_noise_std": 0.15,
}
 
# ---------------------------------------------------------------------------
# 4. EXTRACCIÓN ESPECTRAL Y POLITOPOS LATINOS (FASE 2)
# ---------------------------------------------------------------------------
SPECTRAL = {
    "lambda2_threshold":   0.005,
    "lambda2_increment":   0.015,
    "max_polytope_size":   1000,
    "max_refrag_attempts": 100,
    "merge_rigid_cliques": True,
}
 
# ---------------------------------------------------------------------------
# 5. MOTOR DE RESTRICCIONES (CSP + Forward Checking + Backjumping)
# ---------------------------------------------------------------------------
# ───────────── PERFIL "PRUEBAS" (rápido, para iterar) ─────────────
# Cada restart cuesta segundos; 12 restarts dan buena cobertura para depurar
# en instancias chicas/medianas (lums, bet) en ~2-5 min.
CSP = {
    "forward_checking": True,
    "timeout_seconds":  2000,     # 15 min de tope duro
    "restarts":         100,      # suficiente para que varias semillas toquen 0
    "max_backtracks":   3000,   # presupuesto por restart antes de pasar a reparación
    "gnn_weight":       0.7,
}
 
# ───────────── PERFIL "COMPLETO" (calidad de competencia) ─────────────
# Descomenta este bloque (y comenta el de arriba) para corridas finales.
# Más restarts = mayor probabilidad de una solución válida y de menor costo.
# CSP = {
#     "forward_checking": True,
#     "timeout_seconds":  3600,    # 1 hora de tope duro
#     "restarts":         60,      # más semillas => más chance de 0 errores reales
#     "max_backtracks":   30000,
#     "gnn_weight":       0.7,
# }
 
# ---------------------------------------------------------------------------
# 6. RECOCIDO SIMULADO ESPECTRAL (FASE 5)
# ---------------------------------------------------------------------------
# Para PRUEBAS conviene un SA corto (el CSP ya entrega base válida).
SA = {
    "enabled":            True,
    "max_iterations":     5000,   # PRUEBAS: 50k. COMPLETO: sube a 300000-500000.
    "initial_temp":       2000.0,
    "min_temp":           0.001,
    "cooling_rate":       0.995,
    "kempe_prob":         0.9,
    "boundary_node_bias": 5.0
}
 
# ---------------------------------------------------------------------------
# 7. SECCIONAMIENTO DESACOPLADO (Gale-Shapley - FASE 6)
# ---------------------------------------------------------------------------
SECTIONING = {
    "enabled":                    True,
    "student_pref_travel_weight": 10.0,
    "student_pref_time_weight":   5.0,
    "class_pref_capacity_weight": 1.0
}
 
# ---------------------------------------------------------------------------
# 8. SALIDA Y LOGS
# ---------------------------------------------------------------------------
OUTPUT = {
    "verbose":           True,
    "save_metrics_csv":  True,
    "metrics_csv_path":  r"D:\cegnr\OneDrive - Instituto Tecnológico de León\MCC\PATAT\solver\GNN\logs\metrics.csv",
    "validate_solution": True,
}