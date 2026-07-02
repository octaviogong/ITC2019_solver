"""
=============================================================================
  SOLVER UNIFICADO ITC 2019 — API-Carpio
  Fases: Parser → Extractor Espectral → GNN Híbrida → CSP + Backtracking
  
  USO:
      python solver_main.py                    # usa config.py por defecto
      python solver_main.py --config mi_config.py
      python solver_main.py --instance pu-cs-fal07  # una sola instancia
      python solver_main.py --retrain          # fuerza reentrenamiento GNN
=============================================================================
"""

import argparse
import csv
import importlib.util
import logging
import math
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ── PyTorch / PyG (opcional: si no están disponibles, se usa modo CSP puro) ──
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, BatchNorm
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    print("[AVISO] PyTorch/PyG no disponibles. Se usará CSP puro sin guía GNN.")

# ── SciPy para álgebra espectral ──
try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[AVISO] SciPy no disponible. Extracción espectral desactivada.")


# =============================================================================
#  CARGA DE CONFIGURACIÓN
# =============================================================================

def load_config(path: str = "config.py"):
    """Carga el módulo de configuración desde un path arbitrario."""
    spec = importlib.util.spec_from_file_location("config", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


# =============================================================================
#  ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass
class TimeSlot:
    id: str
    days: str = ""
    start: str = ""
    weeks: str = ""
    # NUEVO: Pre-cálculo para operaciones bitwise O(1)
    days_mask: int = 0
    weeks_mask: int = 0
    start_int: int = 0

@dataclass
class Room:
    id: str
    capacity: int = 0
    features: Set[str] = field(default_factory=set)
    unavailable: List[Dict[str, any]] = field(default_factory=list)
    travel: Dict[str, int] = field(default_factory=dict) # NUEVO: Mapeo inter-aulas (id_destino -> slots)

@dataclass
class CourseClass:
    id: str
    course_id: str
    config_id: str
    limit: int = 0
    length: int = 1
    parent: str = "" # NUEVO: Relación jerárquica para estudiantes
    room_required: bool = True
    allowed_rooms: Dict[str, int] = field(default_factory=dict) # NUEVO: id_aula -> penalty
    allowed_times: Dict[str, int] = field(default_factory=dict) # NUEVO: id_tiempo -> penalty
    students: Set[str] = field(default_factory=set)

@dataclass
class Distribution:
    id: str
    type: str
    required: bool = True
    penalty: int = 1   # NUEVO: Para que SA sepa cuánto cobrar
    classes: List[str] = field(default_factory=list)

@dataclass
class Instance:
    name: str
    optimization: Dict[str, int] = field(default_factory=dict)
    timeslots: Dict[str, TimeSlot] = field(default_factory=dict)
    rooms: Dict[str, Room] = field(default_factory=dict)
    classes: Dict[str, CourseClass] = field(default_factory=dict)
    distributions: List[Distribution] = field(default_factory=list)
    num_weeks: int = 1
    num_days: int = 5
    slots_per_day: int = 10


# =============================================================================
#  FASE 1 — PARSER XML ITC 2019
# =============================================================================

class ITC2019Parser:
    """Parsea un archivo .xml de instancia oficial ITC 2019 respetando la herencia de subpart."""

    def _parse_time(self, tm, inst, time_map):
        """Helper para extraer y registrar tiempos físicos."""
        t_id = tm.get("id")
        t_days = tm.get("days", "0")
        t_start = tm.get("start", "0")
        t_weeks = tm.get("weeks", "0")

        # Conversiones rápidas
        d_mask = int(t_days, 2) if t_days else 0
        w_mask = int(t_weeks, 2) if t_weeks else 0
        s_int  = int(t_start) if t_start else 0

        if not t_id:
            t_key = (t_days, t_start, t_weeks)
            if t_key not in time_map:
                t_id = f"T_d{t_days}_s{t_start}_w{t_weeks[:3]}"
                time_map[t_key] = t_id
                inst.timeslots[t_id] = TimeSlot(
                    id=t_id, days=t_days, start=t_start, weeks=t_weeks,
                    days_mask=d_mask, weeks_mask=w_mask, start_int=s_int
                )
            else:
                t_id = time_map[t_key]
        else:
            if t_id not in inst.timeslots:
                inst.timeslots[t_id] = TimeSlot(
                    id=t_id, days=t_days, start=t_start, weeks=t_weeks,
                    days_mask=d_mask, weeks_mask=w_mask, start_int=s_int
                )
        return t_id

    def parse(self, xml_path: str) -> Instance:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        name = Path(xml_path).stem
        inst = Instance(name=name)

        # Parámetros globales
        inst.num_weeks     = int(root.get("nrWeeks", 1))
        inst.num_days      = int(root.get("nrDays",  5))
        inst.slots_per_day = int(root.get("slotsPerDay", 10))

        # Pesos de optimización
        opt = root.find("optimization")
        if opt is not None:
            for attr in ["time", "room", "distribution", "student"]:
                inst.optimization[attr] = int(opt.get(attr, 0))
        
        time_map = {}

        # Aulas y bloqueos (Hard Constraints)
        for r in root.iter("room"):
            rm = Room(
                id       = r.get("id", ""),
                capacity = int(r.get("capacity", 0)),
            )
            for feat in r.findall("feature"):
                rm.features.add(feat.get("id", ""))
            for una in r.findall("unavailable"):
                d_str = una.get("days", "0")
                w_str = una.get("weeks", "0")
                rm.unavailable.append({
                    "days_mask": int(d_str, 2) if d_str else 0,   # Máscara binaria
                    "weeks_mask": int(w_str, 2) if w_str else 0,  # Máscara binaria
                    "start": int(una.get("start", "0")),
                    "length": int(una.get("length", "0"))
                })
            # Extraer distancias inter-aulas
            for trv in r.findall("travel"):
                rm.travel[trv.get("room", "")] = int(trv.get("value", "0"))
                
            inst.rooms[rm.id] = rm

        # Jerarquía: Course -> Config -> Subpart -> Class
        for course in root.iter("course"):
            course_id = course.get("id", "")
            for cfg in course.findall("config"):
                config_id = cfg.get("id", "")
                
                # Bajar explícitamente al nivel de Subpart para capturar herencia
                for subpart in cfg.findall("subpart"):
                    
                    # 1. Extraer dominios base del Subpart
                    subpart_times = [self._parse_time(tm, inst, time_map) for tm in subpart.findall("time")]
                    subpart_rooms = [rm.get("id", "") for rm in subpart.findall("room")]
                    
                    # NUEVO: Extraer la duración (length) desde el nivel de Subpart
                    subpart_length = int(subpart.get("length", 1))
                    
                    for cls in subpart.findall("class"):
                        # HERENCIA NIVEL 4: Class
                        cls_length = int(cls.get("length", subpart_length))
                        
                        cc = CourseClass(
                            id        = cls.get("id", ""),
                            course_id = course_id,
                            config_id = config_id,
                            limit     = int(cls.get("limit", 0)),
                            length    = cls_length,
                            parent    = cls.get("parent", "") # NUEVO: Jerarquía extraída
                        )
                        
                        # Dominios de Tiempo con Penalizaciones
                        cls_times = {self._parse_time(tm, inst, time_map): int(tm.get("penalty", "0")) for tm in cls.findall("time")}
                        cc.allowed_times = cls_times if cls_times else {t: 0 for t in subpart_times}
                        
                        # Dominios de Aula con Penalizaciones
                        cls_rooms = {rm.get("id", ""): int(rm.get("penalty", "0")) for rm in cls.findall("room")}
                        cc.allowed_rooms = cls_rooms if cls_rooms else {r: 0 for r in subpart_rooms}
                        
                        if cls.get("room", "true").lower() == "false":
                            cc.room_required = False
                            cc.allowed_rooms = {}
                        else:
                            cc.room_required = len(cc.allowed_rooms) > 0

                        # Estudiantes
                        for st in cls.findall("student"):
                            cc.students.add(st.get("id", ""))

                        inst.classes[cc.id] = cc
                        
                        

        # Distribuciones
        for dist in root.iter("distribution"):
            d = Distribution(
                id       = dist.get("id", ""),
                type     = dist.get("type", ""),
                required = dist.get("required", "true").lower() == "true",
                penalty  = int(dist.get("penalty", "1")) # NUEVO: Extracción de costo
            )

        return inst

# =============================================================================
#  FASE 2 — EXTRACTOR ESPECTRAL / CONSTRUCCIÓN DE POLITOPOS
# =============================================================================

class SpectralExtractor:
    """
    Construye el grafo de conflictos y lo descompone en sub-grafos (politopos)
    usando el valor de Fiedler (λ₂) con umbral adaptativo.
    """

    def __init__(self, cfg):
        self.lambda2_threshold  = cfg.SPECTRAL["lambda2_threshold"]
        self.max_size           = cfg.SPECTRAL["max_polytope_size"]
        self.lambda2_increment  = cfg.SPECTRAL["lambda2_increment"]
        self.max_refrag         = cfg.SPECTRAL["max_refrag_attempts"]
        self.merge_rigid        = cfg.SPECTRAL["merge_rigid_cliques"]

    # ------------------------------------------------------------------
    def build_conflict_graph(self, inst: Instance):
        """
        Construye el grafo de conflictos ponderado.
        Aristas = clases que comparten estudiantes o restricciones duras.
        """
        class_ids = list(inst.classes.keys())
        n = len(class_ids)
        idx = {cid: i for i, cid in enumerate(class_ids)}

        # Matriz de adyacencia (ponderada por estudiantes compartidos)
        W = np.zeros((n, n), dtype=np.float32)

        for i, cid_i in enumerate(class_ids):
            for j, cid_j in enumerate(class_ids[i+1:], start=i+1):
                ci = inst.classes[cid_i]
                cj = inst.classes[cid_j]
                shared = len(ci.students & cj.students)
                if shared > 0:
                    W[i, j] = shared
                    W[j, i] = shared

        # Aristas adicionales por distribuciones obligatorias
        for dist in inst.distributions:
            if dist.required:
                for a in dist.classes:
                    for b in dist.classes:
                        if a != b and a in idx and b in idx:
                            i, j = idx[a], idx[b]
                            W[i, j] = max(W[i, j], 1.0)
                            W[j, i] = max(W[j, i], 1.0)

        return class_ids, W, idx

    # ------------------------------------------------------------------
    def _fiedler_value(self, W_sub: np.ndarray) -> float:
        """Calcula el valor de Fiedler (λ₂) de un sub-grafo."""
        if not SCIPY_AVAILABLE or W_sub.shape[0] < 3:
            return 1.0  # asumir bien conectado si no hay SciPy
        D = np.diag(W_sub.sum(axis=1))
        L = D - W_sub
        L_sparse = csr_matrix(L)
        try:
            vals = eigsh(L_sparse, k=2, which="SM", return_eigenvectors=False)
            return float(sorted(vals)[1])
        except Exception:
            return 1.0

    # ------------------------------------------------------------------
    def _partition_by_fiedler(self, node_ids, W_sub, threshold):
        """
        Particiona el sub-grafo usando el vector de Fiedler.
        Devuelve dos listas de índices locales.
        """
        n = W_sub.shape[0]
        if n < 4:
            return list(range(n)), []
        D = np.diag(W_sub.sum(axis=1))
        L = D - W_sub
        L_sparse = csr_matrix(L)
        try:
            vals, vecs = eigsh(L_sparse, k=2, which="SM")
            fiedler_vec = vecs[:, 1]
            median = np.median(fiedler_vec)
            part_a = [i for i in range(n) if fiedler_vec[i] <= median]
            part_b = [i for i in range(n) if fiedler_vec[i] >  median]
            return part_a, part_b
        except Exception:
            half = n // 2
            return list(range(half)), list(range(half, n))

    # ------------------------------------------------------------------
    def decompose(self, inst: Instance):
        """
        Punto de entrada principal.
        Devuelve lista de politopos: cada politopo es una lista de class_ids.
        """
        class_ids, W, idx = self.build_conflict_graph(inst)

        # Fusión de clanes rígidos (contracciones de meta-nodos)
        rigid_groups = []
        if self.merge_rigid:
            rigid_groups = self._find_rigid_cliques(inst, idx)

        polytopes = self._recursive_decompose(class_ids, W, self.lambda2_threshold)
        return polytopes, W, idx

    # ------------------------------------------------------------------
    def _recursive_decompose(self, node_ids, W, threshold, depth=0):
        """Descompone recursivamente usando Fiedler con umbral adaptativo."""
        n = len(node_ids)
        if n <= 1:
            return [node_ids]

        lam2 = self._fiedler_value(W)
        attempts = 0

        # Si el grafo está bien conectado Y es grande → fragmentar
        while lam2 >= threshold and n > self.max_size and attempts < self.max_refrag:
            threshold += self.lambda2_increment
            attempts += 1

        if lam2 < threshold or n <= self.max_size:
            return [node_ids]

        # Particionar
        part_a, part_b = self._partition_by_fiedler(node_ids, W, threshold)
        if not part_a or not part_b:
            return [node_ids]

        ids_a = [node_ids[i] for i in part_a]
        ids_b = [node_ids[i] for i in part_b]

        W_a = W[np.ix_(part_a, part_a)]
        W_b = W[np.ix_(part_b, part_b)]

        return (self._recursive_decompose(ids_a, W_a, threshold, depth+1) +
                self._recursive_decompose(ids_b, W_b, threshold, depth+1))

    # ------------------------------------------------------------------
    def _find_rigid_cliques(self, inst: Instance, idx: dict):
        """Identifica grupos de clases vinculadas por restricciones obligatorias de igualdad."""
        groups = []
        for dist in inst.distributions:
            if dist.required and dist.type in ("SameRoom", "SameDays", "SameAttendees"):
                valid = [c for c in dist.classes if c in idx]
                if len(valid) > 1:
                    groups.append(valid)
        return groups

class LatinBoardCollapser:
    """
    Contracción Algebraica (Union-Find).
    Fusiona clases con restricciones duras de igualdad espacial/temporal en Meta-Variables.
    """
    def __init__(self, class_ids):
        self.parent = {cid: cid for cid in class_ids}
        self.rank = {cid: 0 for cid in class_ids}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1

    def collapse_asterisms(self, inst) -> Dict[str, List[str]]:
        """
        Devuelve un diccionario mapeando: { 'id_meta_variable': ['C1', 'C2', ...] }
        """
        for dist in inst.distributions:
            # Colapsamos si el problema exige estrictamente que compartan espacio/tiempo
            if dist.required and dist.type in ("SameTime", "SameRoom"):
                for i in range(len(dist.classes) - 1):
                    self.union(dist.classes[i], dist.classes[i+1])
        
        meta_groups = {}
        for cid in inst.classes.keys():
            root = self.find(cid)
            if root not in meta_groups:
                meta_groups[root] = []
            meta_groups[root].append(cid)
            
        return meta_groups
# =============================================================================
#  FASE 3 — ARQUITECTURA GNN
# =============================================================================

if GNN_AVAILABLE:

    class GaussianNoise(nn.Module):
        def __init__(self, std: float = 0.05):
            super().__init__()
            self.std = std

        def forward(self, x):
            if self.training and self.std > 0:
                return x + torch.randn_like(x) * self.std
            return x


    class TimetablingGCN(nn.Module):
        """
        GCN para predicción de (timeslot, room) por clase.
        Soporta múltiples capas con BatchNorm, Dropout y ruido gaussiano.
        """

        def __init__(self, in_dim: int, hidden_dims: List[int],
                     out_dim: int, cfg):
            super().__init__()
            gcn_cfg = cfg.GNN

            act_map = {
                "relu":       nn.ReLU(),
                "leaky_relu": nn.LeakyReLU(0.1),
                "elu":        nn.ELU(),
            }
            self.act    = act_map.get(gcn_cfg["activation"], nn.LeakyReLU(0.1))
            self.noise  = GaussianNoise(std=gcn_cfg["gaussian_noise_std"])
            self.drop   = nn.Dropout(p=gcn_cfg["dropout"])

            dims = [in_dim] + hidden_dims + [out_dim]
            self.convs  = nn.ModuleList()
            self.bns    = nn.ModuleList()

            for i in range(len(dims) - 1):
                self.convs.append(GCNConv(dims[i], dims[i+1]))
                if gcn_cfg["batch_norm"] and i < len(dims) - 2:
                    self.bns.append(BatchNorm(dims[i+1]))
                else:
                    self.bns.append(nn.Identity())

        def forward(self, x, edge_index, edge_weight=None):
            x = self.noise(x)
            for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
                x = conv(x, edge_index, edge_weight)
                x = bn(x)
                if i < len(self.convs) - 1:
                    x = self.act(x)
                    x = self.drop(x)
            return x


    class GNNOracle:
        """Entrena y sirve predicciones de la GCN."""

        def __init__(self, cfg):
            self.cfg    = cfg
            self.device = torch.device(cfg.TRAIN["device"])
            self.model  = None
            self.logger = logging.getLogger("GNNOracle")

        # ------------------------------------------------------------------
        def _instance_to_graph(self, inst: Instance, W: np.ndarray,
                                idx: Dict[str, int]) -> Data:
            """Convierte instancia + matriz de conflictos a grafo PyG."""
            class_ids = list(idx.keys())
            n = len(class_ids)

            # Features por nodo: [num_allowed_times, num_allowed_rooms, limit_normalizado]
            feat = []
            max_t = max((len(c.allowed_times) for c in inst.classes.values()), default=1)
            max_r = max((len(c.allowed_rooms) for c in inst.classes.values()), default=1)
            max_l = max((c.limit for c in inst.classes.values()), default=1)

            for cid in class_ids:
                c = inst.classes[cid]
                feat.append([
                    len(c.allowed_times) / max(max_t, 1),
                    len(c.allowed_rooms) / max(max_r, 1),
                    c.limit / max(max_l, 1),
                    len(c.students) / max(n, 1),
                ])
            x = torch.tensor(feat, dtype=torch.float)

            # Aristas
            rows, cols, weights = [], [], []
            for i in range(n):
                for j in range(n):
                    if W[i, j] > 0:
                        rows.append(i)
                        cols.append(j)
                        weights.append(float(W[i, j]))
            edge_index  = torch.tensor([rows, cols], dtype=torch.long)
            edge_weight = torch.tensor(weights, dtype=torch.float)

            # Etiqueta: índice de timeslot asignado (para supervisado)
            # En entrenamiento real esto vendría de soluciones conocidas;
            # aquí usamos un placeholder
            y = torch.zeros(n, dtype=torch.long)

            return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)

        # ------------------------------------------------------------------
        def train(self, instances_graphs: List[Tuple[Instance, np.ndarray, Dict]],
                  weight_path: str):
            """Entrena la GCN sobre la lista de instancias."""
            if not instances_graphs:
                self.logger.warning("Sin datos de entrenamiento.")
                return

            # Inferir dimensiones
            sample_inst, sample_W, sample_idx = instances_graphs[0]
            sample_data = self._instance_to_graph(sample_inst, sample_W, sample_idx)
            in_dim  = sample_data.x.shape[1]
            out_dim = max(len(inst.timeslots) for inst, _, _ in instances_graphs)

            self.model = TimetablingGCN(
                in_dim      = in_dim,
                hidden_dims = self.cfg.GNN["hidden_dims"],
                out_dim     = out_dim,
                cfg         = self.cfg,
            ).to(self.device)

            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr           = self.cfg.TRAIN["lr"],
                weight_decay = self.cfg.TRAIN["weight_decay"],
            )

            dataset = [self._instance_to_graph(inst, W, idx)
                       for inst, W, idx in instances_graphs]

            loader = DataLoader(
                dataset,
                batch_size = self.cfg.TRAIN["batch_size"],
                shuffle    = True,
            )

            self.model.train()
            for epoch in range(1, self.cfg.TRAIN["epochs"] + 1):
                total_loss = 0.0
                for batch in loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    
                    out = self.model(batch.x, batch.edge_index, batch.edge_attr)
                    
                    # 1. Pérdida estándar de clasificación
                    loss_ce = F.cross_entropy(out, batch.y % out.shape[1])
                    
                    # 2. PÉRDIDA ESPECTRAL (Regularización de Tutte/Laplaciano)
                    row, col = batch.edge_index
                    weights = batch.edge_attr
                    
                    # Diferencia euclidiana entre distribuciones de probabilidad de vecinos
                    diff = out[row] - out[col]
                    spectral_loss = torch.sum(weights * torch.norm(diff, p=2, dim=1)) / out.shape[0]
                    
                    # Combinación convexa de pérdidas (0.1 es el peso de regularización)
                    loss = loss_ce + (0.1 * spectral_loss)
                    
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if epoch % 10 == 0 or epoch == 1:
                    self.logger.info(f"  Época {epoch:03d}/{self.cfg.TRAIN['epochs']} "
                                     f"| Loss: {total_loss/len(loader):.4f}")

            Path(weight_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), weight_path)
            self.logger.info(f"Pesos guardados en: {weight_path}")

        # ------------------------------------------------------------------
        def load(self, weight_path: str, in_dim: int, out_dim: int):
            """Carga pesos previamente entrenados."""
            self.model = TimetablingGCN(
                in_dim      = in_dim,
                hidden_dims = self.cfg.GNN["hidden_dims"],
                out_dim     = out_dim,
                cfg         = self.cfg,
            ).to(self.device)
            self.model.load_state_dict(
                torch.load(weight_path, map_location=self.device)
            )
            self.model.eval()

        # ------------------------------------------------------------------
        @torch.no_grad()
        def predict_proba(self, inst: Instance, W: np.ndarray,
                          idx: Dict[str, int]) -> np.ndarray:
            """
            Devuelve matriz (n_clases × n_timeslots) de probabilidades.
            Si el modelo no está disponible, devuelve uniforme.
            """
            if self.model is None:
                n = len(idx)
                n_t = len(inst.timeslots)
                return np.ones((n, n_t)) / n_t

            data = self._instance_to_graph(inst, W, idx).to(self.device)
            logits = self.model(data.x, data.edge_index, data.edge_attr)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            return probs


else:
    # Stubs cuando PyTorch no está disponible
    class GNNOracle:
        def __init__(self, cfg): self.cfg = cfg
        def train(self, *a, **kw): pass
        def load(self, *a, **kw): pass
        def predict_proba(self, inst, W, idx):
            n   = len(idx)
            n_t = len(inst.timeslots)
            return np.ones((n, n_t)) / n_t


# =============================================================================
#  FASE 4 — CSP CON BACKTRACKING + FORWARD CHECKING
# =============================================================================

class Assignment:
    """Estado de asignación parcial."""
    __slots__ = ["timeslot", "room"]
    def __init__(self, timeslot: Optional[str] = None, room: Optional[str] = None):
        self.timeslot = timeslot
        self.room     = room


class CSPSolver:
    """
    Motor CSP Topológico:
    - Asignación Atómica de Meta-Variables (Union-Find)
    - Backtracking guiado por GNN
    - Forward Checking Espectral (Cota de Hoffman)
    """

    def __init__(self, inst: Instance, oracle: GNNOracle, W: np.ndarray, idx: Dict[str, int], cfg):
        self.inst     = inst
        self.oracle   = oracle
        self.W        = W
        self.idx      = idx
        self.cfg      = cfg
        self.logger   = logging.getLogger("CSPSolver")

        self.bt_nodes   = 0
        self.start_time = 0.0

        self.proba = oracle.predict_proba(inst, W, idx)
        self.ts_list = list(inst.timeslots.keys())

        # 1. CONTRACCIÓN ATÓMICA DE META-VARIABLES
        collapser = LatinBoardCollapser(list(self.inst.classes.keys()))
        self.meta_groups = collapser.collapse_asterisms(self.inst)
        self.meta_domains = self._build_meta_domains()
    
    # =========================================================================
    # MÉTODOS BASE DE EVALUACIÓN FÍSICA Y ORDENAMIENTO (No borrar)
    # =========================================================================

    def _room_available_for_ts(self, room_id: str, ts_id: str, cls_length: int) -> bool:
        """Verifica que el aula no esté cerrada usando máscaras binarias O(1)."""
        if room_id == "NO_ROOM" or room_id not in self.inst.rooms:
            return True
        room = self.inst.rooms[room_id]
        if not room.unavailable:
            return True

        ts = self.inst.timeslots.get(ts_id)
        if ts is None:
            return True

        ts_start = ts.start_int
        ts_end   = ts_start + cls_length

        for block in room.unavailable:
            if (ts.days_mask & block["days_mask"]) and (ts.weeks_mask & block["weeks_mask"]):
                b_start = block["start"]
                b_end   = b_start + block["length"]
                if not (ts_end <= b_start or b_end <= ts_start):
                    return False
        return True

    def _select_variable(self, unassigned: List[str], domains: Dict[str, List]) -> str:
        """MRV: elige la variable con el dominio más pequeño."""
        return min(unassigned, key=lambda v: len(domains[v]))

    def _order_values(self, var: str, domain: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Ordena valores por probabilidad GNN descendente."""
        ts_to_prob = {ts: 0.0 for ts in self.ts_list}
        # Para Meta-Variables, usamos el ID de la primera clase del clan para buscar su probabilidad
        first_cid = self.meta_groups[var][0] if hasattr(self, 'meta_groups') and var in self.meta_groups else var
        var_idx = self.idx.get(first_cid, 0)

        if var_idx < self.proba.shape[0]:
            for ti, ts in enumerate(self.ts_list):
                if ti < self.proba.shape[1]:
                    ts_to_prob[ts] = float(self.proba[var_idx, ti])

        gnn_w = self.cfg.CSP["gnn_weight"]
        def score(val):
            ts, r = val
            gnn_score  = gnn_w * ts_to_prob.get(ts, 0.0)
            rand_score = (1.0 - gnn_w) * random.random()
            return gnn_score + rand_score

        return sorted(domain, key=score, reverse=True)

    def _share_bits(self, b1: str, b2: str) -> bool:
        return any(x == '1' and y == '1' for x, y in zip(b1, b2))

    def _subset_bits(self, sub: str, sup: str) -> bool:
        return all(x == '0' or y == '1' for x, y in zip(sub, sup))

    def _first_bit(self, b: str) -> int:
        return b.find('1') if '1' in b else 999

    def _conflicts(self, var: str, ts_id: str, room: str, assignment: Dict[str, Assignment]) -> bool:
        """Verificador físico individual."""
        import re
        cls = self.inst.classes[var]
        ts_a = self.inst.timeslots[ts_id]
        ts_a_start = ts_a.start_int
        ts_a_end = ts_a_start + cls.length

        if room != "NO_ROOM" and room in self.inst.rooms:
            if not self._room_available_for_ts(room, ts_id, cls.length):
                return True

        if cls.allowed_times and ts_id not in cls.allowed_times: return True
        if cls.room_required and cls.allowed_rooms and room not in cls.allowed_rooms: return True

        for other_id, asgn in assignment.items():
            if other_id == var: continue
            other_cls = self.inst.classes[other_id]
            ts_b = self.inst.timeslots[asgn.timeslot]
            room_b = asgn.room
            
            share_days_ab = (ts_a.days_mask & ts_b.days_mask) != 0
            share_weeks_ab = (ts_a.weeks_mask & ts_b.weeks_mask) != 0
            ts_b_start = ts_b.start_int
            ts_b_end = ts_b_start + other_cls.length
            
            overlap_time_ab = not (ts_a_end <= ts_b_start or ts_b_end <= ts_a_start)
            overlap_full = overlap_time_ab and share_days_ab and share_weeks_ab

            if overlap_full and room_b == room and room != "NO_ROOM":
                return True

            shared = cls.students & other_cls.students
            if shared:
                if overlap_full: return True 
                if share_days_ab and share_weeks_ab and room != "NO_ROOM" and room_b != "NO_ROOM":
                    trv = self.inst.rooms[room].travel.get(room_b, 0)
                    if trv > 0:
                        if ts_a_start < ts_b_start and ts_a_end + trv > ts_b_start: return True
                        if ts_b_start < ts_a_start and ts_b_end + trv > ts_a_start: return True

        for dist in self.inst.distributions:
            if not dist.required or var not in dist.classes: continue
            match = re.match(r"([a-zA-Z]+)(?:\(([^)]+)\))?", dist.type.strip())
            if not match: continue
            dtype = match.group(1)
            param = int(match.group(2)) if match.group(2) else 0

            for other_var in dist.classes:
                if other_var == var or other_var not in assignment: continue
                asgn_b = assignment[other_var]
                ts_b = self.inst.timeslots[asgn_b.timeslot]
                other_cls = self.inst.classes[other_var]
                room_b = asgn_b.room
                
                share_days = (ts_a.days_mask & ts_b.days_mask) != 0
                share_weeks = (ts_a.weeks_mask & ts_b.weeks_mask) != 0
                ts_b_start_dist = ts_b.start_int
                ts_b_end_dist = ts_b_start_dist + other_cls.length
                overlap_time = not (ts_a_end <= ts_b_start_dist or ts_b_end_dist <= ts_a_start)
                
                if dtype == "SameRoom" and room != room_b: return True
                elif dtype == "DifferentRoom" and room == room_b and room != "NO_ROOM": return True
                elif dtype == "SameTime" and not ((ts_a_start <= ts_b_start_dist and ts_b_end_dist <= ts_a_end) or (ts_b_start_dist <= ts_a_start and ts_a_end <= ts_b_end_dist)): return True
                elif dtype == "DifferentTime" and overlap_time: return True
                elif dtype == "SameStart" and ts_a_start != ts_b_start_dist: return True
                elif dtype == "SameDays" and not (self._subset_bits(ts_a.days, ts_b.days) or self._subset_bits(ts_b.days, ts_a.days)): return True
                elif dtype == "DifferentDays" and share_days: return True
                elif dtype == "SameWeeks" and not (self._subset_bits(ts_a.weeks, ts_b.weeks) or self._subset_bits(ts_b.weeks, ts_a.weeks)): return True
                elif dtype == "DifferentWeeks" and share_weeks: return True
                elif dtype == "Overlap" and not (overlap_time and share_days and share_weeks): return True
                elif dtype == "NotOverlap" and overlap_time and share_days and share_weeks: return True
                elif dtype == "SameAttendees":
                    if overlap_time and share_days and share_weeks: return True
                    if share_days and share_weeks and room != "NO_ROOM" and room_b != "NO_ROOM":
                        trv = self.inst.rooms[room].travel.get(room_b, 0)
                        if trv > 0:
                            if ts_a_start < ts_b_start and ts_a_end + trv > ts_b_start: return True
                            if ts_b_start < ts_a_start and ts_b_end + trv > ts_a_start: return True
                elif dtype == "Precedence":
                    idx_var = dist.classes.index(var)
                    idx_other = dist.classes.index(other_var)
                    c1_w, c2_w = (ts_a.weeks, ts_b.weeks) if idx_var < idx_other else (ts_b.weeks, ts_a.weeks)
                    c1_d, c2_d = (ts_a.days, ts_b.days) if idx_var < idx_other else (ts_b.days, ts_a.days)
                    c1_end = ts_a_end if idx_var < idx_other else ts_b_end_dist
                    c2_start = ts_b_start_dist if idx_var < idx_other else ts_a_start
                    fw1, fw2 = self._first_bit(c1_w), self._first_bit(c2_w)
                    fd1, fd2 = self._first_bit(c1_d), self._first_bit(c2_d)
                    is_before = (fw1 < fw2) or (fw1 == fw2 and (fd1 < fd2 or (fd1 == fd2 and c1_end <= c2_start)))
                    if not is_before: return True
                elif dtype == "WorkDay" and share_days and share_weeks:
                    if max(ts_a_end, ts_b_end_dist) - min(ts_a_start, ts_b_start_dist) > param: return True
                elif dtype == "MinGap" and share_days and share_weeks:
                    if not (ts_a_end + param <= ts_b_start_dist or ts_b_end_dist + param <= ts_a_start): return True

        return False

    # =========================================================================
    # LÓGICA DE META-VARIABLES (INTERSECCIÓN Y EXPANSIÓN)
    # =========================================================================

    def _build_meta_domains(self) -> Dict[str, List[Tuple[str, str]]]:
        """Calcula la intersección estricta de dominios permitidos para las Meta-Variables."""
        meta_domains = {}
        for root, members in self.meta_groups.items():
            common_times = set(self.ts_list)
            for cid in members:
                cls = self.inst.classes[cid]
                if cls.allowed_times:
                    common_times.intersection_update(cls.allowed_times.keys())
            
            common_rooms = None
            room_required = False
            for cid in members:
                cls = self.inst.classes[cid]
                if cls.room_required:
                    room_required = True
                    if cls.allowed_rooms:
                        if common_rooms is None:
                            common_rooms = set(cls.allowed_rooms.keys())
                        else:
                            common_rooms.intersection_update(cls.allowed_rooms.keys())
            
            if not room_required:
                common_rooms = {"NO_ROOM"}
            elif common_rooms is None:
                common_rooms = set(self.inst.rooms.keys())

            # Construir pares válidos físicos
            max_len = max(self.inst.classes[cid].length for cid in members)
            valid_pairs = []
            for t in common_times:
                for r in common_rooms:
                    if self._room_available_for_ts(r, t, max_len):
                        valid_pairs.append((t, r))
            
            meta_domains[root] = valid_pairs
        return meta_domains

    def _expand_assignment(self, meta_assignment: Dict[str, Assignment]) -> Dict[str, Assignment]:
        """Proyecta la decisión atómica del Meta-Nodo hacia todas sus clases componentes."""
        expanded = {}
        for root, asgn in meta_assignment.items():
            for cid in self.meta_groups[root]:
                expanded[cid] = Assignment(asgn.timeslot, asgn.room)
        return expanded

    def _meta_conflicts(self, meta_var: str, ts_id: str, room: str, current_meta_asgn: Dict[str, Assignment]) -> bool:
        """Verifica la factibilidad de la Meta-Variable completa contra el entorno expandido."""
        expanded_asgn = self._expand_assignment(current_meta_asgn)
        for cid in self.meta_groups[meta_var]:
            # Utiliza tu robusto método _conflicts original para evaluar las piezas individuales
            if self._conflicts(cid, ts_id, room, expanded_asgn):
                return True
        return False

    def _get_meta_conflict_neighbors(self, meta_var: str, unassigned_meta: List[str]) -> List[Tuple[str, bool]]:
        """Extrae el vecindario del hipergrafo para propagación de Forward Checking."""
        meta_neighbors = []
        members = self.meta_groups[meta_var]
        
        # Pre-caché para velocidad O(1)
        if not hasattr(self, '_dist_cache'):
            self._dist_cache = {}
            for dist in self.inst.distributions:
                if dist.required:
                    for c in dist.classes:
                        if c not in self._dist_cache: self._dist_cache[c] = set()
                        self._dist_cache[c].update(dist.classes)
        
        for n_root in unassigned_meta:
            if n_root == meta_var: continue
            n_members = self.meta_groups[n_root]
            
            shared_students = False
            in_dist = False
            
            for c1 in members:
                cls1 = self.inst.classes[c1]
                for c2 in n_members:
                    if cls1.students & self.inst.classes[c2].students:
                        shared_students = True
                    if c2 in self._dist_cache.get(c1, set()):
                        in_dist = True
                        
            if shared_students or in_dist:
                meta_neighbors.append((n_root, shared_students))
                
        return meta_neighbors

    def _meta_forward_check(self, meta_var: str, ts: str, room: str, unassigned_meta: List[str], meta_domains: Dict[str, List]) -> Optional[Dict[str, List]]:
        """Propagación de dominios y Poda de Fiedler/Hoffman sobre Meta-Variables."""
        new_meta_domains = {v: list(d) for v, d in meta_domains.items()}
        
        # Obtenemos los vecinos inmediatos de la meta-variable recién asignada
        conflict_neighbors = self._get_meta_conflict_neighbors(meta_var, unassigned_meta)

        for (n_root, shared_students) in conflict_neighbors:
            new_domain = []
            
            # Filtro estándar de propagación (Forward Checking clásico)
            for (n_ts, n_room) in new_meta_domains[n_root]:
                if n_ts == ts and n_room == room and room != "NO_ROOM":
                    continue
                if shared_students and n_ts == ts:
                    continue
                new_domain.append((n_ts, n_room))

            # 1. Poda de Fiedler (Wipeout absoluto)
            if not new_domain:
                return None 

            # -------------------------------------------------------------
            # 2. Poda de Hoffman-Wilf Corregida para Tableros Latinos
            # -------------------------------------------------------------
            # En lugar de recalcular vecinos (O(n^2)), calculamos el tamaño 
            # del clan interno de la meta-variable actual.
            
            # ¿Cuántas clases físicas reales viven dentro de este meta-nodo?
            clan_size = len(self.meta_groups[n_root])
            unique_timeslots_left = len(set(t for t, r in new_domain))
            
            # Cota Cromática Estricta:
            # Si el meta-nodo agrupa clases (ej. SameRoom o SameTime), y el número de 
            # timeslots *únicos* disponibles cae por debajo del tamaño de clases en el 
            # propio meta-nodo (asumiendo que no todas pueden ir a la misma hora 
            # por límites de capacidad o solapamiento), es matemáticamente inviable.
            # 
            # Ajuste de relajación: Asumimos conservadoramente que pueden empaquetarse
            # de a 2 si no hay conflictos duros entre ellas.
            if unique_timeslots_left < math.ceil(clan_size / 2.0):
                # Antes de podar, confirmamos la densidad residual usando la longitud del dominio completo
                # para evitar falsos positivos en clases con múltiples aulas viables en pocos slots.
                if len(new_domain) < clan_size: 
                    return None # Backjump matemático prematuro: Colapso del politopo
            
            new_meta_domains[n_root] = new_domain

        return new_meta_domains
    
    def _timeout(self) -> bool:
        """Verifica si el CSP ha excedido el límite de tiempo asignado."""
        timeout_limit = self.cfg.CSP.get("timeout_seconds", 300)
        if timeout_limit <= 0:
            return False
        return (time.time() - self.start_time) > timeout_limit
    
    def _greedy_with_restarts(self, unassigned_meta: List[str], meta_domains: Dict[str, List]) -> Dict[str, Assignment]:
        best_meta_assignment = {}
        max_restarts = self.cfg.CSP.get("restarts", 25)
        max_backtracks_per_restart = self.cfg.CSP.get("max_backtracks", 250)

        for restart in range(max_restarts):
            if self._timeout(): break

            current_meta_assignment = {}
            current_domains = deepcopy(meta_domains)
            current_unassigned = list(unassigned_meta)
            
            random.shuffle(current_unassigned)
            backtracks_this_restart = 0
            stack = []
            
            if current_unassigned:
                var = self._select_variable(current_unassigned, current_domains)
                current_unassigned.remove(var)
                values = self._order_values(var, current_domains[var])
                stack.append((var, values, list(current_unassigned), deepcopy(current_domains)))

            while stack:
                if self._timeout() or backtracks_this_restart > max_backtracks_per_restart:
                    break

                var, values, unassigned_at_level, domains_at_level = stack.pop()
                assigned_in_this_frame = False

                for i, (ts, room) in enumerate(values):
                    self.bt_nodes += 1
                    
                    # Usamos _meta_conflicts en lugar de _conflicts
                    if not self._meta_conflicts(var, ts, room, current_meta_assignment):
                        new_doms = domains_at_level
                        if self.cfg.CSP.get("forward_checking", True):
                            # Usamos _meta_forward_check
                            new_doms = self._meta_forward_check(var, ts, room, unassigned_at_level, domains_at_level)
                            if new_doms is None:
                                continue 
                        
                        stack.append((var, values[i+1:], list(unassigned_at_level), deepcopy(domains_at_level)))
                        current_meta_assignment[var] = Assignment(timeslot=ts, room=room)
                        assigned_in_this_frame = True
                        
                        if unassigned_at_level:
                            next_var = self._select_variable(unassigned_at_level, new_doms)
                            next_unassigned = list(unassigned_at_level)
                            next_unassigned.remove(next_var)
                            next_values = self._order_values(next_var, new_doms[next_var])
                            stack.append((next_var, next_values, next_unassigned, new_doms))
                        break

                if not assigned_in_this_frame:
                    backtracks_this_restart += 1
                    if var in current_meta_assignment:
                        del current_meta_assignment[var]

            if len(current_meta_assignment) > len(best_meta_assignment):
                best_meta_assignment = deepcopy(current_meta_assignment)

            if len(best_meta_assignment) == len(unassigned_meta):
                self.logger.info(f"  ✓ Solución Meta-Estructural perfecta encontrada en reinicio {restart + 1}")
                break

        return best_meta_assignment

    def solve(self) -> Optional[Dict[str, Assignment]]:
        """Punto de entrada CSP. Ejecuta la expansión atómica de la solución."""
        self.start_time = time.time()
        
        unassigned_meta = list(self.meta_groups.keys())
        self.logger.info(f"  Contracción Algebraica: {len(self.inst.classes)} clases reducidas a {len(unassigned_meta)} Meta-Variables.")

        # Ejecutamos el motor de meta-variables
        meta_assignment = self._greedy_with_restarts(unassigned_meta, self.meta_domains)
        
        # Expansión final para Fase 5 (SA) y Exportación XML
        final_assignment = self._expand_assignment(meta_assignment)

        elapsed = time.time() - self.start_time
        assigned_count = len(final_assignment)
        total_count = len(self.inst.classes)

        if assigned_count == total_count:
            self.logger.info(f"  ✓ CSP Completo (evaluaciones: {self.bt_nodes}, t: {elapsed:.1f}s)")
            return final_assignment
        elif assigned_count > 0:
            self.logger.warning(f"  ⚠ CSP Parcial: {assigned_count}/{total_count} variables asignadas. Pasando a SA...")
            return final_assignment
        else:
            self.logger.error(f"  ✗ CSP Colapsado. 0 variables asignadas (evaluaciones: {self.bt_nodes})")
            return None


# =============================================================================
#  POST-PROCESO — SIMULATED ANNEALING
# =============================================================================

# =============================================================================
#  POST-PROCESO — SIMULATED ANNEALING CON OPERADORES TOPOLÓGICOS
# =============================================================================

class SimulatedAnnealing:
    """Optimización topológica de penalizaciones blandas usando Cadenas de Kempe y Saltos Diedrales."""

    def __init__(self, inst: Instance, cfg, csp, polytopes):
        self.inst      = inst
        self.cfg       = cfg
        self.csp       = csp
        self.polytopes = polytopes
        self.logger    = logging.getLogger("SA")
        # El reloj maestro universal para aplicar transformaciones geométricas T(t) = t + k
        self.ts_list   = list(inst.timeslots.keys())

    def _penalty(self, assignment: Dict[str, Assignment]) -> float:
        """Calcula el costo total oficial según el estándar ITC 2019."""
        w_time = self.inst.optimization.get("time", 0)
        w_room = self.inst.optimization.get("room", 0)
        w_dist = self.inst.optimization.get("distribution", 0)
        w_stud = self.inst.optimization.get("student", 0)

        p_time, p_room, p_dist, p_stud = 0.0, 0.0, 0.0, 0.0

        for cid, asgn in assignment.items():
            cls = self.inst.classes[cid]
            if asgn.timeslot in cls.allowed_times:
                p_time += cls.allowed_times[asgn.timeslot]
            if asgn.room in cls.allowed_rooms:
                p_room += cls.allowed_rooms[asgn.room]

        for dist in self.inst.distributions:
            if dist.required: continue
            dist_pen = getattr(dist, 'penalty', 1)
            violations = 0
            for i, a in enumerate(dist.classes):
                for b in dist.classes[i+1:]:
                    if a in assignment and b in assignment:
                        if not self._check_soft(dist.type, a, b, assignment[a], assignment[b]):
                            violations += 1
            p_dist += (violations * dist_pen)

        if w_stud > 0:
            student_classes = {}
            for cid, asgn in assignment.items():
                for student in self.inst.classes[cid].students:
                    if student not in student_classes:
                        student_classes[student] = []
                    student_classes[student].append((cid, asgn))
            
            for student, classes in student_classes.items():
                for i in range(len(classes)):
                    for j in range(i + 1, len(classes)):
                        c1, a1 = classes[i]
                        c2, a2 = classes[j]
                        ts1 = self.inst.timeslots[a1.timeslot]
                        ts2 = self.inst.timeslots[a2.timeslot]
                        
                        if (ts1.days_mask & ts2.days_mask) and (ts1.weeks_mask & ts2.weeks_mask):
                            s1 = ts1.start_int
                            e1 = s1 + self.inst.classes[c1].length
                            s2 = ts2.start_int
                            e2 = s2 + self.inst.classes[c2].length
                            
                            if not (e1 <= s2 or e2 <= s1):
                                p_stud += 1
                            elif a1.room != "NO_ROOM" and a2.room != "NO_ROOM":
                                req_travel = self.inst.rooms[a1.room].travel.get(a2.room, 0)
                                if req_travel > 0:
                                    delta_slots = (s2 - e1) if e1 <= s2 else (s1 - e2)
                                    if delta_slots * 5 < req_travel:
                                        p_stud += 1

        return (w_time * p_time) + (w_room * p_room) + (w_dist * p_dist) + (w_stud * p_stud)

    def _surrogate_penalty(self, assignment: Dict[str, Assignment]) -> float:
        """
        Hamiltoniano de Costos H(s).
        Calcula la energía del sistema priorizando penalizaciones estudiantiles (Travel & Conflict)
        usando las máscaras binarias de alta velocidad.
        """
        weights = self.cfg.SURROGATE_WEIGHTS
        w_conf = weights["student_conflict"]
        w_trav = weights["student_travel"]
        w_dist = weights["distribution_soft"]
        w_time = weights["time_preference"]
        w_room = weights["room_preference"]

        p_time, p_room, p_dist, p_trav, p_conf = 0.0, 0.0, 0.0, 0.0, 0.0

        for cid, asgn in assignment.items():
            cls = self.inst.classes[cid]
            if asgn.timeslot in cls.allowed_times:
                p_time += cls.allowed_times[asgn.timeslot]
            if asgn.room in cls.allowed_rooms:
                p_room += cls.allowed_rooms[asgn.room]

        for dist in self.inst.distributions:
            if dist.required: continue
            dist_pen = getattr(dist, 'penalty', 1)
            violations = 0
            for i, a in enumerate(dist.classes):
                for b in dist.classes[i+1:]:
                    if a in assignment and b in assignment:
                        if not self._check_soft(dist.type, a, b, assignment[a], assignment[b]):
                            violations += 1
            p_dist += (violations * dist_pen)

        student_classes = {}
        for cid, asgn in assignment.items():
            for student in self.inst.classes[cid].students:
                if student not in student_classes:
                    student_classes[student] = []
                student_classes[student].append((cid, asgn))
        
        for student, classes in student_classes.items():
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    c1, a1 = classes[i]
                    c2, a2 = classes[j]
                    
                    ts1 = self.inst.timeslots[a1.timeslot]
                    ts2 = self.inst.timeslots[a2.timeslot]
                    
                    if (ts1.days_mask & ts2.days_mask) and (ts1.weeks_mask & ts2.weeks_mask):
                        s1 = ts1.start_int
                        e1 = s1 + self.inst.classes[c1].length
                        s2 = ts2.start_int
                        e2 = s2 + self.inst.classes[c2].length
                        
                        if not (e1 <= s2 or e2 <= s1):
                            weeks_overlap_count = bin(ts1.weeks_mask & ts2.weeks_mask).count('1')
                            p_conf += (1.0 * weeks_overlap_count)
                        elif a1.room != "NO_ROOM" and a2.room != "NO_ROOM":
                            req_travel = self.inst.rooms[a1.room].travel.get(a2.room, 0)
                            if req_travel > 0:
                                delta_slots = (s2 - e1) if e1 <= s2 else (s1 - e2)
                                deficit = max(0, req_travel - (delta_slots * 5))
                                if deficit > 0:
                                    weeks_overlap_count = bin(ts1.weeks_mask & ts2.weeks_mask).count('1')
                                    p_trav += (deficit * weeks_overlap_count)

        return (w_time * p_time) + (w_room * p_room) + (w_dist * p_dist) + (w_trav * p_trav) + (w_conf * p_conf)

    def _check_soft(self, dist_type: str, a: str, b: str, aa: Assignment, ab: Assignment) -> bool:
        """Evaluador físico de restricciones blandas usando máscaras."""
        import re
        match = re.match(r"([a-zA-Z]+)(?:\(([^)]+)\))?", dist_type.strip())
        if not match: return True
        dtype = match.group(1)

        ts_a = self.inst.timeslots[aa.timeslot]
        ts_b = self.inst.timeslots[ab.timeslot]
        cls_a = self.inst.classes[a]
        cls_b = self.inst.classes[b]
        
        share_days = (ts_a.days_mask & ts_b.days_mask) != 0
        share_weeks = (ts_a.weeks_mask & ts_b.weeks_mask) != 0
        
        start_a, start_b = ts_a.start_int, ts_b.start_int
        end_a = start_a + cls_a.length
        end_b = start_b + cls_b.length
        overlap_time = not (end_a <= start_b or end_b <= start_a)

        if dtype == "SameTime": return ts_a.id == ts_b.id
        if dtype == "DifferentTime": return not overlap_time
        if dtype == "SameRoom": return aa.room == ab.room
        if dtype == "DifferentRoom": return aa.room != ab.room
        if dtype == "NotOverlap": return not (overlap_time and share_days and share_weeks)
        if dtype == "DifferentDays": return not share_days
        if dtype == "DifferentWeeks": return not share_weeks
        
        return True 

    def _kempe_chain_guided(self, assignment: Dict[str, Assignment]) -> Dict[str, Assignment]:
        """
        Operador Bipartito Guiado: Selecciona la semilla basándose en la 
        conductancia espectral extraída de los politopos (si está disponible).
        """
        new_asgn = deepcopy(assignment)
        valid_vars = [v for v in new_asgn.keys() if len(self.inst.classes[v].allowed_times) > 1]
        
        if not valid_vars:
            return new_asgn

        weights = []
        bias = self.cfg.SA["boundary_node_bias"]
        
        # Mapeo global de Fiedler precalculado en Fase 2 (Fallback seguro si es lista de listas)
        global_fiedler = {}
        for p in self.polytopes:
            # Verificamos si 'p' es un diccionario (tiene método 'get') o una lista
            if isinstance(p, dict) and 'fiedler_weights' in p:
                global_fiedler.update(p['fiedler_weights'])

        for v in valid_vars:
            base_w = global_fiedler.get(v, 1.0) # Si no hay pesos, asume 1.0 uniforme
            weights.append(base_w * bias if base_w > 0.8 else base_w)
            
        var_x = random.choices(valid_vars, weights=weights, k=1)[0]
        cls_x = self.inst.classes[var_x]
        t_orig = new_asgn[var_x].timeslot
        
        t_alt = random.choice(list(cls_x.allowed_times.keys()))
        if t_orig == t_alt:
            return new_asgn
            
        # --- Lógica de la Cadena de Kempe (BFS) ---
        chain = {}
        queue = [(var_x, t_alt, t_orig)]
        
        while queue:
            curr_var, target_t, source_t = queue.pop(0)
            if curr_var in chain: continue
                
            chain[curr_var] = target_t
            curr_room = new_asgn[curr_var].room
            curr_cls = self.inst.classes[curr_var]
            
            # Extracción topológica O(V) ultra-rápida usando la matriz Laplaciana de Fase 2
            topological_neighbors = []
            if curr_var in self.csp.idx:
                u_idx = self.csp.idx[curr_var]
                for other_var in new_asgn.keys():
                    if other_var != curr_var and other_var in self.csp.idx:
                        v_idx = self.csp.idx[other_var]
                        if self.csp.W[u_idx, v_idx] > 0:
                            topological_neighbors.append(other_var)
            
            for other_var, other_opt in new_asgn.items():
                if other_var == curr_var or other_var in chain: continue
                    
                other_t = other_opt.timeslot
                ts_target = self.inst.timeslots[target_t]
                ts_other = self.inst.timeslots[other_t]
                
                share_days = (ts_target.days_mask & ts_other.days_mask) != 0
                share_weeks = (ts_target.weeks_mask & ts_other.weeks_mask) != 0
                
                overlap_time = not (ts_target.start_int + curr_cls.length <= ts_other.start_int or 
                                    ts_other.start_int + self.inst.classes[other_var].length <= ts_target.start_int)
                
                overlap_full = share_days and share_weeks and overlap_time
                is_topo_neighbor = other_var in topological_neighbors
                is_room_conflict = (overlap_full and curr_room != "NO_ROOM" and curr_room == other_opt.room)
                
                if (overlap_full and is_topo_neighbor) or is_room_conflict:
                    queue.append((other_var, source_t, target_t))
                    
        # Validación
        temp_asgn = {k: Assignment(v.timeslot, v.room) for k, v in new_asgn.items() if k not in chain}
        
        for c_id, new_t in chain.items():
            if new_t not in self.inst.classes[c_id].allowed_times: return assignment 
            c_room = new_asgn[c_id].room
            if not self.csp._room_available_for_ts(c_room, new_t, self.inst.classes[c_id].length): return assignment 
            if self.csp._conflicts(c_id, new_t, c_room, temp_asgn): return assignment 
            temp_asgn[c_id] = Assignment(timeslot=new_t, room=c_room)
            
        for c_id, new_t in chain.items():
            new_asgn[c_id].timeslot = new_t
            
        return new_asgn

    def _dihedral_jump(self, assignment: Dict[str, Assignment]) -> Dict[str, Assignment]:
        """
        Operador Diedral: Aplica rotación geométrica T(t)=(t+k) mod N a un politopo latino 
        completo, trasladando el cuerpo rígido sin romper su estructura SIN interna.
        """
        valid_polys = [p for p in self.polytopes if len(p.get('class_ids', [])) > 1]
        
        if not valid_polys:
            return assignment

        poly = random.choice(valid_polys)
        c_ids = poly['class_ids']
        
        if any(cid not in assignment for cid in c_ids):
            return assignment

        new_asgn = deepcopy(assignment)
        temp_asgn = {k: Assignment(v.timeslot, v.room) for k, v in new_asgn.items() if k not in c_ids}
        
        k_shift = random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) 
        N_times = len(self.ts_list)
        
        for cid in c_ids:
            curr_ts = new_asgn[cid].timeslot
            curr_room = new_asgn[cid].room
            cls_info = self.inst.classes[cid]
            
            try:
                curr_idx = self.ts_list.index(curr_ts)
            except ValueError:
                return assignment 
                
            new_idx = (curr_idx + k_shift) % N_times
            new_ts = self.ts_list[new_idx]
            
            if new_ts not in cls_info.allowed_times:
                return assignment 
            if not self.csp._room_available_for_ts(curr_room, new_ts, cls_info.length):
                return assignment
            if self.csp._conflicts(cid, new_ts, curr_room, temp_asgn):
                return assignment
                
            temp_asgn[cid] = Assignment(timeslot=new_ts, room=curr_room)
            new_asgn[cid].timeslot = new_ts
            
        return new_asgn

    def optimize(self, assignment: Dict[str, Assignment]) -> Dict[str, Assignment]:
        sa_cfg  = self.cfg.SA
        if not sa_cfg["enabled"] or len(assignment) < 2:
            return assignment

        current = deepcopy(assignment)
        best    = deepcopy(assignment)
        T       = sa_cfg["initial_temp"]
        T_min   = sa_cfg["min_temp"]
        alpha   = sa_cfg["cooling_rate"]
        
        # Usamos el Hamiltoniano Surrogado para guiar la búsqueda internamente
        best_pen = self._surrogate_penalty(best)

        for _ in range(sa_cfg["max_iterations"]):
            if T < T_min:
                break
            
            if random.random() < sa_cfg["kempe_prob"]:
                candidate = self._kempe_chain_guided(current)
            else:
                candidate = self._dihedral_jump(current)
            
            delta = self._surrogate_penalty(candidate) - self._surrogate_penalty(current)
            if delta < 0 or random.random() < math.exp(-delta / T):
                current = candidate
                pen = self._surrogate_penalty(current)
                if pen < best_pen:
                    best     = deepcopy(current)
                    best_pen = pen
            T *= alpha

        # Al finalizar, reportamos el costo oficial para el validador
        official_penalty = self._penalty(best)
        self.logger.info(f"  SA Espectral: Energía H(s) = {best_pen:.2f} | Penalty ITC = {official_penalty:.2f}")
        return best
    
class StudentSectioningGaleShapley:
    """Fase 5: Optimización de aforos mediante emparejamientos estables."""
    
    def __init__(self, inst: Instance, cfg):
        self.inst = inst
        self.cfg = cfg
        self.logger = logging.getLogger("GaleShapley")

    def optimize_sectioning(self, assignment: Dict[str, Assignment]) -> Dict[str, Assignment]:
        if not self.cfg.SECTIONING.get("enabled", False):
            return assignment
        
        self.logger.info("  Iniciando Seccionamiento Desacoplado (Gale-Shapley)...")
        
        # 1. Mapeo de capacidades reales basadas en la asignación del CSP + Fallback
        capacities = {}
        for cid, cls in self.inst.classes.items():
            # BUG FIX: Ensure EVERY class has an entry in capacities, even if unassigned
            if cid in assignment:
                room_id = assignment[cid].room
                if room_id and room_id != "NO_ROOM" and room_id in self.inst.rooms:
                    capacities[cid] = self.inst.rooms[room_id].capacity
                else:
                    capacities[cid] = cls.limit or 999
            else:
                # Fallback para variables no asignadas por el CSP
                capacities[cid] = cls.limit or 999
                
        # 2. Identificar cursos con múltiples secciones (donde podemos mover estudiantes)
        courses_to_sections = {}
        for cid, cls in self.inst.classes.items():
            if cls.course_id not in courses_to_sections:
                courses_to_sections[cls.course_id] = []
            courses_to_sections[cls.course_id].append(cid)
            
        moves_made = 0
        
        # 3. Aplicar lógica de balanceo intra-curso
        for course_id, sections in courses_to_sections.items():
            if len(sections) < 2:
                continue 
                
            # Separar secciones saturadas de las que tienen espacio
            overfull = [s for s in sections if len(self.inst.classes[s].students) > capacities[s]]
            underfull = [s for s in sections if len(self.inst.classes[s].students) < capacities[s]]
            
            for saturated_sec in overfull:
                cls_sat = self.inst.classes[saturated_sec]
                excess = len(cls_sat.students) - capacities[saturated_sec]
                
                students_to_move = list(cls_sat.students)[:excess]
                
                for student in students_to_move:
                    best_alt = None
                    for alt_sec in underfull:
                        cls_alt = self.inst.classes[alt_sec]
                        if len(cls_alt.students) < capacities[alt_sec]:
                            # Omitimos lógica compleja de solapamiento temporal aquí por ahora
                            best_alt = alt_sec
                            break
                            
                    if best_alt:
                        self.inst.classes[saturated_sec].students.remove(student)
                        self.inst.classes[best_alt].students.add(student)
                        moves_made += 1
                        
        self.logger.info(f"  Gale-Shapley completado: {moves_made} estudiantes reasignados para cumplir aforos.")
        return assignment
# =============================================================================
#  ESCRITOR DE SOLUCIÓN XML ITC 2019
# =============================================================================

# =============================================================================
#  ESCRITOR DE SOLUCIÓN XML ITC 2019
# =============================================================================

# =============================================================================
#  ESCRITOR DE SOLUCIÓN XML ITC 2019
# =============================================================================

def write_solution(inst: Instance, assignment: Dict[str, Assignment], out_path: str):
    root = ET.Element("solution")
    root.set("name", inst.name)
    
    # Metadatos oficiales para el validador ITC 2019
    root.set("cores", "1")
    root.set("technique", "GNN & Latin Polytopes (API-Carpio)")
    root.set("author", "Octavio Eduardo González Grajeda, Juan Martín Carpio Valadez, Lucero de Montserrat Ortiz Aguilar, Héctor José Puga Soberanes, Manuel Ornelas Rodríguez")
    
    # Ordenar las clases por ID para que el XML sea legible y consistente
    for cid in sorted(assignment.keys(), key=lambda x: int(x) if x.isdigit() else x):
        asgn = assignment[cid]
        cls_el = ET.SubElement(root, "class")
        cls_el.set("id", cid)
        
        if asgn.timeslot and asgn.timeslot in inst.timeslots:
            ts = inst.timeslots[asgn.timeslot]
            cc = inst.classes[cid]
            
            # Limpiar y truncar cadenas de bits por si el parser heredó de más
            clean_days = ts.days[:inst.num_days] if ts.days else ""
            clean_weeks = ts.weeks[:inst.num_weeks] if ts.weeks else ""
            
            if clean_days and clean_weeks and ts.start:
                cls_el.set("days", clean_days)
                cls_el.set("start", str(ts.start))
                
                # Prioridad 1: Extraer longitud real del ID del slot si la tiene 
                # (útil si el parser original la mapeó así)
                if "_" in ts.id and "l" in ts.id:
                    try:
                        # Extraer de un ID tipo T_d1010100_s90_l10_w11...
                        parts = ts.id.split("_")
                        for p in parts:
                            if p.startswith("l"):
                                cls_el.set("length", p[1:])
                                break
                    except:
                        cls_el.set("length", str(cc.length))
                else:
                    # Prioridad 2: Longitud heredada de la clase
                    cls_el.set("length", str(cc.length))
                    
                cls_el.set("weeks", clean_weeks)
            else:
                cls_el.set("time", asgn.timeslot)
                
        if asgn.room and asgn.room != "NO_ROOM":
            cls_el.set("room", asgn.room)
            
        for st_id in sorted(list(inst.classes[cid].students), key=lambda x: int(x) if x.isdigit() else x):
            ET.SubElement(cls_el, "student", {"id": st_id})
            
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE solution PUBLIC "-//ITC 2019//DTD Problem Format/EN" "http://www.itc2019.org/competition-format.dtd">\n')
        xml_str = ET.tostring(root, encoding="unicode")
        f.write(xml_str)


# =============================================================================
#  ORQUESTADOR PRINCIPAL
# =============================================================================

class SolverPipeline:

    def __init__(self, cfg):
        self.cfg     = cfg
        self.logger  = logging.getLogger("Pipeline")
        self.parser  = ITC2019Parser()
        self.extractor = SpectralExtractor(cfg)
        self.oracle  = GNNOracle(cfg)

    # ------------------------------------------------------------------
    def _discover_instances(self) -> List[str]:
        inst_dir = Path(self.cfg.PATHS["instances_dir"])
        if not inst_dir.exists():
            self.logger.error(f"Directorio de instancias no existe: {inst_dir}")
            return []
        all_xml = sorted(inst_dir.glob("*.xml"))
        run_only = self.cfg.INSTANCES.get("run_only")
        exclude  = set(self.cfg.INSTANCES.get("exclude", []))

        if run_only:
            paths = [inst_dir / f"{n}.xml" for n in run_only if n not in exclude]
        else:
            paths = [p for p in all_xml if p.stem not in exclude]

        return [str(p) for p in paths if p.exists()]

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _maybe_train(self, xml_paths: List[str]):
        # NUEVO: Si es una sola instancia, guardamos los pesos con su nombre
        base_path = Path(self.cfg.PATHS["model_weights"])
        if len(xml_paths) == 1:
            inst_name = Path(xml_paths[0]).stem
            weight_path = str(base_path.parent / f"{base_path.stem}_{inst_name}{base_path.suffix}")
        else:
            weight_path = str(base_path)

        retrain = self.cfg.TRAIN.get("retrain", False)

        if not GNN_AVAILABLE:
            return
        if Path(weight_path).exists() and not retrain:
            self.logger.info(f"Pesos GNN encontrados en {weight_path}. "
                             "Saltando entrenamiento (usa retrain=True para forzar).")
            return

        self.logger.info("─── Iniciando entrenamiento GNN ───")
        random.seed(self.cfg.TRAIN["seed"])
        np.random.seed(self.cfg.TRAIN["seed"])

        random.shuffle(xml_paths)
        split = int(len(xml_paths) * self.cfg.TRAIN["train_ratio"])
        
        # FIX: Evitar que el split sea 0 si solo estamos corriendo 1 instancia
        if split == 0 and len(xml_paths) > 0:
            split = 1
            
        train_paths = xml_paths[:split]

        graphs = []
        for p in train_paths:
            try:
                inst = self.parser.parse(p)
                _, W, idx = self.extractor.build_conflict_graph(inst)
                graphs.append((inst, W, idx))
            except Exception as e:
                self.logger.warning(f"  Error parseando {p}: {e}")

        self.oracle.train(graphs, weight_path)

    # ------------------------------------------------------------------
    def _maybe_load_oracle(self, inst: Instance, W: np.ndarray,
                           idx: Dict[str, int]):
        
        # NUEVO: Buscar los pesos específicos de esta instancia
        base_path = Path(self.cfg.PATHS["model_weights"])
        weight_path = str(base_path.parent / f"{base_path.stem}_{inst.name}{base_path.suffix}")

        if not GNN_AVAILABLE or not Path(weight_path).exists():
            # Fallback a la ruta general si no existe la específica
            weight_path = str(base_path)
            if not Path(weight_path).exists():
                return
                
        if self.oracle.model is not None:
            return  # ya cargado

        in_dim  = 4  # features por nodo
        out_dim = max(len(inst.timeslots), 100) # Se ajustará dinámicamente en el load
        try:
            # FIX: Pasamos el out_dim exacto de la instancia actual
            self.oracle.load(weight_path, in_dim, len(inst.timeslots))
        except Exception as e:
            self.logger.warning(f"No se pudo cargar el modelo: {e}")

    # ------------------------------------------------------------------
    def solve_instance(self, xml_path: str) -> Dict:
        name = Path(xml_path).stem
        self.logger.info(f"\n{'═'*60}")
        self.logger.info(f"  Instancia: {name}")
        self.logger.info(f"{'═'*60}")
        t0 = time.time()

        try:
            # F1: Parser
            inst = self.parser.parse(xml_path)
            self.logger.info(f"  Clases: {len(inst.classes)} | "
                             f"Aulas: {len(inst.rooms)} | "
                             f"Timeslots: {len(inst.timeslots)}")

            # F2: Espectral
            polytopes, W, idx = self.extractor.decompose(inst)
            self.logger.info(f"  Politopos: {len(polytopes)} | "
                             f"max_size: {max(len(p) for p in polytopes)}")

            # F3: Cargar oráculo
            self._maybe_load_oracle(inst, W, idx)

            # F4: CSP
            csp    = CSPSolver(inst, self.oracle, W, idx, self.cfg)
            asgn   = csp.solve()

            if asgn is None:
                return {"instance": name, "status": "INFEASIBLE",
                        "time": time.time() - t0}

            # Post-proceso SA con Inyección del Laplaciano y los Politopos Latinos
            # NUEVO: Pasamos la variable 'polytopes' generada en la Fase 2
            sa   = SimulatedAnnealing(inst, self.cfg, csp, polytopes) 
            asgn = sa.optimize(asgn)

            # Fase 5: Seccionamiento de Estudiantes
            gs = StudentSectioningGaleShapley(inst, self.cfg)
            asgn = gs.optimize_sectioning(asgn)

            # Escribir solución
            out_path = str(Path(self.cfg.PATHS["solutions_dir"]) / f"{name}+sol.xml")
            write_solution(inst, asgn, out_path)
            self.logger.info(f"  Solución escrita: {out_path}")

            elapsed = time.time() - t0
            return {"instance": name, "status": "OK",
                    "classes": len(asgn), "time": elapsed}

        except Exception as e:
            self.logger.exception(f"  Error en {name}: {e}")
            return {"instance": name, "status": "ERROR", "error": str(e),
                    "time": time.time() - t0}

    # ------------------------------------------------------------------
    def run(self, single_instance: Optional[str] = None):
        # Setup logging
        log_path = self.cfg.PATHS.get("log_file", "solver.log")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level   = logging.INFO,
            format  = "%(asctime)s [%(name)s] %(message)s",
            handlers = [
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ]
        )
        self.logger = logging.getLogger("Pipeline")

        # Descubrir instancias
        if single_instance:
            xml_paths = [str(Path(self.cfg.PATHS["instances_dir"]) /
                             f"{single_instance}.xml")]
        else:
            xml_paths = self._discover_instances()

        if not xml_paths:
            self.logger.error("Sin instancias para procesar.")
            return

        self.logger.info(f"Instancias a procesar: {len(xml_paths)}")

        # Entrenamiento GNN (si aplica)
        self._maybe_train(xml_paths)

        # Resolver
        results = []
        for path in xml_paths:
            r = self.solve_instance(path)
            results.append(r)

        # Reporte final
        self.logger.info(f"\n{'═'*60}")
        self.logger.info("  RESUMEN")
        self.logger.info(f"{'═'*60}")
        ok  = sum(1 for r in results if r["status"] == "OK")
        inf = sum(1 for r in results if r["status"] == "INFEASIBLE")
        err = sum(1 for r in results if r["status"] == "ERROR")
        self.logger.info(f"  OK={ok}  INFEASIBLE={inf}  ERROR={err}  "
                         f"TOTAL={len(results)}")

        # CSV de métricas
        if self.cfg.OUTPUT.get("save_metrics_csv"):
            csv_path = self.cfg.OUTPUT["metrics_csv_path"]
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["instance","status","classes","time","error"])
                w.writeheader()
                for r in results:
                    w.writerow({k: r.get(k, "") for k in w.fieldnames})
            self.logger.info(f"  Métricas CSV: {csv_path}")


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solver ITC 2019 — API-Carpio (GNN + Backtracking + SA)"
    )
    parser.add_argument(
        "--config",
        default = "config.py",
        help    = "Ruta al archivo de configuración (default: config.py)",
    )
    parser.add_argument(
        "--instance",
        default = None,
        help    = "Nombre de una sola instancia a resolver (sin .xml)",
    )
    parser.add_argument(
        "--retrain",
        action  = "store_true",
        help    = "Forzar reentrenamiento de la GNN aunque existan pesos guardados",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.retrain:
        cfg.TRAIN["retrain"] = True

    pipeline = SolverPipeline(cfg)
    pipeline.run(single_instance=args.instance)