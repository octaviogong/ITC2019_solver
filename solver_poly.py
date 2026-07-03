# -*- coding: utf-8 -*-
"""
=============================================================================
  SOLVER UNIFICADO ITC 2019 — API-Carpio
  Fases: Parser → Topología Supernodal → Geometría PILS → Relajación LP (Packing)
         (CON CLIQUE INJECTION) → Cortes Poliédricos → CSP Redondeo Dependiente → SA
=============================================================================
"""

import argparse
import csv
import importlib.util
import logging
import math
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from collections import deque, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, NamedTuple

import numpy as np

# ── SciPy (única dependencia numérica: el oráculo estocástico es el LP x*) ──
try:
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh
    from scipy.optimize import linprog
    from scipy.cluster.vq import kmeans2
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    linprog = None
    print("[AVISO] SciPy no disponible. Relajación de empaquetamiento y clustering espectral desactivados.")

def load_config(path: str = "config.py"):
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
    days_mask: int = 0
    weeks_mask: int = 0
    start_int: int = 0
    length: int = 0

@dataclass
class Room:
    id: str
    capacity: int = 0
    features: Set[str] = field(default_factory=set)
    unavailable: List[Dict[str, any]] = field(default_factory=list)
    travel: Dict[str, int] = field(default_factory=dict)

@dataclass
class CourseClass:
    id: str
    course_id: str
    config_id: str
    subpart_id: str = ""
    limit: int = 0
    length: int = 1
    parent: str = ""
    room_required: bool = True
    allowed_rooms: Dict[str, int] = field(default_factory=dict)
    allowed_times: Dict[str, int] = field(default_factory=dict)
    students: Set[str] = field(default_factory=set)

@dataclass
class Distribution:
    id: str
    type: str
    required: bool = True
    penalty: int = 1
    classes: List[str] = field(default_factory=list)
    dtype: str = field(default="", repr=False)

@dataclass
class Instance:
    name: str
    optimization: Dict[str, int] = field(default_factory=dict)
    timeslots: Dict[str, TimeSlot] = field(default_factory=dict)
    rooms: Dict[str, Room] = field(default_factory=dict)
    classes: Dict[str, CourseClass] = field(default_factory=dict)
    distributions: List[Distribution] = field(default_factory=list)
    # student_id -> lista de course_ids en que está inscrito (del bloque <students>)
    students: Dict[str, List[str]] = field(default_factory=dict)
    num_weeks: int = 1
    num_days: int = 5
    slots_per_day: int = 10

class Assignment:
    __slots__ = ["timeslot", "room"]
    def __init__(self, timeslot: Optional[str] = None, room: Optional[str] = None):
        self.timeslot = timeslot
        self.room     = room

def slot_len(inst: "Instance", cid: str, ts_id: Optional[str]) -> int:
    """Longitud REAL de la clase en el timeslot asignado. En ITC la longitud es
    propiedad del timeslot (MWF 50min vs TR 75min), no de la clase; se usa la del
    timeslot y, si éste no la trae (tiempos heredados del subpart), la de la clase."""
    ts = inst.timeslots.get(ts_id) if ts_id else None
    if ts is not None and ts.length > 0:
        return ts.length
    c = inst.classes.get(cid)
    return c.length if c else 1

# =============================================================================
#  SIMETRÍAS REALES DE LA INSTANCIA (Symmetry Breaking)
#  Detección de objetos matemáticamente intercambiables. Intercambiar dos
#  gemelas produce soluciones equivalentes en costo y factibilidad, por lo que
#  imponer un orden lexicográfico entre ellas poda ramas redundantes SIN perder
#  soluciones (cualquier solución se puede permutar a su forma canónica).
# =============================================================================

def detect_class_twins(inst: "Instance") -> List[List[str]]:
    """
    Grupos de clases GEMELAS: mismo course/config/subpart y parent, mismos
    dominios de tiempo y aula (incluidas penalizaciones), mismo limit/length,
    misma membresía en distribuciones y mismos estudiantes. Se excluyen clases
    que son padre de otra y las que participan en 'Precedence' (orden-sensible),
    porque intercambiarlas NO sería neutro.
    """
    parents: Set[str] = {c.parent for c in inst.classes.values() if c.parent}
    order_sensitive: Set[str] = set()
    dist_of: Dict[str, Set[int]] = defaultdict(set)
    for k, d in enumerate(inst.distributions):
        for c in d.classes:
            dist_of[c].add(k)
            if d.dtype == "Precedence":
                order_sensitive.add(c)
    groups: Dict[Tuple, List[str]] = defaultdict(list)
    for cid, cls in inst.classes.items():
        if cid in parents or cid in order_sensitive:
            continue
        sig = (cls.course_id, cls.config_id, cls.subpart_id, cls.parent,
               cls.room_required, cls.limit, cls.length,
               frozenset(cls.allowed_times.items()),
               frozenset(cls.allowed_rooms.items()),
               frozenset(cls.students),
               frozenset(dist_of.get(cid, ())))
        groups[sig].append(cid)
    def _k(x: str):
        return (0, int(x)) if x.isdigit() else (1, x)
    return [sorted(m, key=_k) for m in groups.values() if len(m) > 1]


def detect_twin_rooms(inst: "Instance") -> List[List[str]]:
    """
    Grupos de aulas GEMELAS: misma capacidad, mismos features, mismos bloques de
    indisponibilidad y misma matriz de traslados hacia el resto (ignorando el
    traslado entre las propias candidatas, que es simétrico). Las gemelas solo
    son intercambiables GLOBALMENTE, así que se usan para canonicalizar el orden
    de exploración, no para podar localmente.
    """
    pre: Dict[Tuple, List[str]] = defaultdict(list)
    for rid, rm in inst.rooms.items():
        unav = tuple(sorted((b["days_mask"], b["weeks_mask"], b["start"], b["length"])
                            for b in rm.unavailable))
        pre[(rm.capacity, frozenset(rm.features), unav)].append(rid)

    def _twin_pair(r1: str, r2: str) -> bool:
        t1 = {k: v for k, v in inst.rooms[r1].travel.items() if k != r2}
        t2 = {k: v for k, v in inst.rooms[r2].travel.items() if k != r1}
        return t1 == t2

    def _k(x: str):
        return (0, int(x)) if x.isdigit() else (1, x)

    out: List[List[str]] = []
    for cand in pre.values():
        if len(cand) < 2:
            continue
        cand = sorted(cand, key=_k)
        used: Set[str] = set()
        for i, r in enumerate(cand):
            if r in used:
                continue
            grp = [r]
            for s in cand[i + 1:]:
                if s not in used and _twin_pair(r, s) and all(_twin_pair(g, s) for g in grp[1:]):
                    grp.append(s)
            if len(grp) > 1:
                out.append(grp)
                used.update(grp)
    return out


# =============================================================================
#  FASE 1 — PARSER XML ITC 2019
# =============================================================================

class ITC2019Parser:
    def _parse_time(self, tm, inst, time_map):
        t_id = tm.get("id")
        t_days = tm.get("days", "0")
        t_start = tm.get("start", "0")
        t_weeks = tm.get("weeks", "0")
        t_len   = int(tm.get("length")) if tm.get("length") else 0

        d_mask = int(t_days.ljust(7, '0'), 2) if t_days else 0
        w_mask = int(t_weeks.ljust(16, '0'), 2) if t_weeks else 0
        s_int  = int(t_start) if t_start else 0

        if not t_id:
            # La longitud forma parte de la IDENTIDAD del timeslot: en ITC un mismo
            # (days,start,weeks) puede tener longitudes distintas según la opción
            # (p.ej. MWF 50min vs TR 75min). Guardarla en el id evita colisiones y
            # que se use una longitud equivocada al calcular solapes.
            t_key = (t_days, t_start, t_weeks, t_len)
            if t_key not in time_map:
                t_id = f"T_d{t_days}_s{t_start}_l{t_len}_w{t_weeks}"
                time_map[t_key] = t_id
                inst.timeslots[t_id] = TimeSlot(
                    id=t_id, days=t_days, start=t_start, weeks=t_weeks,
                    days_mask=d_mask, weeks_mask=w_mask, start_int=s_int, length=t_len
                )
            else:
                t_id = time_map[t_key]
        else:
            if t_id not in inst.timeslots:
                inst.timeslots[t_id] = TimeSlot(
                    id=t_id, days=t_days, start=t_start, weeks=t_weeks,
                    days_mask=d_mask, weeks_mask=w_mask, start_int=s_int, length=t_len
                )
        return t_id

    def parse(self, xml_path: str) -> Instance:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        inst = Instance(name=Path(xml_path).stem)

        inst.num_weeks     = int(root.get("nrWeeks", 1))
        inst.num_days      = int(root.get("nrDays",  5))
        inst.slots_per_day = int(root.get("slotsPerDay", 10))

        opt = root.find("optimization")
        if opt is not None:
            for attr in ["time", "room", "distribution", "student"]:
                inst.optimization[attr] = int(opt.get(attr, 0))
        
        time_map = {}

        rooms_container = root.find("rooms")
        for r in (rooms_container.findall("room") if rooms_container is not None else []):
            rm = Room(id=r.get("id", ""), capacity=int(r.get("capacity", 0)))
            for feat in r.findall("feature"):
                rm.features.add(feat.get("id", ""))
            for una in r.findall("unavailable"):
                d_str = una.get("days", "0")
                w_str = una.get("weeks", "0")
                rm.unavailable.append({
                    "days_mask": int(d_str.ljust(7, '0'), 2) if d_str else 0,
                    "weeks_mask": int(w_str.ljust(16, '0'), 2) if w_str else 0,
                    "start": int(una.get("start", "0")),
                    "length": int(una.get("length", "0"))
                })
            for trv in r.findall("travel"):
                dest_id = trv.get("room", "")
                val = int(trv.get("value", "0"))
                rm.travel[dest_id] = val
            inst.rooms[rm.id] = rm

        for room_id, rm in inst.rooms.items():
            for dest_id, val in list(rm.travel.items()):
                if dest_id in inst.rooms:
                    if room_id not in inst.rooms[dest_id].travel:
                        inst.rooms[dest_id].travel[room_id] = val

        for course in root.iter("course"):
            course_id = course.get("id", "")
            for cfg in course.findall("config"):
                config_id = cfg.get("id", "")
                for subpart in cfg.findall("subpart"):
                    subpart_id = subpart.get("id", "")
                    subpart_times = [self._parse_time(tm, inst, time_map) for tm in subpart.findall("time")]
                    subpart_rooms = [rm.get("id", "") for rm in subpart.findall("room")]
                    subpart_length = int(subpart.get("length", 0))

                    for cls in subpart.findall("class"):
                        cls_own_times = cls.findall("time")
                        if cls_own_times:
                            cls_length = 0
                            for tm in cls_own_times:
                                if tm.get("length"):
                                    cls_length = int(tm.get("length"))
                                    break
                            if cls_length == 0:
                                cls_length = int(cls.get("length", subpart_length or 1))
                        else:
                            cls_length = int(cls.get("length", subpart_length or 1))

                        cc = CourseClass(
                            id=cls.get("id", ""), course_id=course_id, config_id=config_id,
                            subpart_id=subpart_id,
                            limit=int(cls.get("limit", 0)), length=cls_length, parent=cls.get("parent", "")
                        )
                        cls_times = {self._parse_time(tm, inst, time_map): int(tm.get("penalty", "0")) for tm in cls_own_times}
                        cc.allowed_times = cls_times if cls_times else {t: 0 for t in subpart_times}
                        cls_rooms = {rm.get("id", ""): int(rm.get("penalty", "0")) for rm in cls.findall("room")}
                        cc.allowed_rooms = cls_rooms if cls_rooms else {r: 0 for r in subpart_rooms}

                        if cls.get("room", "true").lower() == "false":
                            cc.room_required = False
                            cc.allowed_rooms = {}
                        else:
                            cc.room_required = len(cc.allowed_rooms) > 0

                        for st in cls.findall("student"): cc.students.add(st.get("id", ""))
                        inst.classes[cc.id] = cc

        for dist in root.iter("distribution"):
            dist_type_raw = dist.get("type", "")
            _m = re.match(r"([a-zA-Z]+)", dist_type_raw.strip())
            d = Distribution(
                id       = dist.get("id", ""),
                type     = dist_type_raw,
                required = dist.get("required", "false").lower() == "true",
                penalty  = int(dist.get("penalty", "1")),
                dtype    = _m.group(1) if _m else ""
            )
            for c in dist.findall("class"):
                d.classes.append(c.get("id", ""))
            inst.distributions.append(d)

        # ── ESTUDIANTES ──────────────────────────────────────────────────
        # En ITC 2019 los estudiantes NO son hijos de <class>: vienen en un
        # bloque <students>, cada <student> con los <course> en que se inscribe.
        # El solver debe seccionarlos luego en clases concretas.
        students_container = root.find("students")
        if students_container is not None:
            for st in students_container.findall("student"):
                sid = st.get("id", "")
                courses = [c.get("id", "") for c in st.findall("course")]
                inst.students[sid] = courses

        return inst

# =============================================================================
#  FASE 2 — EXTRACTOR ESPECTRAL BASE
# =============================================================================

class SpectralExtractor:
    """
    Construye dos grafos distintos sobre las clases:
      (1) build_conflict_graph -> W_conf DURO (estudiantes compartidos + distribuciones
          duras). Es el que consume el CSP/LP/Kempe: solo codifica SEPARACIONES reales.
      (2) build_multigraph -> W_multi de AFINIDAD (multigrafo de 4 capas). Solo para
          clustering espectral; agrupar por afinidad no implica separar.
    decompose() hace clustering espectral REAL sobre W_multi (Laplaciana normalizada +
    eigsh + K-Means) y devuelve micro-politopos cohesivos de tamaño acotado, matando el
    "politopo gigante" que dejaba inoperante al SA.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.logger = logging.getLogger("SpectralExtractor")
        spec: Dict[str, Any] = getattr(cfg, "SPECTRAL", {}) if cfg is not None else {}
        self.target_size: int = int(spec.get("polytope_target_size", 6))
        self.max_size: int = int(spec.get("polytope_max_size", 8))
        self.max_eigs: int = int(spec.get("max_eigenvectors", 200))
        self.spectral_max_n: int = int(spec.get("spectral_max_n", 2500))
        self.scarce_room_threshold: int = int(spec.get("scarce_room_threshold", 3))

    # ---- (1) Grafo de conflicto DURO (para CSP/LP: solo separaciones reales) ----
    def build_conflict_graph(self, inst: Instance) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
        class_ids: List[str] = list(inst.classes.keys())
        n: int = len(class_ids)
        idx: Dict[str, int] = {cid: i for i, cid in enumerate(class_ids)}
        W: np.ndarray = np.zeros((n, n), dtype=np.float32)

        # Estudiantes compartidos vía índice invertido (evita el doble bucle O(N^2)).
        student_classes: Dict[str, List[int]] = defaultdict(list)
        for cid in class_ids:
            for s in inst.classes[cid].students:
                student_classes[s].append(idx[cid])
        for cs in student_classes.values():
            for a in range(len(cs)):
                ia = cs[a]
                for b in range(a + 1, len(cs)):
                    ib = cs[b]
                    W[ia, ib] += 1.0
                    W[ib, ia] += 1.0

        # Distribuciones duras => arista (co-restricción que exige separación/atención).
        for dist in inst.distributions:
            if not dist.required:
                continue
            members = [idx[c] for c in dist.classes if c in idx]
            for a in range(len(members)):
                ia = members[a]
                for b in range(a + 1, len(members)):
                    ib = members[b]
                    if W[ia, ib] < 1.0:
                        W[ia, ib] = 1.0
                        W[ib, ia] = 1.0
        return class_ids, W, idx

    # ---- (2) MULTIGRAFO de afinidad de 4 capas (para clustering espectral) ----
    def build_multigraph(self, inst: Instance, class_ids: List[str], idx: Dict[str, int]):
        n: int = len(class_ids)
        acc: Dict[Tuple[int, int], float] = defaultdict(float)

        def add(i: int, j: int, w: float) -> None:
            if i == j:
                return
            acc[(i, j) if i < j else (j, i)] += w

        # Capa 1 — Estudiantes (grafo bipartito proyectado): estudiantes_compartidos × 10.
        student_classes: Dict[str, List[int]] = defaultdict(list)
        for cid in class_ids:
            for s in inst.classes[cid].students:
                student_classes[s].append(idx[cid])
        for cs in student_classes.values():
            for a in range(len(cs)):
                for b in range(a + 1, len(cs)):
                    add(cs[a], cs[b], 10.0)

        # Capa 2 — Distribuciones duras (hipergrafo): +50 en SameAttendees/NotOverlap/SameTime.
        for d in inst.distributions:
            if d.required and d.dtype in ("SameAttendees", "NotOverlap", "SameTime"):
                m = [idx[c] for c in d.classes if c in idx]
                for a in range(len(m)):
                    for b in range(a + 1, len(m)):
                        add(m[a], m[b], 50.0)

        # Capa 3 — Curso / jerarquía: +5 si comparten course_id o relación parent-child.
        course_classes: Dict[str, List[int]] = defaultdict(list)
        for cid in class_ids:
            course_classes[inst.classes[cid].course_id].append(idx[cid])
        for cs in course_classes.values():
            for a in range(len(cs)):
                for b in range(a + 1, len(cs)):
                    add(cs[a], cs[b], 5.0)
        for cid in class_ids:
            par = inst.classes[cid].parent
            if par and par in idx:
                add(idx[cid], idx[par], 5.0)

        # Capa 4 — Competencia de salas escasas: +2 entre clases que solo caben en un
        # conjunto reducido de aulas y comparten alguna (proxy de laboratorio escaso).
        room_constrained: Dict[str, List[int]] = defaultdict(list)
        for cid in class_ids:
            cls = inst.classes[cid]
            if cls.room_required and 0 < len(cls.allowed_rooms) <= self.scarce_room_threshold:
                for r in cls.allowed_rooms:
                    room_constrained[r].append(idx[cid])
        for cs in room_constrained.values():
            u = sorted(set(cs))
            for a in range(len(u)):
                for b in range(a + 1, len(u)):
                    add(u[a], u[b], 2.0)

        if not acc:
            return csr_matrix((n, n), dtype=np.float64)
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        for (a, b), w in acc.items():
            rows += [a, b]
            cols += [b, a]
            data += [w, w]
        return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

    # ---- Emisión de politopos (trocea clusters grandes a <= max_size) ----
    def _emit(self, polytopes: List[Dict[str, Any]], groups: List[List[int]], class_ids: List[str]) -> None:
        for members in groups:
            if not members:
                continue
            for s in range(0, len(members), self.max_size):
                chunk = members[s:s + self.max_size]
                cids = [class_ids[c] for c in chunk]
                polytopes.append({"class_ids": cids, "fiedler_weights": {c: 1.0 for c in cids}})

    def _components_fallback(self, class_ids: List[str], W_multi) -> List[Dict[str, Any]]:
        # Componentes conexas + troceo (para N grande o si eigsh falla). El troceo es lo
        # que evita el politopo gigante aun sin espectral.
        n = len(class_ids)
        adj: Dict[int, List[int]] = defaultdict(list)
        coo = W_multi.tocoo()
        for i, j in zip(coo.row.tolist(), coo.col.tolist()):
            if i < j:
                adj[i].append(j)
                adj[j].append(i)
        seen = [False] * n
        groups: List[List[int]] = []
        for s in range(n):
            if seen[s]:
                continue
            comp: List[int] = []
            stack = [s]
            seen[s] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            groups.append(comp)
        polytopes: List[Dict[str, Any]] = []
        self._emit(polytopes, groups, class_ids)
        return polytopes

    # ---- Clustering espectral real (Laplaciana normalizada + eigsh + K-Means) ----
    def _spectral_clusters(self, class_ids: List[str], W_multi) -> List[Dict[str, Any]]:
        n = len(class_ids)
        deg = np.asarray(W_multi.sum(axis=1)).ravel()
        connected = np.where(deg > 0)[0]
        isolated = np.where(deg <= 0)[0]

        polytopes: List[Dict[str, Any]] = []
        for i in isolated.tolist():
            polytopes.append({"class_ids": [class_ids[i]], "fiedler_weights": {class_ids[i]: 1.0}})

        m = int(len(connected))
        if m == 0:
            return polytopes
        if m <= self.max_size:
            self._emit(polytopes, [connected.tolist()], class_ids)
            return polytopes

        sub = W_multi[connected][:, connected].astype(np.float64)
        d = np.asarray(sub.sum(axis=1)).ravel()
        d_inv_sqrt = 1.0 / np.sqrt(d)
        Dm = diags(d_inv_sqrt)
        # N_adj = D^-1/2 W D^-1/2. La Laplaciana normalizada es L = I - N_adj, así que los
        # valores propios MÁS PEQUEÑOS de L (excl. 0) son los MÁS GRANDES de N_adj: se
        # extraen con which='LA' (estable y rápido, evita el 'SM' sobre L singular).
        N_adj = Dm @ sub @ Dm
        K = max(2, math.ceil(m / self.target_size))
        n_comp = int(max(1, min(K, self.max_eigs, m - 2)))
        try:
            _, vecs = eigsh(N_adj, k=n_comp, which="LA")
        except Exception as e:
            self.logger.warning(f"  eigsh no convergió ({e}); fallback a componentes+troceo.")
            return self._components_fallback(class_ids, W_multi)

        # Normalización de filas (Ng–Jordan–Weiss) antes de K-Means.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = vecs / norms

        k_clusters = int(min(K, m))
        seed = int(getattr(self.cfg, "TRAIN", {}).get("seed", 66)) if self.cfg is not None else 66
        np.random.seed(seed)
        try:
            _, labels = kmeans2(emb, k_clusters, minit="++", iter=25, missing="warn")
        except Exception:
            _, labels = kmeans2(emb, k_clusters, minit="random", iter=25, missing="warn")

        clusters: Dict[int, List[int]] = defaultdict(list)
        for local_i, lab in enumerate(labels.tolist()):
            clusters[int(lab)].append(int(connected[local_i]))
        self._emit(polytopes, list(clusters.values()), class_ids)
        return polytopes

    def decompose(self, inst: Instance) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, int]]:
        class_ids, W_conf, idx = self.build_conflict_graph(inst)
        n = len(class_ids)
        if n == 0:
            return [], W_conf, idx
        if not SCIPY_AVAILABLE or n <= self.max_size:
            return ([{"class_ids": class_ids, "fiedler_weights": {c: 1.0 for c in class_ids}}], W_conf, idx)

        W_multi = self.build_multigraph(inst, class_ids, idx)
        if n > self.spectral_max_n:
            self.logger.info(f"  [Espectral] N={n} > spectral_max_n={self.spectral_max_n}: "
                             f"componentes+troceo (eficiencia).")
            polys = self._components_fallback(class_ids, W_multi)
        else:
            polys = self._spectral_clusters(class_ids, W_multi)
        return polys, W_conf, idx

# =============================================================================
#  PASO 1 — ANÁLISIS TOPOLÓGICO Y FORMULACIÓN SUPERNODAL
# =============================================================================

class SupernodeTransformer:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("SupernodalPipeline")
        
    def extract_supernodes(
        self, 
        num_classes: int, 
        conflict_matrix: np.ndarray, 
        allowed_slots: Dict[int, Set[int]]
    ) -> Tuple[Dict[int, List[int]], np.ndarray, Dict[int, int], Dict[int, Set[int]]]:
        
        node_signatures = {}
        for u in range(num_classes):
            neighbors = tuple(np.where(conflict_matrix[u] == 1)[0])
            closed_neighborhood = tuple(sorted(set(neighbors) | {u}))
            slots_signature = tuple(sorted(allowed_slots.get(u, set())))
            node_signatures[u] = (closed_neighborhood, slots_signature)
            
        signature_groups = defaultdict(list)
        for u, signature in node_signatures.items():
            signature_groups[signature].append(u)
            
        supernodes = {}
        supernode_weights = {}
        super_allowed_slots = {}
        class_to_supernode = {}
        
        for s_id, (signature, classes) in enumerate(signature_groups.items()):
            supernodes[s_id] = classes
            supernode_weights[s_id] = len(classes)
            super_allowed_slots[s_id] = allowed_slots.get(classes[0], set())
            for c in classes:
                class_to_supernode[c] = s_id
                
        num_supernodes = len(supernodes)
        
        super_conflict_matrix = np.zeros((num_supernodes, num_supernodes), dtype=int)
        for u in range(num_classes):
            for v in range(num_classes):
                if conflict_matrix[u, v] == 1:
                    s_u = class_to_supernode[u]
                    s_v = class_to_supernode[v]
                    if s_u != s_v:
                        super_conflict_matrix[s_u, s_v] = 1
                        super_conflict_matrix[s_v, s_u] = 1
                        
        return supernodes, super_conflict_matrix, supernode_weights, super_allowed_slots

# =============================================================================
#  PASO 2 — MODELADO GEOMÉTRICO MEDIANTE PILS
# =============================================================================

class Asterism(NamedTuple):
    name: str
    points: Set[Tuple[int, int]]

class LatinBoard:
    def __init__(self, size: int):
        self.size = size
        self.points: List[Tuple[int, int]] = [(r, c) for r in range(size) for c in range(size)]
        self.asterisms: Dict[str, Asterism] = {}

    def add_asterism(self, name: str, points: Set[Tuple[int, int]]):
        self.asterisms[name] = Asterism(name=name, points=points)

class PILSGeometryManager:
    def __init__(self, total_order: int, supernode_sizes: List[int], logger: Optional[logging.Logger] = None):
        self.order = total_order
        self.partitions = supernode_sizes
        self.logger = logger or logging.getLogger("PILSGeometry")

    def validate_rational_outline_existence(self) -> bool:
        sum_partitions = sum(self.partitions)
        if sum_partitions > self.order:
            return False
        for idx, sub_size in enumerate(self.partitions):
            if sub_size > self.order // 2:
                return False
        return True

    def build_pils_orthogonal_mask(self) -> np.ndarray:
        pils_mask = np.ones((self.order, self.order), dtype=bool)
        current_offset = 0
        for sub_size in self.partitions:
            end_offset = current_offset + sub_size
            if end_offset <= self.order:
                pils_mask[current_offset:end_offset, current_offset:end_offset] = False
            current_offset = end_offset
        return pils_mask

# =============================================================================
#  PASO 3 — RELAJACIÓN LP (SET PACKING CON FACETAS ESTRICTAS DE CLIQUE)
# =============================================================================

# =============================================================================
#  PASO 3 — RELAJACIÓN LP (SET PACKING CON SPARSITY Y CLIQUE REDUNDANCY PRUNING)
# =============================================================================

class StochasticPackingRounder:
    def __init__(self, num_slots: int, logger: Optional[logging.Logger] = None):
        self.num_slots = num_slots
        self.logger = logger or logging.getLogger("StochasticRounder")
        if linprog is None:
            self.logger.error("SciPy es requerido para la relajación lineal (linprog).")

    def _build_lp_relaxation(
        self, 
        num_supernodes: int, 
        super_conflict_matrix: np.ndarray, 
        super_allowed_slots: Dict[int, Set[int]],
        strict_cliques: Set[frozenset],
        super_weights: Dict[int, int]
    ) -> Tuple[np.ndarray, Optional[Any], Optional[np.ndarray], Optional[Any], Optional[np.ndarray], List[Tuple[float, float]]]:
        num_vars = num_supernodes * self.num_slots
        c = np.full(num_vars, -1.0) 
        
        # --- USO DE MATRICES DISPERSAS (Evita MemoryError) ---
        A_eq_row, A_eq_col, A_eq_data = [], [], []
        b_eq = []
        eq_idx = 0
        
        A_ub_row, A_ub_col, A_ub_data = [], [], []
        b_ub = []
        ub_idx = 0
        
        # 1. Asignación exacta proporcional al tamaño del Supernodo
        for i in range(num_supernodes):
            for j in range(self.num_slots):
                A_eq_row.append(eq_idx)
                A_eq_col.append(i * self.num_slots + j)
                A_eq_data.append(1.0)
            b_eq.append(float(super_weights[i]))
            eq_idx += 1
            
        # 2. INYECCIÓN ESTRICTA DE FACETAS DE CLIQUE
        covered_pairs = set() # Poda geométrica de redundancia
        cliques_added = 0
        
        for clique in strict_cliques:
            if len(clique) < 2: continue
            
            # Registrar los pares cubiertos para no duplicarlos
            c_list = list(clique)
            for _i in range(len(c_list)):
                for _k in range(_i + 1, len(c_list)):
                    u, v = min(c_list[_i], c_list[_k]), max(c_list[_i], c_list[_k])
                    covered_pairs.add((u, v))
            
            for j in range(self.num_slots):
                for i in clique:
                    A_ub_row.append(ub_idx)
                    A_ub_col.append(i * self.num_slots + j)
                    A_ub_data.append(1.0)
                b_ub.append(1.0)
                ub_idx += 1
            cliques_added += 1
            
        self.logger.info(f"  Inyectados {cliques_added} Hiperplanos de Clique duro en el LP.")

        # 3. Restricciones Pairwise (Solo conflictos NO cubiertos)
        pairwise_added = 0
        for i in range(num_supernodes):
            for k in range(i + 1, num_supernodes):
                if super_conflict_matrix[i, k] == 1 and (i, k) not in covered_pairs:
                    for j in range(self.num_slots):
                        A_ub_row.append(ub_idx)
                        A_ub_col.append(i * self.num_slots + j)
                        A_ub_data.append(1.0)
                        
                        A_ub_row.append(ub_idx)
                        A_ub_col.append(k * self.num_slots + j)
                        A_ub_data.append(1.0)
                        
                        b_ub.append(1.0)
                        ub_idx += 1
                    pairwise_added += 1
                    
        self.logger.info(f"  Añadidas {pairwise_added} aristas pairwise (Redundancia evitada: {len(covered_pairs)} pares).")

        # 4. Límites (Bounds)
        bounds = [(0.0, 1.0) for _ in range(num_vars)]
        for i in range(num_supernodes):
            allowed = super_allowed_slots.get(i, set(range(self.num_slots)))
            for j in range(self.num_slots):
                if j not in allowed:
                    bounds[i * self.num_slots + j] = (0.0, 0.0) 
                    
        # 5. Compilación final a Formato CSR
        A_eq_sparse = csr_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(eq_idx, num_vars)) if eq_idx > 0 else None
        A_ub_sparse = csr_matrix((A_ub_data, (A_ub_row, A_ub_col)), shape=(ub_idx, num_vars)) if ub_idx > 0 else None
                    
        return (c, A_eq_sparse, np.array(b_eq) if b_eq else None, A_ub_sparse, np.array(b_ub) if b_ub else None, bounds)

    def solve_fractional_landscape(
        self, 
        num_supernodes: int, 
        super_conflict_matrix: np.ndarray, 
        super_allowed_slots: Dict[int, Set[int]],
        strict_cliques: Set[frozenset],
        super_weights: Dict[int, int]
    ) -> np.ndarray:
        self.logger.info("Construyendo politopo de relajación de empaquetamiento (LP)...")
        c, A_eq, b_eq, A_ub, b_ub, bounds = self._build_lp_relaxation(
            num_supernodes, super_conflict_matrix, super_allowed_slots, strict_cliques, super_weights
        )
        
        self.logger.info("Resolviendo LP continuo mediante matrices dispersas (Método HiGHS)...")
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not res.success:
            self.logger.warning(f"Fallo en LP fraccional: {res.message}. Retornando distribución uniforme.")
            fallback = np.zeros((num_supernodes, self.num_slots))
            for i in range(num_supernodes):
                allowed = list(super_allowed_slots.get(i, set(range(self.num_slots))))
                if allowed: fallback[i, allowed] = 1.0 / len(allowed)
            return fallback

        self.logger.info("Relajación lineal convergida con éxito. Probabilidades extraídas.")
        return res.x.reshape((num_supernodes, self.num_slots))

# =============================================================================
#  PASO 4 — SEPARACIÓN POLIÉDRICA (BRANCH AND CUT)
# =============================================================================

class PolyhedralCutSeparator:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("PolyhedralCuts")

    def separate_wheel_inequalities(self, super_conflict_matrix: np.ndarray, x_star: np.ndarray, tolerance: float = 1e-4) -> List[Tuple[int, int, List[int]]]:
        num_supernodes, num_slots = x_star.shape
        violated_cuts = []
        for j in range(num_slots):
            vals_in_slot = x_star[:, j]
            active_nodes = np.where(vals_in_slot > tolerance)[0]
            if len(active_nodes) < 6: continue
            for hub in active_nodes:
                hub_neighbors = [v for v in active_nodes if super_conflict_matrix[hub, v] == 1]
                if len(hub_neighbors) < 5: continue
                np.random.shuffle(hub_neighbors)
                C = hub_neighbors[:5] 
                is_hole = True
                for idx in range(5):
                    if super_conflict_matrix[C[idx], C[(idx+1)%5]] == 0:
                        is_hole = False
                        break
                if not is_hole: continue
                rhs = (len(C) - 1) / 2.0
                lhs = rhs * vals_in_slot[hub] + np.sum(vals_in_slot[C])
                if lhs > rhs + tolerance:
                    violated_cuts.append((j, hub, C))
        return violated_cuts

class GraphPropagationOracle:
    """
    Oráculo de MENSAJERÍA sobre grafos (familia GNN, SIN entrenamiento).

    Realiza inferencia de campo medio desenrollada sobre el grafo de conflicto: en
    cada iteración cada clase actualiza su distribución sobre timeslots restando la
    masa que sus vecinos en conflicto ponen en cada horario (ANTI-difusión). Es la
    dirección correcta para un grafo de conflicto — una GCN estándar difundiría y
    JUNTARÍA a los vecinos, justo lo contrario de lo deseado.

    Aporta lo que el x* del LP no modela: preferencia de horario (penalización) y
    coordinación con la red de vecinos. La versión ENTRENABLE (Θ aprendidos por
    imitación del corpus de soluciones) es la evolución natural; ésta no requiere
    datos etiquetados ni torch. Devuelve P de forma (num_clases, num_slots).
    """
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.logger = logging.getLogger("GraphOracle")
        gp: Dict[str, Any] = getattr(cfg, "GRAPH_ORACLE", {}) if cfg is not None else {}
        self.iters: int = int(gp.get("iterations", 8))
        self.beta: float = float(gp.get("repulsion", 1.5))
        self.max_cells: int = int(gp.get("max_cells", 4_000_000))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)

    def predict(self, inst: Instance, class_ids: List[str], W: np.ndarray,
                ts_list: List[str], twin_groups: Optional[List[List[str]]] = None) -> Optional[np.ndarray]:
        n, T = len(class_ids), len(ts_list)
        if n == 0 or T == 0 or n * T > self.max_cells:
            if n * T > self.max_cells:
                self.logger.info(f"  [GraphOracle] omitido: {n}x{T} celdas > max_cells={self.max_cells}.")
            return None
        ts_index = {t: i for i, t in enumerate(ts_list)}
        NEG = -1.0e9
        base = np.full((n, T), NEG, dtype=np.float64)
        for c, cid in enumerate(class_ids):
            at = inst.classes[cid].allowed_times
            if not at:
                base[c, :] = 0.0
            else:
                for t, pen in at.items():
                    j = ts_index.get(t)
                    if j is not None:
                        base[c, j] = -float(pen)   # menor penalización => logit mayor
        Wc = (W > 0).astype(np.float64)
        P = self._softmax(base)
        for _ in range(self.iters):
            neighbor_mass = Wc @ P                 # agregación de mensajes de vecinos
            P = self._softmax(base - self.beta * neighbor_mass)

        # ── CANONICALIZACIÓN (equivariancia real): las clases GEMELAS reciben
        # exactamente el mismo prior. Sus filas convergen casi idénticas por
        # construcción; el promedio del grupo garantiza igualdad EXACTA (elimina
        # ruido de flotantes) y respeta la simetría real de la instancia.
        if twin_groups:
            cindex = {cid: i for i, cid in enumerate(class_ids)}
            for grp in twin_groups:
                rows = [cindex[c] for c in grp if c in cindex]
                if len(rows) > 1:
                    P[rows, :] = P[rows, :].mean(axis=0, keepdims=True)
        return P

    def build_canonical_features(self, inst: Instance, class_ids: List[str],
                                 twin_groups: Optional[List[List[str]]] = None) -> np.ndarray:
        """
        Punto de ingesta para la futura GNN ENTRENADA. Produce una matriz de
        features (num_clases × F) con dos garantías de equivariancia REAL:
          1. Codificación CÍCLICA del tiempo: día de la semana y hora del día se
             representan con sin/cos (traslación toroidal de la semana), sin
             imponer simetrías poliédricas D_n inexistentes en el horario.
          2. Clases GEMELAS -> filas EXACTAMENTE idénticas (se calcula la fila
             del representante canónico y se copia al resto del grupo).
        Features por clase: [n_times, pen_time_min, pen_time_media,
        sin/cos(día medio), sin/cos(hora media), n_rooms, pen_room_min,
        pen_room_media, room_required, length, limit].
        """
        spd = max(1, inst.slots_per_day)
        n = len(class_ids)
        F = 13
        X = np.zeros((n, F), dtype=np.float64)

        def _row(cid: str) -> np.ndarray:
            cls = inst.classes[cid]
            at = cls.allowed_times
            v = np.zeros(F, dtype=np.float64)
            if at:
                pens = np.array([float(p) for p in at.values()])
                sd = cd = sh = ch = 0.0
                cnt = 0
                for t in at:
                    ts = inst.timeslots.get(t)
                    if not ts:
                        continue
                    m, b = ts.days_mask, 0
                    while m:
                        if m & 1:
                            ang = 2.0 * math.pi * b / 7.0
                            sd += math.sin(ang); cd += math.cos(ang)
                            cnt += 1
                        m >>= 1; b += 1
                    frac = (ts.start_int % spd) / spd
                    sh += math.sin(2.0 * math.pi * frac); ch += math.cos(2.0 * math.pi * frac)
                k = max(1, cnt); kt = max(1, len(at))
                v[0] = len(at); v[1] = pens.min(); v[2] = pens.mean()
                v[3] = sd / k; v[4] = cd / k
                v[5] = sh / kt; v[6] = ch / kt
            ar = cls.allowed_rooms
            if ar:
                rp = np.array([float(p) for p in ar.values()])
                v[7] = len(ar); v[8] = rp.min(); v[9] = rp.mean()
            v[10] = 1.0 if cls.room_required else 0.0
            v[11] = float(cls.length)
            v[12] = float(cls.limit)
            return v

        canon: Dict[str, str] = {}
        for grp in (twin_groups or []):
            for c in grp:
                canon[c] = grp[0]
        cache: Dict[str, np.ndarray] = {}
        for i, cid in enumerate(class_ids):
            rep = canon.get(cid, cid)
            if rep not in cache:
                cache[rep] = _row(rep)
            X[i, :] = cache[rep]
        return X


# =============================================================================
#  FASE 5 — CSP TOPOLÓGICO Y REDONDEO DEPENDIENTE
#  Oráculo estocástico: x* del LP (StochasticPackingRounder) enriquecido por un
#  oráculo de mensajería sobre grafos (GraphPropagationOracle). Sin torch.
# =============================================================================

class CSPSolver:
    def __init__(self, inst: Instance, W: np.ndarray, idx: Dict[str, int], cfg):
        self.inst = inst
        self.W = W
        self.idx = idx
        self.cfg = cfg
        self.logger = logging.getLogger("CSPSolver")
        self.bt_nodes = 0
        self.start_time = 0.0
        self._room_avail_cache: Dict[Tuple[str, str, int], bool] = {}
        self.ts_list = list(inst.timeslots.keys())

        # ── SYMMETRY BREAKING (simetrías REALES de la instancia) ─────────────
        # Clases gemelas: orden lexicográfico encadenado (prev <= var <= next).
        # Aulas gemelas: canonicalización determinista del orden de exploración.
        self._twin_prev: Dict[str, str] = {}
        self._twin_next: Dict[str, str] = {}
        self._room_twin_pos: Dict[str, int] = {}
        self._ts_rank: Dict[str, int] = {t: i for i, t in enumerate(self.ts_list)}
        self._room_rank: Dict[str, int] = {r: i for i, r in enumerate(inst.rooms.keys())}
        self._class_twin_groups: List[List[str]] = []
        if self.cfg.CSP.get("symmetry_breaking", True):
            self._class_twin_groups = detect_class_twins(inst)
            for grp in self._class_twin_groups:
                for a_t, b_t in zip(grp, grp[1:]):
                    self._twin_next[a_t] = b_t
                    self._twin_prev[b_t] = a_t
            room_twin_groups = detect_twin_rooms(inst)
            for grp in room_twin_groups:
                for pos, r in enumerate(grp):
                    self._room_twin_pos[r] = pos
            n_tc = sum(len(g) for g in self._class_twin_groups)
            n_tr = sum(len(g) for g in room_twin_groups)
            self.logger.info(f"  [Simetría] clases gemelas: {len(self._class_twin_groups)} grupos "
                             f"({n_tc} clases) | aulas gemelas: {len(room_twin_groups)} grupos "
                             f"({n_tr} aulas). Orden lexicográfico activo.")

        # -------------------------------------------------------------
        # INYECCIÓN DE LAS FASES ARQUITECTÓNICAS DENTRO DEL CSP
        # -------------------------------------------------------------
        class_ids = list(self.inst.classes.keys())
        num_classes = len(class_ids)
        conflict_matrix = (self.W > 0).astype(int)
        
        allowed_slots = {}
        for i, cid in enumerate(class_ids):
            allowed_slots[i] = set(self.ts_list.index(ts) for ts in self.inst.classes[cid].allowed_times) if self.inst.classes[cid].allowed_times else set(range(len(self.ts_list)))
            
        # Paso 1: Topología
        # ── FIX CRÍTICO ──────────────────────────────────────────────────
        # NO colapsar clases en supernodos. Una arista en el grafo de conflicto
        # (estudiantes compartidos o distribución dura tipo SameAttendees/NotOverlap)
        # significa "deben ir SEPARADAS", no "son intercambiables". Colapsarlas y
        # expandir a un (timeslot, aula) idéntico produce choques de aula y viola
        # las restricciones duras. Además, dos clases distintas NUNCA pueden
        # compartir aula+hora. => Cada clase es su propio nodo/meta-variable.
        supernodes     = {i: [i] for i in range(num_classes)}
        super_W        = conflict_matrix
        super_weights  = {i: 1 for i in range(num_classes)}
        super_allowed  = allowed_slots
        self.meta_groups = {cid: [cid] for cid in class_ids}
        # ─────────────────────────────────────────────────────────────────
        
        # Extracción Estricta de Cliques (Para Inyección directa al LP)
        class_id_to_sid = {}
        for s_id, c_indices in supernodes.items():
            for c_idx in c_indices:
                class_id_to_sid[class_ids[c_idx]] = s_id

        strict_cliques = set()
        
        # Cliques por Estudiantes Compartidos
        student_to_classes = defaultdict(list)
        for cid, cls in self.inst.classes.items():
            for st in cls.students: student_to_classes[st].append(cid)
        for st, cids in student_to_classes.items():
            if len(cids) > 1:
                s_clique = frozenset({class_id_to_sid[c] for c in cids if c in class_id_to_sid})
                if len(s_clique) > 1: strict_cliques.add(s_clique)

        # Cliques por Distribuciones (SameAttendees / NotOverlap)
        # Reemplaza el bloque de Distribuciones por este:
        for dist in self.inst.distributions:
            if dist.required and dist.dtype in ("SameAttendees", "NotOverlap", "DifferentTime"):
                if len(dist.classes) > 1:
                    s_clique = frozenset({class_id_to_sid[c] for c in dist.classes if c in class_id_to_sid})
                    if len(s_clique) > 1: strict_cliques.add(s_clique)

        # Paso 3 & 4: Relajación LP (guía suave de ordenamiento; opcional)
        # El LP solo sesga el orden del CSP y NO modela aulas. En instancias
        # masivas (muni ~925k vars) construirlo es caro y aporta poco, así que
        # se omite por tamaño o por bandera de config, cayendo a orden por
        # costo/escasez (fallback ya soportado en _order_values).
        self.fractional_guide = {}
        num_vars_lp = len(supernodes) * len(self.ts_list)
        use_lp = self.cfg.CSP.get("use_lp_guide", True)
        lp_cap = self.cfg.CSP.get("lp_max_vars", 120000)
        if not use_lp:
            self.logger.info("  Guía LP desactivada por configuración (use_lp_guide=False).")
        elif num_vars_lp > lp_cap:
            self.logger.info(f"  Guía LP omitida: {num_vars_lp} vars > lp_max_vars={lp_cap}. Orden por costo/escasez.")
        elif SCIPY_AVAILABLE and linprog is not None:
            rounder = StochasticPackingRounder(len(self.ts_list), self.logger)
            x_star = rounder.solve_fractional_landscape(len(supernodes), super_W, super_allowed, strict_cliques, super_weights)
            for s_id, classes in supernodes.items():
                root_cid = class_ids[classes[0]]
                self.fractional_guide[root_cid] = x_star[s_id]

        # ── ORÁCULO DE MENSAJERÍA SOBRE GRAFOS (familia GNN, sin entrenamiento) ──
        # Enriquece/define la guía con preferencia de horario + coordinación de
        # vecinos (lo que el LP no modela). Se mezcla con x* (peso graph_oracle_weight).
        if self.cfg.CSP.get("use_graph_oracle", True):
            gw = float(self.cfg.CSP.get("graph_oracle_weight", 0.5))
            P = GraphPropagationOracle(self.cfg).predict(self.inst, class_ids, self.W, self.ts_list,
                                                         twin_groups=self._class_twin_groups)
            if P is not None:
                blended = 0
                for c, cid in enumerate(class_ids):
                    p_graph = P[c]
                    if cid in self.fractional_guide:
                        x = np.asarray(self.fractional_guide[cid], dtype=np.float64)
                        s = x.sum()
                        x = (x / s) if s > 0 else p_graph
                        self.fractional_guide[cid] = (1.0 - gw) * x + gw * p_graph
                    else:
                        self.fractional_guide[cid] = p_graph
                    blended += 1
                self.logger.info(f"  [GraphOracle] prior de mensajería aplicado a {blended} clases "
                                 f"(w={gw}, {'mezclado con x*' if self.fractional_guide else 'solo'}).")

        # -------------------------------------------------------------
        self.meta_domains = self._build_meta_domains()

        self._required_dists_by_class: Dict[str, List[Distribution]] = {}
        for dist in self.inst.distributions:
            if not dist.required: continue
            for cid in dist.classes:
                self._required_dists_by_class.setdefault(cid, []).append(dist)

        self._scarce_rooms: Set[str] = set()
        for cid, dom in self.meta_domains.items():
            cls = self.inst.classes[cid]
            if not cls.room_required: continue
            rooms_in_dom = set(r for _, r in dom if r not in ("NO_ROOM", ""))
            if len(rooms_in_dom) == 1:
                self._scarce_rooms |= rooms_in_dom

    def _val_rank(self, ts_id: Optional[str], room: Optional[str]) -> int:
        """Rango total y estable de un valor (timeslot, aula) para el orden
        lexicográfico entre clases gemelas. NO_ROOM/'' ranquean como 0."""
        r = self._room_rank.get(room or "", -1) + 1
        return self._ts_rank.get(ts_id or "", 0) * (len(self._room_rank) + 2) + r

    def _soft_cost(self, meta_assignment: Dict[str, Assignment]) -> float:
        # Costo blando aproximado (tiempo+aula) con pesos ITC, para comparar restarts.
        # Big-M: una clase que REQUIERE aula pero quedó en NO_ROOM no es válida para
        # el validador (cuenta como "sin asignar"); se penaliza fuerte para que
        # keep-cheapest / run_parallel jamás prefieran una solución con NO_ROOM.
        wt = self.inst.optimization.get("time", 1)
        wr = self.inst.optimization.get("room", 1)
        BIG_M = 1_000_000.0
        tot = 0.0
        for root, a in meta_assignment.items():
            for cid in self.meta_groups[root]:
                cls = self.inst.classes[cid]
                tot += cls.allowed_times.get(a.timeslot, 0) * wt
                if a.room not in ("NO_ROOM", "", None):
                    tot += cls.allowed_rooms.get(a.room, 0) * wr
                elif cls.room_required:
                    tot += BIG_M
        return tot

    def _timeout(self) -> bool:
        limit = self.cfg.CSP.get("timeout_seconds", 300)
        return False if limit <= 0 else (time.time() - self.start_time) > limit

    def _room_available_for_ts(self, room_id: str, ts_id: str, cls_length: int) -> bool:
        cache_key = (room_id, ts_id, cls_length)
        cached = self._room_avail_cache.get(cache_key)
        if cached is not None: return cached
        result = self._room_available_for_ts_compute(room_id, ts_id, cls_length)
        self._room_avail_cache[cache_key] = result
        return result

    def _room_available_for_ts_compute(self, room_id: str, ts_id: str, cls_length: int) -> bool:
        if room_id in ("NO_ROOM", "") or room_id not in self.inst.rooms: return True
        room = self.inst.rooms[room_id]
        if not room.unavailable: return True
        ts = self.inst.timeslots.get(ts_id)
        if not ts: return True
        ts_start = ts.start_int
        ts_end = ts_start + cls_length
        for blk in room.unavailable:
            if (ts.days_mask & blk["days_mask"]) and (ts.weeks_mask & blk["weeks_mask"]):
                b_start, b_end = blk.get("start", 0), blk.get("start", 0) + blk.get("length", 0)
                if not (ts_end <= b_start or b_end <= ts_start): return False
        return True

    def _conflicts(self, var: str, ts_id: str, room: str, assignment: Dict[str, Assignment]) -> bool:
        cls = self.inst.classes[var]
        ts_a = self.inst.timeslots.get(ts_id)
        if not ts_a: return True
        len_a = slot_len(self.inst, var, ts_id)
        sa, ea = ts_a.start_int, ts_a.start_int + len_a

        if not self._room_available_for_ts(room, ts_id, len_a): return True
        if cls.allowed_times and ts_id not in cls.allowed_times: return True
        if cls.room_required and cls.allowed_rooms and room not in cls.allowed_rooms: return True

        for other_id, asgn in assignment.items():
            if other_id == var: continue
            other_cls = self.inst.classes[other_id]
            ts_b = self.inst.timeslots.get(asgn.timeslot)
            if not ts_b: continue
            room_b = asgn.room
            sb, eb = ts_b.start_int, ts_b.start_int + slot_len(self.inst, other_id, asgn.timeslot)
            
            share_d = (ts_a.days_mask & ts_b.days_mask) != 0
            share_w = (ts_a.weeks_mask & ts_b.weeks_mask) != 0
            overlap_t = not (ea <= sb or eb <= sa)
            overlap_full = overlap_t and share_d and share_w

            if overlap_full and room_b == room and room not in ("NO_ROOM", ""): return True
            if cls.students & other_cls.students:
                if overlap_full: return True 
                if share_d and share_w and room not in ("NO_ROOM", "") and room_b not in ("NO_ROOM", ""):
                    trv = self.inst.rooms.get(room, Room(id=room)).travel.get(room_b, 0)
                    if trv > 0 and (sa < sb and ea + trv > sb or sb < sa and eb + trv > sa): return True

        for dist in self._required_dists_by_class.get(var, ()):
            dtype = dist.dtype
            if not dtype: continue

            for other_var in dist.classes:
                if other_var == var or other_var not in assignment: continue
                asgn_b = assignment[other_var]
                ts_b = self.inst.timeslots.get(asgn_b.timeslot)
                if not ts_b: continue
                other_cls = self.inst.classes[other_var]
                room_b, sb, eb = asgn_b.room, ts_b.start_int, ts_b.start_int + slot_len(self.inst, other_var, asgn_b.timeslot)
                share_d = (ts_a.days_mask & ts_b.days_mask) != 0
                share_w = (ts_a.weeks_mask & ts_b.weeks_mask) != 0
                overlap_t = not (ea <= sb or eb <= sa)
                overlap_full = overlap_t and share_d and share_w
                
                if dtype == "SameRoom" and room != room_b: return True
                elif dtype == "DifferentRoom" and room == room_b and room not in ("NO_ROOM", ""): return True
                elif dtype == "SameTime":
                    contained_ab = (sa <= sb and eb <= ea)
                    contained_ba = (sb <= sa and ea <= eb)
                    if not (contained_ab or contained_ba): return True
                elif dtype == "SameStart" and sa != sb: return True
                elif dtype == "DifferentTime" and overlap_t: return True
                elif dtype == "SameDays":
                    if not ((ts_a.days_mask & ts_b.days_mask) == ts_a.days_mask or
                            (ts_a.days_mask & ts_b.days_mask) == ts_b.days_mask): return True
                elif dtype == "DifferentDays" and share_d: return True
                elif dtype == "SameWeeks":
                    if not ((ts_a.weeks_mask & ts_b.weeks_mask) == ts_a.weeks_mask or
                            (ts_a.weeks_mask & ts_b.weeks_mask) == ts_b.weeks_mask): return True
                elif dtype == "DifferentWeeks" and share_w: return True
                elif dtype == "Overlap" and not overlap_full: return True
                elif dtype == "NotOverlap" and overlap_full: return True
                elif dtype == "SameAttendees":
                    if overlap_full: return True
                    if share_d and share_w and room not in ("NO_ROOM","") and room_b not in ("NO_ROOM",""):
                        trv = self.inst.rooms.get(room, Room(id=room)).travel.get(room_b, 0)
                        if trv == 0 and room_b in self.inst.rooms:
                            trv = self.inst.rooms[room_b].travel.get(room, 0)
                        if trv > 0 and (sa < sb and ea + trv > sb or sb < sa and eb + trv > sa): return True
                elif dtype == "Precedence":
                    idx_var   = dist.classes.index(var)
                    idx_other = dist.classes.index(other_var)
                    if idx_var < idx_other:
                        c1_w, c2_w = ts_a.weeks_mask, ts_b.weeks_mask
                        c1_d, c2_d = ts_a.days_mask,  ts_b.days_mask
                        c1_end, c2_start = ea, sb
                    else:
                        c1_w, c2_w = ts_b.weeks_mask, ts_a.weeks_mask
                        c1_d, c2_d = ts_b.days_mask,  ts_a.days_mask
                        c1_end, c2_start = eb, sa

                    def _first_bit(mask: int) -> int:
                        return (mask & -mask).bit_length() - 1 if mask else 999

                    fw1, fw2 = _first_bit(c1_w), _first_bit(c2_w)
                    fd1, fd2 = _first_bit(c1_d), _first_bit(c2_d)
                    is_before = (fw1 < fw2) or (fw1 == fw2 and (fd1 < fd2 or (fd1 == fd2 and c1_end <= c2_start)))
                    if not is_before: return True
                elif dtype == "WorkDay":
                    param = int(re.search(r'\d+', dist.type).group()) if re.search(r'\d+', dist.type) else 0
                    if share_d and share_w and param > 0:
                        if max(ea, eb) - min(sa, sb) > param: return True
                elif dtype == "MinGap":
                    param = int(re.search(r'\d+', dist.type).group()) if re.search(r'\d+', dist.type) else 0
                    if share_d and share_w and param > 0:
                        if not (ea + param <= sb or eb + param <= sa): return True
        return False

    def _build_meta_domains(self) -> Dict[str, List[Tuple[str, str]]]:
        meta_domains = {}
        for root, members in self.meta_groups.items():
            common_times = set(self.ts_list)
            for cid in members:
                if self.inst.classes[cid].allowed_times: common_times.intersection_update(self.inst.classes[cid].allowed_times.keys())
            
            common_rooms, room_req = None, False
            for cid in members:
                cls = self.inst.classes[cid]
                if cls.room_required:
                    room_req = True
                    if cls.allowed_rooms:
                        if common_rooms is None: common_rooms = set(cls.allowed_rooms.keys())
                        else: common_rooms.intersection_update(cls.allowed_rooms.keys())
            
            if not room_req: common_rooms = {"NO_ROOM"}
            elif common_rooms is None: common_rooms = set(self.inst.rooms.keys())

            valid_pairs = [(t, r) for t in common_times for r in common_rooms
                           if self._room_available_for_ts(r, t, slot_len(self.inst, root, t))]
            meta_domains[root] = valid_pairs
        return meta_domains

    def _expand_assignment(self, meta_assignment: Dict[str, Assignment]) -> Dict[str, Assignment]:
        return {cid: Assignment(asgn.timeslot, asgn.room) for root, asgn in meta_assignment.items() for cid in self.meta_groups[root]}

    def _meta_conflicts(self, meta_var: str, ts_id: str, room: str, current_meta_asgn: Dict[str, Assignment]) -> bool:
        expanded_asgn = self._expand_assignment(current_meta_asgn)
        for cid in self.meta_groups[meta_var]:
            if self._conflicts(cid, ts_id, room, expanded_asgn): return True
        return False

    def _meta_conflicts_with_culprits(self, meta_var: str, ts_id: str, room: str, current_meta_asgn: Dict[str, Assignment]) -> Set[str]:
        cid = meta_var  
        if not self._conflicts(cid, ts_id, room, current_meta_asgn):
            return set()

        if self._conflicts(cid, ts_id, room, {}):
            return set()

        culprits: Set[str] = set()
        for other_meta, other_asgn in current_meta_asgn.items():
            if self._conflicts(cid, ts_id, room, {other_meta: other_asgn}):
                culprits.add(other_meta)
        return culprits

    def _order_values(self, var: str, domain: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Ordenamiento Estocástico Guiado (Redondeo Dependiente):
        Renormaliza las probabilidades del LP sobre el dominio residual, asegurando que 
        si un aula/tiempo fue tomado, su colapso a cero impulse negativamente 
        la elección hacia las opciones sobrevivientes.
        """
        cls = self.inst.classes[var]
        is_flexible = len(domain) > 1

        prob_map = {}
        total_prob = 0.0
        
        # Extraer probabilidades solo de las franjas horarias AÚN DISPONIBLES en el dominio
        if var in self.fractional_guide:
            for ts_id, room_id in domain:
                try:
                    ts_idx = self.ts_list.index(ts_id)
                    prob = self.fractional_guide[var][ts_idx]
                    prob_map[(ts_id, room_id)] = prob
                    total_prob += prob
                except ValueError:
                    prob_map[(ts_id, room_id)] = 0.0

        def score(val):
            ts_id, room_id = val
            t_pen = cls.allowed_times.get(ts_id, 0)
            r_pen = cls.allowed_rooms.get(room_id, 0)
            scarcity = 100.0 if (is_flexible and room_id in self._scarce_rooms) else 0.0
            # Canonicalización de aulas GEMELAS: 0.6*pos domina el tiebreak
            # aleatorio (0..0.5) pero nunca revierte una penalización entera
            # estrictamente mejor => orden determinista solo entre gemelas.
            twin_bias = 0.6 * self._room_twin_pos.get(room_id, 0)

            prob_bonus = 0.0
            if total_prob > 0:
                # Normalización Dependiente: 
                # (Valor de la celda) / (Suma de la probabilidad residual)
                normalized_prob = prob_map[val] / total_prob
                prob_bonus = -1000.0 * normalized_prob 

            return t_pen + r_pen + scarcity + prob_bonus + twin_bias + random.uniform(0, 0.5)
        
        return sorted(domain, key=score)

    def _greedy_with_restarts(self, unassigned_meta: List[str], meta_domains: Dict[str, List]) -> Dict[str, Assignment]:
        best_meta_assignment = {}
        max_restarts = self.cfg.CSP.get("restarts", 25)
        max_backtracks_per_restart = self.cfg.CSP.get("max_backtracks", 250)

        # ── PROPAGACIÓN "NAKED SINGLES" ──────────────────────────────────
        # Las clases con dominio de tamaño 1 no tienen alternativa: se fijan
        # ANTES de cualquier restart y su (tiempo, aula) queda reservado en el
        # contexto de conflictos, de modo que ninguna clase flexible pueda
        # ocupar ese slot. Esto garantiza que las clases rígidas siempre entren.
        self._pinned_vars: Set[str] = set()
        pinned: Dict[str, Assignment] = {}
        for v in unassigned_meta:
            dom = meta_domains.get(v, [])
            if len(dom) == 1:
                ts, room = dom[0]
                pinned[v] = Assignment(timeslot=ts, room=room)
                self._pinned_vars.add(v)
        free_vars = [v for v in unassigned_meta if v not in pinned]
        if pinned:
            self.logger.info(f"  Fijadas {len(pinned)} clases rígidas (|dominio|=1) por propagación.")

        # Parada temprana por estancamiento: una vez COMPLETA la asignación, si
        # tras 'stagnation' restarts consecutivos no baja el costo, cortamos
        # (evita gastar los 100 restarts / timeout sin ganar nada — el "cuelgue").
        stagnation = self.cfg.CSP.get("stagnation_restarts", 8)
        min_restarts = self.cfg.CSP.get("min_restarts", 3)
        n_total = len(unassigned_meta)
        best_cost = float("inf")
        no_improve = 0

        for restart in range(max_restarts):
            if self._timeout(): break

            def _domain_size(v): return len(meta_domains.get(v, []))
            shuffled = list(free_vars)
            random.shuffle(shuffled)  
            ordered_unassigned = sorted(shuffled, key=_domain_size)
            position = {v: i for i, v in enumerate(ordered_unassigned)}

            # El contexto arranca con las clases rígidas ya colocadas.
            current_meta_assignment = {k: Assignment(a.timeslot, a.room) for k, a in pinned.items()}
            nogoods: Dict[str, Set[Tuple[str, str]]] = {v: set() for v in free_vars}
            backtracks_this_restart = 0
            # assignment_order NO incluye las fijas: nunca se deshacen en backjump.
            assignment_order: List[str] = []

            pending = deque(ordered_unassigned)
            stalled: Set[str] = set()  

            while pending:
                if self._timeout() or backtracks_this_restart > max_backtracks_per_restart:
                    break

                var = pending.popleft()
                if var in current_meta_assignment:
                    stalled.discard(var)
                    continue

                domain = [v for v in self._order_values(var, meta_domains.get(var, [])) if v not in nogoods[var]]

                # ── SYMMETRY BREAKING lex: prev_gemela <= var <= next_gemela ──
                # Poda sólida: toda solución tiene una forma canónica que cumple
                # el orden, así que filtrar no pierde soluciones. Si el filtro
                # vacía el dominio, la gemela que impuso la cota es culprit (el
                # backjump debe poder revisitarla).
                lex_culprits: Set[str] = set()
                _pt = self._twin_prev.get(var)
                if _pt and _pt in current_meta_assignment:
                    _lo = self._val_rank(current_meta_assignment[_pt].timeslot, current_meta_assignment[_pt].room)
                    _pre = len(domain)
                    domain = [v for v in domain if self._val_rank(v[0], v[1]) >= _lo]
                    if _pre and not domain: lex_culprits.add(_pt)
                _nt = self._twin_next.get(var)
                if _nt and _nt in current_meta_assignment:
                    _hi = self._val_rank(current_meta_assignment[_nt].timeslot, current_meta_assignment[_nt].room)
                    _pre = len(domain)
                    domain = [v for v in domain if self._val_rank(v[0], v[1]) <= _hi]
                    if _pre and not domain: lex_culprits.add(_nt)

                placed = False
                culprits: Set[str] = set(lex_culprits)

                for ts, room in domain:
                    self.bt_nodes += 1
                    conflicting_with = self._meta_conflicts_with_culprits(var, ts, room, current_meta_assignment)
                    if not conflicting_with:
                        current_meta_assignment[var] = Assignment(timeslot=ts, room=room)
                        assignment_order.append(var)
                        placed = True
                        break
                    else:
                        culprits |= conflicting_with

                if placed:
                    stalled.clear()  
                    continue

                backtracks_this_restart += 1

                if not culprits or not assignment_order:
                    if var in stalled: continue
                    stalled.add(var)
                    pending.append(var)
                    continue

                jump_target = None
                for back_var in reversed(assignment_order):
                    if back_var in culprits:
                        jump_target = back_var
                        break
                if jump_target is None:
                    jump_target = assignment_order[-1]

                undone_vars = []
                while assignment_order and assignment_order[-1] != jump_target:
                    undone_vars.append(assignment_order.pop())
                if assignment_order:
                    undone_vars.append(assignment_order.pop()) 

                if undone_vars:
                    culprit_var = undone_vars[-1]
                    culprit_val = (current_meta_assignment[culprit_var].timeslot, current_meta_assignment[culprit_var].room)
                    nogoods[culprit_var].add(culprit_val)

                for undone in undone_vars:
                    del current_meta_assignment[undone]

                to_requeue = set(undone_vars) | {var}
                stalled -= to_requeue  
                remaining_pending = [v for v in pending if v not in to_requeue]
                requeue_sorted = sorted(to_requeue, key=lambda v: position[v])
                pending = deque(requeue_sorted + remaining_pending)

            # Preferir (a) más clases asignadas y, a igualdad, (b) menor costo blando.
            # Así los restarts dejan de ser solo "encontrar factible" y pasan a
            # muestrear el layout más barato (explota la varianza entre semillas).
            cur_n, best_n = len(current_meta_assignment), len(best_meta_assignment)
            if cur_n > best_n or (cur_n == best_n and cur_n > 0 and
                                  self._soft_cost(current_meta_assignment) < self._soft_cost(best_meta_assignment)):
                best_meta_assignment = deepcopy(current_meta_assignment)
            # Seguimos iterando para abaratar, PERO paramos si el costo se estanca.
            if len(best_meta_assignment) == n_total:
                bc = self._soft_cost(best_meta_assignment)
                if bc < best_cost - 1e-9:
                    best_cost, no_improve = bc, 0
                else:
                    no_improve += 1
                if restart + 1 >= min_restarts and no_improve >= stagnation:
                    self.logger.info(f"  Parada por estancamiento: {no_improve} restarts sin mejora "
                                     f"(costo blando ≈ {best_cost:.0f}, restart {restart+1}/{max_restarts}).")
                    break

        missing = [v for v in unassigned_meta if v not in best_meta_assignment]
        for v in missing:
            if self._try_repair_insert(v, best_meta_assignment, meta_domains): continue  

        missing = [v for v in unassigned_meta if v not in best_meta_assignment]
        for v in missing:
            assigned = False
            for ts, room in meta_domains.get(v, []):
                if not self._meta_conflicts(v, ts, room, best_meta_assignment):
                    best_meta_assignment[v] = Assignment(ts, room)
                    assigned = True
                    break
            if not assigned:
                cls_info = self.inst.classes[v]
                fallback_ts = next(iter(cls_info.allowed_times), self.ts_list[0] if self.ts_list else "")
                if fallback_ts:
                    best_meta_assignment[v] = Assignment(fallback_ts, "NO_ROOM")
                    self.logger.warning(f"  ⚠ Clase {v}: dominio vacío o totalmente en conflicto. Asignada con NO_ROOM.")

        return best_meta_assignment

    def _try_repair_insert(self, v: str, assignment: Dict[str, Assignment], meta_domains: Dict[str, List], max_moves: int = 4) -> bool:
        for ts, room in self._order_values(v, meta_domains.get(v, [])):
            trial = dict(assignment)
            if self._relocate_into(v, ts, room, trial, meta_domains, depth=max_moves, moving=set()):
                for k, a in trial.items(): assignment[k] = a
                return True
        return False

    def _relocate_into(self, v: str, ts: str, room: str, trial: Dict[str, Assignment], meta_domains: Dict[str, List], depth: int, moving: Set[str]) -> bool:
        culprits = self._meta_conflicts_with_culprits(v, ts, room, {k: a for k, a in trial.items() if k != v})
        if not culprits:
            trial[v] = Assignment(ts, room)
            return True
        if depth <= 0 or culprits & moving: return False
        # No se puede reubicar una clase rígida (dominio 1): abortar esta rama.
        if culprits & getattr(self, "_pinned_vars", set()): return False

        snapshot = dict(trial)
        trial[v] = Assignment(ts, room)
        new_moving = moving | {v} | culprits
        for cstar in culprits:
            del trial[cstar]
            placed = False
            for c_ts, c_room in self._order_values(cstar, meta_domains.get(cstar, [])):
                if self._relocate_into(cstar, c_ts, c_room, trial, meta_domains, depth - 1, new_moving):
                    placed = True
                    break
            if not placed:
                trial.clear()
                trial.update(snapshot)
                return False
        return True

    def solve(self) -> Optional[Dict[str, Assignment]]:
        self.start_time = time.time()
        unassigned_meta = list(self.meta_groups.keys())
        meta_assignment = self._greedy_with_restarts(unassigned_meta, self.meta_domains)
        final_assignment = self._expand_assignment(meta_assignment)
        elapsed = time.time() - self.start_time
        if len(final_assignment) == len(self.inst.classes): 
            self.logger.info(f"  ✓ CSP Completo (evaluaciones: {self.bt_nodes}, t: {elapsed:.1f}s)")
        else: 
            self.logger.warning(f"  ⚠ CSP Parcial. Pasando a SA...")
        return final_assignment

# =============================================================================
#  FASE 6 — RECOCIDO SIMULADO (SA)
# =============================================================================

class SimulatedAnnealing:
    def __init__(self, inst: Instance, cfg, csp, polytopes):
        self.inst = inst
        self.cfg = cfg
        self.csp = csp
        self.polytopes = polytopes
        self.logger = logging.getLogger("SA")
        self.ts_list = list(inst.timeslots.keys())
        # ── SIN {0,c}: cohortes = micro-politopos espectrales (bloques fuerte-
        # mente conectados por estudiantes/distribuciones/curso). Solo pueden
        # exceder el umbral c los de tamaño > c, así que precomputamos esos.
        self._sin_c: int = int(self.cfg.SA.get("sin_balance_c", 4))
        self._sin_w: float = float(self.cfg.SURROGATE_WEIGHTS.get("sin_balance", 3.0))
        self._cohorts: List[List[str]] = [
            p.get("class_ids", []) for p in (polytopes or [])
            if isinstance(p, dict) and len(p.get("class_ids", [])) > self._sin_c
        ]

    def _surrogate_penalty(self, assignment: Dict[str, Assignment]) -> float:
        w_time = self.cfg.SURROGATE_WEIGHTS.get("time_preference", 2.0)
        w_room = self.cfg.SURROGATE_WEIGHTS.get("room_preference", 1.0)
        w_dist = self.cfg.SURROGATE_WEIGHTS.get("distribution_soft", 5.0)
        p_time, p_room, p_dist = 0.0, 0.0, 0.0

        for cid, asgn in assignment.items():
            cls = self.inst.classes[cid]
            if asgn.timeslot in cls.allowed_times: p_time += cls.allowed_times[asgn.timeslot] * w_time
            if asgn.room in cls.allowed_rooms: p_room += cls.allowed_rooms[asgn.room] * w_room

        for dist in self.inst.distributions:
            if dist.required: continue
            for i, a in enumerate(dist.classes):
                for b in dist.classes[i+1:]:
                    if a not in assignment or b not in assignment: continue
                    aa, ab = assignment[a], assignment[b]
                    ts_a, ts_b = self.inst.timeslots.get(aa.timeslot), self.inst.timeslots.get(ab.timeslot)
                    if not ts_a or not ts_b: continue
                    cls_a, cls_b = self.inst.classes[a], self.inst.classes[b]
                    sa, ea = ts_a.start_int, ts_a.start_int + slot_len(self.inst, a, aa.timeslot)
                    sb, eb = ts_b.start_int, ts_b.start_int + slot_len(self.inst, b, ab.timeslot)
                    share_d = (ts_a.days_mask & ts_b.days_mask) != 0
                    share_w = (ts_a.weeks_mask & ts_b.weeks_mask) != 0
                    overlap_t = not (ea <= sb or eb <= sa)
                    overlap_full = overlap_t and share_d and share_w
                    dtype = re.match(r"([a-zA-Z]+)", dist.type.strip())
                    dtype = dtype.group(1) if dtype else ""

                    violated = False
                    if dtype == "SameRoom" and aa.room != ab.room: violated = True
                    elif dtype == "DifferentRoom" and aa.room == ab.room: violated = True
                    elif dtype == "NotOverlap" and overlap_full: violated = True
                    elif dtype == "DifferentTime" and overlap_t: violated = True
                    elif dtype == "SameDays":
                        if not ((ts_a.days_mask & ts_b.days_mask) == ts_a.days_mask or (ts_a.days_mask & ts_b.days_mask) == ts_b.days_mask): violated = True
                    elif dtype == "DifferentDays" and share_d: violated = True

                    if violated: p_dist += dist.penalty * w_dist
        # ── PENALIZACIÓN SIN {0,c}: dispersión de carga por cohorte y día ──
        # Objetivo tipo tablero balanceado: si una cohorte (micro-politopo
        # espectral) concentra más de c clases el MISMO día, se penaliza
        # linealmente el exceso. Castiga horarios hiper-concentrados y premia
        # la intersección uniforme a lo largo de la semana.
        p_balance = 0.0
        if self._sin_w > 0.0 and self._cohorts:
            c_max = self._sin_c
            for cohort in self._cohorts:
                day_counts = [0, 0, 0, 0, 0, 0, 0]
                for cid in cohort:
                    a = assignment.get(cid)
                    if not a or not a.timeslot:
                        continue
                    ts = self.inst.timeslots.get(a.timeslot)
                    if not ts:
                        continue
                    m, b = ts.days_mask, 0
                    while m:
                        if m & 1:
                            day_counts[b] += 1
                        m >>= 1
                        b += 1
                for cnt in day_counts:
                    if cnt > c_max:
                        p_balance += float(cnt - c_max)
        return p_time + p_room + p_dist + p_balance * self._sin_w

    def _kempe_chain_guided(self, assignment: Dict[str, Assignment]) -> Dict[str, Assignment]:
        new_asgn = deepcopy(assignment)
        valid_vars = [v for v in new_asgn.keys() if len(self.inst.classes[v].allowed_times) > 1]
        if not valid_vars: return assignment
        var_x = random.choice(valid_vars)
        cls_x = self.inst.classes[var_x]
        t_orig, t_alt = new_asgn[var_x].timeslot, random.choice(list(cls_x.allowed_times.keys()))
        if t_orig == t_alt: return assignment
            
        chain, queue = {}, [(var_x, t_alt, t_orig)]
        while queue:
            curr_var, target_t, source_t = queue.pop(0)
            if curr_var in chain: continue
            chain[curr_var] = target_t
            curr_room, curr_cls = new_asgn[curr_var].room, self.inst.classes[curr_var]
            
            topological_neighbors = []
            if curr_var in self.csp.idx:
                u_idx = self.csp.idx[curr_var]
                for other_var in new_asgn.keys():
                    if other_var != curr_var and other_var in self.csp.idx and self.csp.W[u_idx, self.csp.idx[other_var]] > 0:
                        topological_neighbors.append(other_var)
            
            for other_var, other_opt in new_asgn.items():
                if other_var == curr_var or other_var in chain: continue
                ts_target, ts_other = self.inst.timeslots[target_t], self.inst.timeslots[other_opt.timeslot]
                len_c = slot_len(self.inst, curr_var, target_t)
                len_o = slot_len(self.inst, other_var, other_opt.timeslot)
                overlap_full = not (ts_target.start_int + len_c <= ts_other.start_int or ts_other.start_int + len_o <= ts_target.start_int) and (ts_target.days_mask & ts_other.days_mask) != 0 and (ts_target.weeks_mask & ts_other.weeks_mask) != 0
                if (overlap_full and other_var in topological_neighbors) or (overlap_full and curr_room != "NO_ROOM" and curr_room == other_opt.room): queue.append((other_var, source_t, target_t))
                    
        temp_asgn = {k: Assignment(v.timeslot, v.room) for k, v in new_asgn.items() if k not in chain}
        for c_id, new_t in chain.items():
            c_room = new_asgn[c_id].room
            if new_t not in self.inst.classes[c_id].allowed_times or self.csp._room_available_for_ts(c_room, new_t, slot_len(self.inst, c_id, new_t)) == False or self.csp._conflicts(c_id, new_t, c_room, temp_asgn): return assignment 
            temp_asgn[c_id] = Assignment(timeslot=new_t, room=c_room)
            
        for c_id, new_t in chain.items(): new_asgn[c_id].timeslot = new_t
        return new_asgn

    def _dihedral_jump(self, assignment: Dict[str, Assignment]) -> Dict[str, Assignment]:
        valid_polys = [p['class_ids'] for p in self.polytopes if isinstance(p, dict) and 'class_ids' in p and len(p['class_ids']) > 1]
        if not valid_polys: return assignment
        c_ids = random.choice(valid_polys)
        if any(cid not in assignment for cid in c_ids): return assignment

        new_asgn, temp_asgn = deepcopy(assignment), {k: Assignment(v.timeslot, v.room) for k, v in assignment.items() if k not in c_ids}
        k_shift = random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) 
        
        for cid in c_ids:
            try: curr_idx = self.ts_list.index(new_asgn[cid].timeslot)
            except ValueError: return assignment 
            new_ts, curr_room, cls_info = self.ts_list[(curr_idx + k_shift) % len(self.ts_list)], new_asgn[cid].room, self.inst.classes[cid]
            if new_ts not in cls_info.allowed_times or self.csp._room_available_for_ts(curr_room, new_ts, slot_len(self.inst, cid, new_ts)) == False or self.csp._conflicts(cid, new_ts, curr_room, temp_asgn): return assignment
            temp_asgn[cid], new_asgn[cid].timeslot = Assignment(timeslot=new_ts, room=curr_room), new_ts
        return new_asgn

    def _time_room_move(self, assignment: Dict[str, Assignment]) -> Dict[str, Assignment]:
        """
        Movimiento CONJUNTO tiempo+aula: reubica una clase a (otra hora, aula más
        barata). Escapa del óptimo local donde el aula barata está ocupada a la
        hora actual pero libre en otra. Usa pesos ~ITC (time y room).
        """
        wt = self.inst.optimization.get("time", 1)
        wr = self.inst.optimization.get("room", 1)
        cand = [v for v, a in assignment.items()
                if self.inst.classes[v].room_required
                and len(self.inst.classes[v].allowed_rooms) > 1
                and a.room not in ("NO_ROOM", "", None)]
        if not cand:
            return assignment
        cand.sort(key=lambda v: self.inst.classes[v].allowed_rooms.get(assignment[v].room, 0), reverse=True)
        v = random.choice(cand[:max(2, len(cand) // 3)])
        cls = self.inst.classes[v]; av = assignment[v]
        cur_cost = cls.allowed_times.get(av.timeslot, 0) * wt + cls.allowed_rooms.get(av.room, 0) * wr
        rest = {k: a for k, a in assignment.items() if k != v}
        times = list(cls.allowed_times.keys()); random.shuffle(times)
        rooms = sorted(cls.allowed_rooms.items(), key=lambda kv: kv[1])[:8]
        best, best_cost = None, cur_cost
        for t in times[:8]:
            for room, rpen in rooms:
                c = cls.allowed_times.get(t, 0) * wt + rpen * wr
                if c >= best_cost:
                    continue
                if not self.csp._conflicts(v, t, room, rest):
                    best, best_cost = (t, room), c
        if best:
            new_asgn = deepcopy(assignment)
            new_asgn[v] = Assignment(best[0], best[1])
            return new_asgn
        return assignment

    def _room_swap(self, assignment: Dict[str, Assignment]) -> Dict[str, Assignment]:
        """
        INTERCAMBIO de aulas entre dos clases. Necesario porque el greedy ya
        deja a cada clase en su aula barata LIBRE; las aulas baratas restantes
        están ocupadas. Intercambiar destraba esos casos sin desalojar a nadie.
        """
        with_room = [v for v, a in assignment.items()
                     if self.inst.classes[v].room_required and a.room not in ("NO_ROOM", "", None)]
        if len(with_room) < 2:
            return assignment
        # Sesgo: elegir v entre las clases con aula MÁS CARA (mayor ganancia potencial).
        with_room.sort(key=lambda v: self.inst.classes[v].allowed_rooms.get(assignment[v].room, 0), reverse=True)
        top = with_room[:max(2, len(with_room) // 3)]
        v = random.choice(top)
        av = assignment[v]; cls_v = self.inst.classes[v]
        # candidatos w cuya aula sería más barata para v (y viceversa suele bajar)
        random.shuffle(with_room)
        for w in with_room:
            if w == v: continue
            aw = assignment[w]; cls_w = self.inst.classes[w]
            if aw.room == av.room: continue
            # compatibilidad de aulas
            if aw.room not in cls_v.allowed_rooms or av.room not in cls_w.allowed_rooms: continue
            before = cls_v.allowed_rooms.get(av.room, 0) + cls_w.allowed_rooms.get(aw.room, 0)
            after  = cls_v.allowed_rooms.get(aw.room, 0) + cls_w.allowed_rooms.get(av.room, 0)
            if after >= before: continue
            # factibilidad dura del intercambio (cada quien contra el resto, sin el otro)
            rest = {k: a for k, a in assignment.items() if k not in (v, w)}
            rest_w = dict(rest); rest_w[w] = Assignment(aw.timeslot, av.room)   # w toma aula de v
            if self.csp._conflicts(v, av.timeslot, aw.room, rest_w): continue
            rest_v = dict(rest); rest_v[v] = Assignment(av.timeslot, aw.room)
            if self.csp._conflicts(w, aw.timeslot, av.room, rest_v): continue
            new_asgn = deepcopy(assignment)
            new_asgn[v] = Assignment(av.timeslot, aw.room)
            new_asgn[w] = Assignment(aw.timeslot, av.room)
            return new_asgn
        return assignment

    def _room_reassign(self, assignment: Dict[str, Assignment]) -> Dict[str, Assignment]:
        """
        Movimiento de AULA: reubica una clase a un aula más barata (menor
        penalización) que esté libre en su timeslot actual. Es el movimiento que
        faltaba: sin él el SA nunca puede bajar la penalización de aula, que suele
        ser el término dominante del costo.
        """
        candidates = [v for v, a in assignment.items()
                      if self.inst.classes[v].room_required
                      and len(self.inst.classes[v].allowed_rooms) > 1
                      and a.room not in ("NO_ROOM", "", None)]
        if not candidates:
            return assignment
        v = random.choice(candidates)
        cls = self.inst.classes[v]
        cur = assignment[v]
        cur_pen = cls.allowed_rooms.get(cur.room, 0)
        # Aulas ordenadas por penalización ascendente; solo intentamos mejorar.
        cheaper = sorted(((r, p) for r, p in cls.allowed_rooms.items() if p < cur_pen and r != cur.room),
                         key=lambda kv: kv[1])
        if not cheaper:
            return assignment
        others = {k: a for k, a in assignment.items() if k != v}
        for room, _ in cheaper:
            if not self.csp._conflicts(v, cur.timeslot, room, others):
                new_asgn = deepcopy(assignment)
                new_asgn[v] = Assignment(cur.timeslot, room)
                return new_asgn
        return assignment

    def optimize(self, assignment: Dict[str, Assignment]) -> Dict[str, Assignment]:
        if not self.cfg.SA["enabled"] or len(assignment) < 2: return assignment
        current, best = deepcopy(assignment), deepcopy(assignment)
        T, T_min, alpha = self.cfg.SA["initial_temp"], self.cfg.SA["min_temp"], self.cfg.SA["cooling_rate"]
        cur_pen = best_pen = self._surrogate_penalty(best)   # cachear energía actual

        # SA hiper-agresivo abaratando aulas: por defecto ~75% de los movimientos
        # atacan el room_penalty (reasignar/intercambiar/tiempo+aula). El resto usa
        # Kempe y saltos diédricos sobre los micro-politopos espectrales.
        room_prob = self.cfg.SA.get("room_move_prob", 0.75)
        kempe_prob = self.cfg.SA.get("kempe_prob", 0.9)
        init_pen = cur_pen
        accepted = 0
        improved = 0
        iters = 0
        # ── TELEMETRÍA POR MOVIMIENTO (Req: calibrar el salto toroidal) ──
        # int = intentos, efec = efectivos (produjeron candidato != actual),
        # acep = aceptados por Metrópolis, mej = mejoras estrictas (ΔH<0).
        # 'toroidal' = _dihedral_jump: shift cíclico sobre el toroide discreto
        # (días × slots) aplicado a un micro-politopo espectral completo.
        mstats: Dict[str, Dict[str, int]] = {
            m: {"int": 0, "efec": 0, "acep": 0, "mej": 0}
            for m in ("room_reassign", "room_swap", "time_room", "kempe", "toroidal")
        }

        for _ in range(self.cfg.SA["max_iterations"]):
            if T < T_min: break
            iters += 1
            r = random.random()
            if r < room_prob * 0.35:
                move, candidate = "room_reassign", self._room_reassign(current)
            elif r < room_prob * 0.70:
                move, candidate = "room_swap", self._room_swap(current)
            elif r < room_prob:
                move, candidate = "time_room", self._time_room_move(current)
            elif random.random() < kempe_prob:
                move, candidate = "kempe", self._kempe_chain_guided(current)
            else:
                move, candidate = "toroidal", self._dihedral_jump(current)
            mstats[move]["int"] += 1
            if candidate is current:
                # Movimiento infactible/no-op: no gastar una evaluación de H.
                T *= alpha
                continue
            mstats[move]["efec"] += 1
            cand_pen = self._surrogate_penalty(candidate)
            delta = cand_pen - cur_pen
            if delta < -1e-9:
                improved += 1
                mstats[move]["mej"] += 1
            if delta < 0 or random.random() < math.exp(-delta / T):
                current, cur_pen = candidate, cand_pen
                accepted += 1
                mstats[move]["acep"] += 1
                if cur_pen < best_pen:
                    best, best_pen = deepcopy(current), cur_pen
            T *= alpha
        mejora = (1 - best_pen / init_pen) * 100 if init_pen > 0 else 0.0
        self.logger.info(f"  [SA] H inicial={init_pen:.0f} -> H final={best_pen:.0f} "
                         f"({mejora:+.1f}%) | iters={iters} | mejoras_estrictas={improved} | aceptadas={accepted}")
        def _fmt(m: str) -> str:
            s = mstats[m]
            rech = s["efec"] - s["acep"]
            return f"{m} {s['int']}i/{s['efec']}e/{s['acep']}a-{rech}r/{s['mej']}m"
        self.logger.info("  [SA·mov] " + " | ".join(_fmt(m) for m in
                         ("toroidal", "kempe", "room_reassign", "room_swap", "time_room"))
                         + "  (i=intentos, e=efectivos, a=aceptados, r=rechazados, m=mejoras)")
        return best

# =============================================================================
#  FASE 7 — SECCIONAMIENTO JERÁRQUICO DE ESTUDIANTES
# =============================================================================

class StudentSectioningGaleShapley:
    """
    Seccionamiento jerárquico de estudiantes. Para cada estudiante y cada curso
    en que está inscrito, elige UNA clase por subpart de un config, respetando:
      - relación parent-child (una clase con parent solo se toma si el parent ya
        fue elegido),
      - cupo (limit) de cada clase cuando es posible,
      - minimizando solapes de horario con el resto del horario del estudiante.
    Produce self.student_enrollment: class_id -> [student_ids].
    """
    def __init__(self, inst: Instance, cfg):
        self.inst = inst
        self.cfg = cfg
        self.logger = logging.getLogger("Sectioning")
        self.student_enrollment: Dict[str, List[str]] = defaultdict(list)
        self.class_load: Dict[str, int] = defaultdict(int)
        self._build_course_structure()

    def _build_course_structure(self):
        # course -> config -> subpart -> [class_ids]
        self.course_configs: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for cid, cls in self.inst.classes.items():
            self.course_configs[cls.course_id][cls.config_id][cls.subpart_id].append(cid)
        # orden topológico de subparts por config (padres antes que hijos)
        self.config_subpart_order: Dict[Tuple[str, str], List[str]] = {}
        for course, configs in self.course_configs.items():
            for config_id, subparts in configs.items():
                # subpart de cada clase padre
                dep = {sp: set() for sp in subparts}   # sp -> set(subparts de los que depende)
                for sp, cids in subparts.items():
                    for cid in cids:
                        par = self.inst.classes[cid].parent
                        if par and par in self.inst.classes:
                            dep[sp].add(self.inst.classes[par].subpart_id)
                order, placed = [], set()
                # orden estable tipo Kahn
                while len(placed) < len(subparts):
                    progress = False
                    for sp in subparts:
                        if sp in placed: continue
                        if dep[sp] <= placed:
                            order.append(sp); placed.add(sp); progress = True
                    if not progress:   # ciclo raro: colocar el resto tal cual
                        for sp in subparts:
                            if sp not in placed: order.append(sp); placed.add(sp)
                        break
                self.config_subpart_order[(course, config_id)] = order

    def _overlaps(self, ts_a: str, len_a: int, ts_b: str, len_b: int) -> bool:
        A, B = self.inst.timeslots.get(ts_a), self.inst.timeslots.get(ts_b)
        if not A or not B: return False
        if not ((A.days_mask & B.days_mask) and (A.weeks_mask & B.weeks_mask)): return False
        return not (A.start_int + len_a <= B.start_int or B.start_int + len_b <= A.start_int)

    def _conflicts_with_schedule(self, cid: str, schedule: List[str], assignment: Dict[str, Assignment]) -> int:
        a = assignment.get(cid)
        if not a or not a.timeslot: return 0
        la = slot_len(self.inst, cid, a.timeslot)
        n = 0
        for other in schedule:
            b = assignment.get(other)
            if not b or not b.timeslot: continue
            if self._overlaps(a.timeslot, la, b.timeslot, slot_len(self.inst, other, b.timeslot)): n += 1
        return n

    def _select_for_config(self, course: str, config_id: str, prev_schedule: List[str],
                           assignment: Dict[str, Assignment]) -> Optional[List[str]]:
        subparts = self.course_configs[course][config_id]
        chosen: List[str] = []
        chosen_set: set = set()
        for sp in self.config_subpart_order[(course, config_id)]:
            candidates = subparts[sp]
            # respetar parent: si la clase tiene parent, debe estar ya elegido
            valid = [c for c in candidates
                     if (not self.inst.classes[c].parent) or (self.inst.classes[c].parent in chosen_set)]
            if not valid:
                valid = candidates   # sin opción válida: relajar parent (evita quedar sin inscribir)
            # preferir con cupo disponible
            with_cap = [c for c in valid if self.inst.classes[c].limit <= 0
                        or self.class_load[c] < self.inst.classes[c].limit]
            pool = with_cap or valid
            # elegir la que menos solapa con el horario acumulado
            horizon = prev_schedule + chosen
            best = min(pool, key=lambda c: (self._conflicts_with_schedule(c, horizon, assignment),
                                            self.class_load[c]))
            chosen.append(best); chosen_set.add(best)
        return chosen

    def optimize_sectioning(self, assignment: Dict[str, Assignment]) -> Dict[str, Assignment]:
        if not self.cfg.SECTIONING.get("enabled", False) or not self.inst.students:
            self.logger.info("  Seccionamiento saltado: deshabilitado o instancia sin estudiantes.")
            return assignment
        self.student_enrollment.clear(); self.class_load.clear()
        n_students = len(self.inst.students)
        n_conflict_students = 0
        total_conflicts = 0
        for sid, courses in self.inst.students.items():
            schedule: List[str] = []
            for course in courses:
                configs = self.course_configs.get(course)
                if not configs:
                    continue
                # elegir el config que produzca el horario con menos solapes
                best_sel, best_confl = None, None
                for config_id in configs:
                    sel = self._select_for_config(course, config_id, schedule, assignment)
                    if sel is None: continue
                    confl = sum(self._conflicts_with_schedule(c, schedule + [x for x in sel if x != c], assignment)
                                for c in sel)
                    if best_confl is None or confl < best_confl:
                        best_sel, best_confl = sel, confl
                if best_sel:
                    for cid in best_sel:
                        self.student_enrollment[cid].append(sid)
                        self.class_load[cid] += 1
                        self.inst.classes[cid].students.add(sid)
                        schedule.append(cid)
                    total_conflicts += best_confl or 0
            # detectar si el estudiante quedó con algún solape
            if any(self._conflicts_with_schedule(c, [x for x in schedule if x != c], assignment) for c in schedule):
                n_conflict_students += 1
        enrolled_pairs = sum(len(v) for v in self.student_enrollment.values())
        self.logger.info(f"  Seccionamiento: {n_students} estudiantes inscritos, "
                         f"{enrolled_pairs} pares (estudiante,clase), "
                         f"{n_conflict_students} con solape de horario, "
                         f"~{total_conflicts} solapes internos.")
        # advertir clases sobre cupo
        over = [cid for cid, n in self.class_load.items()
                if self.inst.classes[cid].limit > 0 and n > self.inst.classes[cid].limit]
        if over:
            self.logger.info(f"  ⚠ {len(over)} clases quedaron por encima de su límite de cupo.")
        return assignment

# =============================================================================
#  ESCRITOR Y ORQUESTADOR
# =============================================================================

def write_solution(inst: Instance, assignment: Dict[str, Assignment], out_path: str,
                   elapsed: float = 0.0, student_enrollment: Optional[Dict[str, List[str]]] = None):
    root = ET.Element("solution", name=inst.name, runtime=f"{elapsed:.1f}", cores="1", technique="GNN & Latin Polytopes (API-Carpio)", institution="Instituto Tecnológico de León", country="Mexico")
    enroll = student_enrollment or {}
    for cid in sorted(assignment.keys(), key=lambda x: int(x) if x.isdigit() else x):
        asgn = assignment[cid]
        cls_el = ET.SubElement(root, "class", id=cid)
        if asgn.timeslot and asgn.timeslot in inst.timeslots:
            ts = inst.timeslots[asgn.timeslot]
            if ts.days and ts.start and ts.weeks:
                cls_el.set("days",   ts.days)
                cls_el.set("start",  str(ts.start))
                cls_el.set("length", str(slot_len(inst, cid, asgn.timeslot)))
                cls_el.set("weeks",  ts.weeks)
        if asgn.room and asgn.room not in ("NO_ROOM", ""):
            cls_el.set("room", asgn.room)
        # En ITC 2019 los estudiantes inscritos van como hijos <student> de la clase.
        for sid in enroll.get(cid, ()):
            ET.SubElement(cls_el, "student", id=sid)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE solution PUBLIC "-//ITC 2019//DTD Problem Format/EN" "http://www.itc2019.org/competition-format.dtd">\n')
        ET.indent(ET.ElementTree(root), space="  ")
        f.write(ET.tostring(root, encoding="unicode"))

class SolverPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger("Pipeline")
        self.parser = ITC2019Parser()
        self.extractor = SpectralExtractor(cfg)

    def solve_instance(self, xml_path: str) -> Dict:
        name, t0 = Path(xml_path).stem, time.time()
        self.logger.info(f"\n{'═'*60}\n  Instancia: {name}\n{'═'*60}")
        inst = self.parser.parse(xml_path)
        self.logger.info(f"  Clases: {len(inst.classes)} | Aulas: {len(inst.rooms)} | "
                         f"Timeslots: {len(inst.timeslots)} | Estudiantes: {len(inst.students)} | "
                         f"Distribuciones: {len(inst.distributions)}")
        polytopes, W, idx = self.extractor.decompose(inst)
        # ── MÉTRICAS DE POLITOPOS ESPECTRALES ──
        sizes = sorted((len(p['class_ids']) for p in polytopes), reverse=True)
        movibles = [s for s in sizes if s > 1]
        singletons = len(sizes) - len(movibles)
        edges = int((W > 0).sum() // 2)
        media = float(np.mean(movibles)) if movibles else 0.0
        self.logger.info(f"  [Politopos] {len(polytopes)} clústeres espectrales | aristas duras: {edges} | "
                         f"movibles(>1): {len(movibles)} | aislados: {singletons} | "
                         f"mayor: {sizes[0] if sizes else 0} | medio(movibles): {media:.1f}")

        csp = CSPSolver(inst, W, idx, self.cfg)
        asgn = csp.solve()
        if not asgn: return {"instance": name, "status": "INFEASIBLE", "time": time.time() - t0}
        # métrica de costo blando tras CSP
        self.logger.info(f"  [CSP] costo blando (tiempo+aula) ≈ {csp._soft_cost({k: v for k, v in asgn.items()}):.0f}")

        asgn = SimulatedAnnealing(inst, self.cfg, csp, polytopes).optimize(asgn)

        sectioner = StudentSectioningGaleShapley(inst, self.cfg)
        asgn = sectioner.optimize_sectioning(asgn)
        enrollment = sectioner.student_enrollment

        elapsed, out_path = time.time() - t0, str(Path(self.cfg.PATHS["solutions_dir"]) / f"{name}.xml")
        write_solution(inst, asgn, out_path, elapsed=elapsed, student_enrollment=enrollment)
        self.logger.info(f"  Solución escrita: {out_path}")
        return {"instance": name, "status": "OK", "classes": len(asgn), "time": elapsed}

    def run(self, single_instance: Optional[str] = None):
        log_path = self.cfg.PATHS.get("log_file", "solver.log")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
        self.logger = logging.getLogger("Pipeline")
        inst_dir = Path(self.cfg.PATHS["instances_dir"])
        xml_paths = [str(inst_dir / f"{single_instance}.xml")] if single_instance else [str(p) for p in sorted(inst_dir.glob("*.xml")) if p.exists()]
        if not xml_paths: return self.logger.error("Sin instancias para procesar.")
        
        results = [self.solve_instance(p) for p in xml_paths]
        self.logger.info(f"\n{'═'*60}\n  RESUMEN\n{'═'*60}\n  OK={sum(1 for r in results if r['status'] == 'OK')}  INFEASIBLE={sum(1 for r in results if r['status'] == 'INFEASIBLE')}  ERROR={sum(1 for r in results if r['status'] == 'ERROR')}  TOTAL={len(results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--instance", default=None)
    args = parser.parse_args()
    pipeline = SolverPipeline(load_config(args.config))
    pipeline.run(single_instance=args.instance)