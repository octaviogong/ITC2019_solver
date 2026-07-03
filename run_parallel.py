# -*- coding: utf-8 -*-
"""
Runner multiproceso: reparte los restarts del CSP entre varios procesos
(1 por núcleo físico) y se queda con la mejor solución. Windows-safe (spawn):
- La función worker es de nivel de módulo.
- Usa 'import solver_fixed as S' (no importlib con rutas).
- Todo bajo if __name__ == '__main__'.
Uso:  python run_parallel.py --instance bet-sum18 --config config.py --jobs 4
"""
import argparse, os, random, time, multiprocessing as mp
import solver_fixed as S

def _worker(args):
    xml_path, config_path, seed, n_restarts, do_sa = args
    random.seed(seed)
    import numpy as _np; _np.random.seed(seed % (2**31 - 1))
    cfg = S.load_config(config_path)
    cfg.CSP["restarts"] = n_restarts
    inst = S.ITC2019Parser().parse(xml_path)
    pol, W, idx = S.SpectralExtractor(cfg).decompose(inst)
    csp = S.CSPSolver(inst, S.GNNOracle(cfg), W, idx, cfg)
    asgn = csp.solve()
    if do_sa:
        asgn = S.SimulatedAnnealing(inst, cfg, csp, pol).optimize(asgn)
    # serializar (spawn: solo tipos simples)
    ser = {cid: (a.timeslot, a.room) for cid, a in asgn.items()}
    n_asg = sum(1 for a in asgn.values() if a.timeslot)
    # costo blando aprox (tiempo+aula) con pesos ITC
    wt = inst.optimization.get("time", 1); wr = inst.optimization.get("room", 1)
    cost = 0
    for cid, a in asgn.items():
        cl = inst.classes[cid]
        cost += cl.allowed_times.get(a.timeslot, 0) * wt
        if a.room not in ("NO_ROOM", "", None):
            cost += cl.allowed_rooms.get(a.room, 0) * wr
    return (n_asg, cost, seed, ser)

def solve_parallel(instance, config_path, jobs, total_restarts, do_sa=False):
    cfg = S.load_config(config_path)
    inst_dir = cfg.PATHS["instances_dir"]
    xml_path = os.path.join(inst_dir, f"{instance}.xml")
    if not os.path.exists(xml_path):        # permitir ruta directa
        xml_path = instance if os.path.exists(instance) else xml_path
    name = os.path.splitext(os.path.basename(xml_path))[0]
    per = max(1, total_restarts // jobs)
    tasks = [(xml_path, config_path, 1000 + k, per, do_sa) for k in range(jobs)]
    t0 = time.time()
    with mp.Pool(processes=jobs) as pool:
        results = pool.map(_worker, tasks)
    best = max(results, key=lambda r: (r[0], -r[1]))   # más asignadas, menor costo
    print(f"[parallel] jobs={jobs} restarts/job={per} tiempo={time.time()-t0:.1f}s")
    for n, c, s, _ in sorted(results, key=lambda r: (-r[0], r[1])):
        print(f"   worker seed={s}: asignadas={n} costo≈{c}")
    print(f"[parallel] MEJOR: asignadas={best[0]} costo≈{best[1]} (seed={best[2]})")

    # ── Reconstruir la mejor asignación, seccionar estudiantes y ESCRIBIR ──
    best_ser = best[3]
    inst = S.ITC2019Parser().parse(xml_path)
    assignment = {cid: S.Assignment(ts, room) for cid, (ts, room) in best_ser.items()}

    enrollment = {}
    if cfg.SECTIONING.get("enabled", False) and inst.students:
        sec = S.StudentSectioningGaleShapley(inst, cfg)
        sec.optimize_sectioning(assignment)
        enrollment = sec.student_enrollment
        print(f"[parallel] Seccionamiento: {len(inst.students)} estudiantes, "
              f"{sum(len(v) for v in enrollment.values())} inscripciones.")

    out_dir = cfg.PATHS["solutions_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.xml")
    S.write_solution(inst, assignment, out_path, elapsed=time.time() - t0, student_enrollment=enrollment)
    print(f"[parallel] Solución escrita en: {out_path}")
    return best

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--config", default="config.py")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)//2))
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--sa", action="store_true")
    a = ap.parse_args()
    solve_parallel(a.instance, a.config, a.jobs, a.restarts, a.sa)