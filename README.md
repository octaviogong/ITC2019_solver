# 🎓 API-Carpio: Hybrid OR-CP & GNN Solver for University Timetabling (ITC 2019)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-GNN-EE4C2C)
![SciPy](https://img.shields.io/badge/SciPy-HiGHS-8CAAE6)
![Optimization](https://img.shields.io/badge/Optimization-NP--Hard-success)

Este repositorio contiene un solucionador de estado del arte para el **University Course Timetabling Problem** basado en el formato de la [International Timetabling Competition (ITC) 2019](https://www.itc2019.org/). 

El motor central implementa la arquitectura **API-Carpio**, un pipeline híbrido que fusiona Investigación de Operaciones (OR), Constraint Programming (CP), Teoría Espectral de Grafos y **Graph Neural Networks (GNNs)** para resolver de forma escalable la alta dimensionalidad de los horarios universitarios.

---

## 🧠 Arquitectura del Pipeline (7 Fases)

El solucionador no depende de la fuerza bruta; utiliza una deconstrucción geométrica y topológica del problema a través de las siguientes fases:

1. **Parser XML (ITC 2019):** Ingesta completa de restricciones duras, distribuciones lógicas, dependencias *parent-child* y estudiantes.
2. **Extractor Espectral:** Construcción de un multigrafo ponderado (4 capas) y cálculo de la matriz Laplaciana normalizada. Utiliza `eigsh` y K-Means para agrupar las clases en "Micro-Politopos Topológicos", aislando regiones de alta densidad de conflicto.
3. **Relajación Lineal (LP Set Packing):** Modelado del paisaje fraccional. Inyecta facetas estrictas de *clique* para evitar la redundancia y utiliza matrices dispersas (`csr_matrix`) resueltas con el motor **HiGHS** de SciPy, evitando el colapso de memoria RAM (*MemoryError*).
4. **GNN Oracle (Inferencia en Grafos):** Implementación de una Red Neuronal de Grafos (PyTorch / PyG) que realiza inferencia de propagación de mensajes sobre el grafo de conflictos. Modela preferencias de horario y coordinación de vecindades.
5. **CSP Topológico (Guided Search):** Búsqueda de restricciones con **Redondeo Dependiente**. El ordenamiento de variables no es ciego; está guiado matemáticamente por una mezcla ponderada entre las probabilidades fraccionales del LP ($x^*$) y los *logits* predictivos de la GNN.
6. **Recocido Simulado Espectral (SA):** Optimización post-búsqueda que minimiza penalizaciones blandas (espacio y tiempo). Emplea movimientos agresivos de reasignación de aulas, *Kempe Chains* guiadas y **saltos diédricos** sobre los micro-politopos espectrales.
7. **Seccionamiento de Estudiantes (Gale-Shapley):** Enrolamiento *greedy* basado en la capacidad de las aulas y la prevención heurística de colisiones en los horarios de los estudiantes individuales.

---

## ⚙️ Requisitos y Dependencias

Asegúrate de contar con Python 3.8+ y las siguientes librerías instaladas:

```bash
pip install numpy scipy
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install torch_geometric
```
🚀 Uso y Ejecución
El repositorio ofrece dos vías principales de ejecución dependiendo del poder de cómputo y el rigor requerido.

1. Ejecución Paralela (Recomendada)
Para instancias masivas y de competencia, el script run_parallel.py lanza múltiples instancias del CSP (restarts) en procesos paralelos aislados, aprovechando todos los núcleos físicos de la CPU. Explora distintas semillas estocásticas para evadir mínimos locales.

```bash
python run_parallel.py --instance muni-fsps-spr17c --config config.py --jobs 6
--instance: Nombre de la instancia (sin la extensión .xml).

--jobs: Número de workers paralelos (sugerido: el número de núcleos físicos de tu procesador).
```
2. Ejecución Estándar (Single-Core / Debugging)
Para pruebas rápidas y revisión detallada del log (telemetría, análisis espectral, inyección de cliques).

```bash
python solver_fixed.py --instance bet-sum18 --config config.py
```
🎛 Configuración (config.py)
Todo el comportamiento del solver se controla de manera centralizada en config.py. Las rutas, los hiperparámetros de la GNN y el esquema estocástico se configuran aquí:

PATHS: Directorios para las instancias de entrada, guardado de XMLs de salida, logs y pesos pre-entrenados del modelo GNN (gnn_weights.pt).

SURROGATE_WEIGHTS: Función del Hamiltoniano para ajustar qué penaliza más el Recocido Simulado (Tiempo vs. Aula).

CSP: Activa/Desactiva el uso de matrices dispersas (LP), el peso de la red neuronal (graph_oracle_weight = 0.5) y la cantidad de retrocesos máximos.

SA: Configura el Recocido Simulado (Temperatura inicial, factor de enfriamiento, probabilidad de Kempe y movimientos agresivos de aula).

📊 Salida de Resultados
Al finalizar, el programa reportará en la terminal el costo blando alcanzado, el tiempo total y la energía final del SA. Generará un archivo de salida en formato XML (por defecto en el directorio soluciones/) totalmente listo para ser evaluado en el Validador Oficial de ITC 2019.

Desarrollado en el Instituto Tecnológico de León, México. 🇲🇽
