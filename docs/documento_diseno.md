# Documento de Diseño — PSO con Estrategias de Paralelismo

**Asignatura:** Programación Paralela  
**Alumno:** Ander Elorriaga  
**Versión implementada:** V0 · V1 · V2 · V3  
**Fecha:** Abril 2026

---

## 1. Objetivo

El objetivo de este proyecto es diseñar e implementar una solución completa de
*Particle Swarm Optimization* (PSO) en Python siguiendo principios de ingeniería
del software, y utilizarla como banco de pruebas para comparar distintas
estrategias de evaluación paralela y concurrente. El núcleo del algoritmo es
siempre el mismo; lo único que cambia entre versiones es **cómo se evalúa el
fitness** de las partículas, que es el paso más costoso del algoritmo.

---

## 2. El algoritmo PSO

PSO es un método de optimización estocástica inspirado en el comportamiento de
bandadas de pájaros. Se mantiene un *enjambre* de partículas, cada una con
posición **x** y velocidad **v** en un espacio de búsqueda de dimensión *d*.
En cada iteración se aplica la siguiente actualización:

```
v ← w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)
x ← x + v
```

donde *w* es el peso de inercia, *c1* y *c2* son los coeficientes cognitivo y
social, *r1* y *r2* son vectores de números aleatorios en [0,1], *pbest* es la
mejor posición personal de cada partícula y *gbest* es la mejor posición global
del enjambre. El objetivo es minimizar una función continua sujeta a restricciones
de caja (*box constraints*).

### Criterio de parada

El algoritmo se detiene cuando se alcanza el número máximo de iteraciones
(`max_iter`) o cuando la mejora en el mejor fitness global es inferior a una
tolerancia `tol` durante `stagnation_iter` iteraciones consecutivas.

### Gestión de límites

He implementado dos estrategias, elegibles por configuración:

- **Clamp** (por defecto): se recortan las posiciones al dominio [lb, ub] y se
  niega y reduce a la mitad la velocidad al impactar con el borde. Es simple,
  siempre garantiza que las partículas estén dentro del dominio y no provoca
  oscilaciones.
- **Reflect**: se refleja la posición sobre el borde violado y se niega la
  velocidad. Conserva mejor la energía cinética de las partículas, pero puede
  generar oscilaciones en impactos múltiples (mitigado con un clamp de seguridad
  tras varias reflexiones).

He elegido *clamp* como estrategia por defecto porque su comportamiento es
predecible y no introduce inestabilidades numéricas.

### Topologías

- **Global (gBest)**: todos los partículas son atraídas por el mejor global.
  Converge rápido, pero es más susceptible a caer en mínimos locales.
- **Ring (lBest)**: cada partícula sólo conoce a sus *k* vecinos más cercanos
  en un anillo. Converge más lento pero explora mejor el espacio de búsqueda.

---

## 3. Estructura del proyecto

```
p1/
├── pso/                   Paquete principal
│   ├── core/              Motor PSO y estructuras de datos
│   │   ├── pso.py         Clase PSO: punto de entrada único del algoritmo
│   │   ├── swarm.py       Dataclasses: SwarmState, IterationRecord, PSOResult
│   │   ├── bounds.py      Estrategias de límites: ClampStrategy, ReflectStrategy
│   │   └── topology.py    Topologías: GlobalTopology, RingTopology
│   ├── objectives/        Funciones benchmark
│   │   └── benchmarks.py  Sphere, Rosenbrock, Rastrigin, Ackley + registro
│   ├── parallel/          Estrategias de evaluación (V0–V3)
│   │   ├── base.py        Interfaz abstracta FitnessEvaluator
│   │   ├── v0_sequential.py  Bucle Python simple
│   │   ├── v1_threading.py   ThreadPoolExecutor
│   │   ├── v2_multiprocessing.py  ProcessPoolExecutor + batching
│   │   └── v3_asyncio.py    asyncio.gather con latencia simulada
│   ├── experiments/       Orquestación de experimentos
│   │   ├── runner.py      RunConfig + run_experiment()
│   │   └── grid_search.py  Búsqueda de hiperparámetros en rejilla
│   ├── io/
│   │   └── persistence.py  Guardado/carga (YAML, JSON, CSV, NPZ)
│   └── viz/
│       └── visualizer.py   Animaciones 2D/3D, curvas de convergencia, gráficas
├── scripts/
│   ├── run_pso.py          Ejecución de un experimento
│   ├── run_benchmarks.py   Suite completa de benchmarks
│   ├── run_grid_search.py  Grid search de hiperparámetros
│   └── make_viz.py         Generación de visualizaciones
├── tests/
│   ├── test_pso.py         45 tests unitarios del motor PSO
│   └── test_benchmarks.py  Tests de las funciones benchmark
├── config/
│   └── default.yaml        Configuración por defecto
├── results/                Resultados guardados en disco
└── pyproject.toml
```

### Diagrama de dependencias entre módulos

```
scripts  ──►  experiments  ──►  core
                   │             │
                   ├──►  parallel ◄─── core
                   └──►  objectives
                   
io  ────────────────────────────►  core (sólo dataclasses)
viz ────────────────────────────►  io + objectives
```

Los módulos `io` y `viz` son hojas del grafo de dependencias: no dependen de
`parallel` ni de `experiments`, lo que permite usarlos de forma independiente
para análisis post-ejecución.

---

## 4. Componentes principales

### 4.1 `pso/core/pso.py` — Motor PSO

La clase `PSO` es el único punto de entrada al algoritmo. Acepta cualquier
implementación de `FitnessEvaluator`, `BoundsStrategy` y `Topology`, lo que
desacopla completamente el algoritmo de la estrategia de evaluación. El bucle
principal es siempre secuencial (las operaciones de actualización de velocidad y
posición son baratas con NumPy); sólo el paso de evaluación del fitness se delega
al evaluador.

El método `run()` devuelve un objeto `PSOResult` con la mejor posición, su fitness,
el historial de convergencia y métricas de tiempo desglosadas (tiempo de evaluación,
tiempo de actualización, tiempo total).

### 4.2 `pso/objectives/benchmarks.py` — Funciones benchmark

He implementado cuatro funciones estándar de la literatura de optimización:

| Función | Expresión | Mínimo global |
|---------|-----------|---------------|
| Sphere | Σ xᵢ² | 0 en x = (0,…,0) |
| Rosenbrock | Σ [100(xᵢ₊₁−xᵢ²)² + (1−xᵢ)²] | 0 en x = (1,…,1) |
| Rastrigin | 10d + Σ [xᵢ² − 10·cos(2πxᵢ)] | 0 en x = (0,…,0) |
| Ackley | −20e^(−0.2√(Σxᵢ²/d)) − e^(Σcos(2πxᵢ)/d) + 20 + e | 0 en x = (0,…,0) |

Todas las funciones están definidas a **nivel de módulo** para que sean
serializables mediante `pickle`, requisito imprescindible para el uso con
`multiprocessing` en Windows (que utiliza el método de inicio *spawn*).

### 4.3 `pso/parallel/` — Estrategias de evaluación

La interfaz `FitnessEvaluator` define un único método:

```python
def evaluate(self, positions: np.ndarray) -> np.ndarray:
    # positions: (n_partículas, d)  →  fitness: (n_partículas,)
```

Las cuatro implementaciones son intercambiables desde el punto de vista del
motor PSO:

**V0 — Secuencial** (`SequentialEvaluator`): evalúa cada partícula en un bucle
Python. Es el *baseline* para calcular el speedup del resto de versiones.

**V1 — Hilos** (`ThreadingEvaluator`): usa `ThreadPoolExecutor` para evaluar
partículas concurrentemente. El GIL de CPython impide el paralelismo real en
código Python puro, pero se libera durante operaciones NumPy/SciPy, por lo que V1
puede mejorar tiempos cuando la función objetivo llama a extensiones C o implica
I/O. Para funciones triviales como Sphere, el overhead de crear hilos puede hacer
que V1 sea más lento que V0.

**V2 — Procesos** (`MultiprocessingEvaluator`): usa `ProcessPoolExecutor`, que
crea procesos independientes y elude completamente el GIL. Cada partícula (o
*batch* de partículas) se serializa con `pickle`, se envía a un worker, se evalúa
y se devuelve el resultado. He implementado un parámetro `batch_size` para agrupar
varias partículas por tarea y reducir el número de llamadas IPC: con funciones
baratas, el overhead de serialización domina y conviene agrupar; con funciones
costosas, se puede reducir el batch para mejorar el balance de carga.

Limitación: la función objetivo debe ser serializable por `pickle` (funciones a
nivel de módulo sí lo son; lambdas y funciones locales, no). En Windows, los
scripts de entrada deben incluir la guarda `if __name__ == '__main__':` para
evitar la creación recursiva de procesos.

**V3 — Asyncio** (`AsyncioEvaluator`): usa `asyncio.gather` para evaluar todas
las partículas de forma concurrente en un único hilo. Asyncio es concurrencia
cooperativa, **no paralelismo**: no puede acelerar trabajo CPU-bound. Su utilidad
aparece cuando la función objetivo implica espera I/O (llamada a una API REST,
consulta a una base de datos, simulador remoto). He diseñado un escenario de
evaluación asimétrica en el que cada partícula sufre una latencia I/O aleatoria
(`asyncio.sleep`). En ese caso, el tiempo total de evaluación es aproximadamente
max(latencias) en lugar de Σ(latencias), lo que puede suponer un speedup de un
orden de magnitud respecto a V0 con latencias de decenas de milisegundos.

El método `evaluate()` de esta clase llama a `asyncio.run()` internamente, de
modo que la interfaz con el motor PSO sigue siendo síncrona.

### 4.4 `pso/experiments/runner.py` — Runner

La clase `RunConfig` es un dataclass que recoge todos los parámetros de un
experimento (benchmark, dimensión, hiperparámetros PSO, estrategia de evaluación,
semilla). La función `run_experiment(cfg)` construye el evaluador adecuado y
ejecuta el PSO, devolviendo un `PSOResult` enriquecido con la configuración
completa. Esto garantiza que cada resultado guarda toda la información necesaria
para reproducirlo.

### 4.5 `pso/experiments/grid_search.py` — Grid search

`grid_search()` genera el producto cartesiano de un diccionario de parámetros
(por ejemplo `w ∈ {0.4, 0.7, 0.9}`, `c1 ∈ {1.0, 1.5, 2.0}`, `c2 ∈ {1.0, 1.5, 2.0}`)
y lanza un experimento por cada combinación y semilla. Devuelve los resultados
ordenados por fitness medio, lo que permite identificar la mejor configuración
directamente.

### 4.6 `pso/io/persistence.py` — Persistencia

Por cada ejecución se crea un directorio `results/<run_id>/` con:

- `config.yaml`: configuración completa (parámetros, versión de Python, número de
  cores, *hash* del commit git, timestamp). Se elige YAML por ser legible por
  humanos y fácilmente editable.
- `result.json`: mejor posición, mejor fitness, tiempo total, número de
  iteraciones. JSON por ser ligero y universal.
- `history.csv`: métrica por iteración (best fitness, mean fitness, tiempos de
  evaluación y actualización). CSV por compatibilidad con pandas, Excel y R.
- `trajectories.npz` (opcional): posiciones de todas las partículas en cada
  iteración. Formato NPZ (NumPy comprimido) por eficiencia en almacenamiento;
  sólo se activa con `--trajectories` ya que puede ocupar decenas de MB en
  ejecuciones largas.

### 4.7 `pso/viz/visualizer.py` — Visualización

Proporciona cinco tipos de gráficas:
- **Convergencia**: curva de mejor fitness por iteración para una o varias
  ejecuciones (escala logarítmica).
- **Animación 2D**: contorno de la función + posiciones del enjambre + mejor
  global + curva de convergencia, animados frame a frame. Se guarda como GIF
  (Pillow) o MP4 (FFmpeg).
- **Animación 3D**: nube de puntos en 3D para dimensión 3.
- **Speedup**: gráfica de barras con el speedup de cada estrategia respecto a V0.
- **Boxplot**: distribución del fitness final por estrategia a lo largo de
  múltiples semillas.

---

## 5. Logging y observabilidad

He configurado el sistema de logging de Python de forma estándar para librerías
(`NullHandler` en el logger raíz del paquete). Los scripts de entrada configuran
el handler y el nivel deseado. El formato es:

```
2026-04-07 21:34:39.070 [INFO] pso.core — iter=186 gbest=5.14e-11 eval_ms=0.1
```

Cada `IterationRecord` almacena `eval_s` y `update_s` por separado, lo que
permite identificar en qué paso se concentra el tiempo y cuantificar el overhead
de paralelismo.

---

## 6. Reproducibilidad

Toda ejecución acepta un parámetro `seed` que inicializa el generador
`np.random.default_rng(seed)`. La semilla se guarda en `config.yaml` junto al
hash del commit git. Las versiones V1-V3 producen resultados idénticos a V0 para
la misma semilla porque:

- El RNG del PSO (posiciones iniciales, velocidades, r1/r2) es siempre el mismo
  objeto y se llama en el mismo orden.
- Las funciones objetivo son deterministas; sólo varía cuándo/cómo se llaman,
  no qué devuelven.
- En V3, el RNG de latencias es independiente del RNG del PSO.

Los tests verifican explícitamente esta propiedad comparando fitness y posición
entre V0, V1, V2 y V3 para la misma semilla.

---

## 7. Tests

He escrito 45 tests unitarios distribuidos en dos ficheros:

**`tests/test_pso.py`** cubre:
- Reproducibilidad por semilla (V0, V1, V2, V3).
- Que las posiciones siempre permanecen dentro de los límites (clamp y reflect).
- Que el mejor global nunca empeora (monotonicidad).
- Que Sphere d=5 converge a un valor inferior a 1e-4.
- Que la parada temprana por estancamiento se activa.
- Que todos los evaluadores devuelven el mismo array de fitness para las mismas posiciones.

**`tests/test_benchmarks.py`** cubre:
- Que los mínimos conocidos son correctos (Sphere en cero, Rosenbrock en unos, etc.).
- Que las funciones aceptan dimensiones arbitrarias.
- Que no producen NaN para entradas aleatorias.
- El registro `BENCHMARKS` y la generación de instancias reproducibles.

---

## 8. Decisiones, trade-offs y limitaciones

| Decisión | Alternativa descartada | Razonamiento |
|----------|------------------------|--------------|
| `clamp` como estrategia de límites por defecto | `reflect` | Más simple, sin riesgo de oscilaciones, comportamiento predecible |
| Topología global como predeterminada | Ring | Mayor velocidad de convergencia para benchmarks estándar |
| `ProcessPoolExecutor` re-creado en cada llamada | Pool persistente | Simplifica el código; el overhead de creación es medible y educativamente valioso |
| `asyncio.run()` dentro de `evaluate()` | Event loop global externo | Mantiene la interfaz síncrona del motor PSO sin modificarla |
| NPZ para trayectorias | HDF5, Parquet | Sin dependencias adicionales; compresión razonable para arrays densos |
| CSV para historial | Parquet | Legibilidad directa, compatibilidad universal, sin dependencias |

**Limitaciones conocidas:**

- V2 requiere que la función objetivo sea serializable con `pickle`. Lambdas y
  cierres locales no funcionan en Windows.
- V3 no acelera funciones CPU-bound: el beneficio sólo se materializa con
  latencias I/O genuinas o simuladas.
- V2 crea un nuevo pool de procesos en cada iteración del PSO, lo que introduce
  un overhead fijo por iteración. Para benchmarks con funciones muy rápidas (como
  Sphere d=2), este overhead hace que V2 sea entre 10 y 50 veces más lento que V0.
- La animación 3D no representa la superficie de la función objetivo (sería
  demasiado costoso computar un volumen), sólo las posiciones de las partículas.
- Las trayectorias no se guardan por defecto para evitar ficheros de cientos de MB
  en experimentos con muchas iteraciones y partículas.

---

## 9. Cómo reproducir los experimentos

```bash
# Instalar dependencias
pip install -e ".[dev]"

# Una ejecución con V0 (secuencial)
python scripts/run_pso.py --benchmark sphere --dim 10 --strategy v0 --seed 42

# Suite completa (todas las funciones, d=2/10/30, 5 semillas, 4 estrategias)
python scripts/run_benchmarks.py

# Grid search de hiperparámetros
python scripts/run_grid_search.py --benchmark rastrigin --dim 30 --strategy v0

# Animación del enjambre en 2D
python scripts/make_viz.py --mode swarm2d

# Curvas de convergencia comparando resultados guardados
python scripts/make_viz.py --mode convergence --runs-dir results

# Tests
pytest -v
```

Todos los resultados se guardan en `results/` con metadatos completos de
reproducibilidad. Los parámetros por defecto están en `config/default.yaml` y
pueden sobreescribirse vía CLI.
