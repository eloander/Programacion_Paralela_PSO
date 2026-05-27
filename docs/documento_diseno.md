# Documento de Diseño — PSO con Estrategias de Paralelismo
## Caso de uso: Aterrizaje autónomo de cohete

**Asignatura:** Programación Paralela  
**Alumno:** Ander Elorriaga  
**Versión implementada:** V0 · V1 · V2 · V3 · V4 · V5  
**Fecha:** Mayo 2026

---

## 1. Motivación y objetivo

### 1.1 El problema que resolvemos

Un cohete regresa de una misión y necesita aterrizar de pie sobre una plataforma.
Dispone de un motor que puede generar empuje a lo largo de su eje y un sistema de
control de orientación. El desafío es encontrar los **parámetros óptimos del
controlador** que consigan un aterrizaje suave, vertical y en el punto exacto,
sin agotar el combustible ni perder la estabilidad.

Este es exactamente el problema que los ingenieros de SpaceX, Blue Origin y la ESA
resuelven —con modelos mucho más ricos— cuando diseñan los controladores de sus
vehículos reutilizables. En este proyecto lo resolvemos con PSO.

### 1.2 Por qué PSO y por qué paralelismo

El controlador tiene 5 parámetros que interactúan de forma no lineal a través
de un simulador físico. La función que mapea parámetros a calidad de aterrizaje:

- **No tiene forma cerrada**: sólo puede evaluarse ejecutando la simulación completa.
- **No es diferenciable**: los eventos discretos (aterrizaje, activación del freno)
  crean discontinuidades en la función de coste.
- **Es multimodal**: existen muchos controladores que aterrizan, pero sólo algunos
  lo hacen de forma suave y precisa.

Estas características hacen que los métodos basados en gradiente (descenso por
gradiente, BFGS) sean inaplicables. PSO, al no requerir derivadas y explorar el
espacio de búsqueda con múltiples agentes simultáneos, es una elección natural.

Y aquí aparece el paralelismo de forma orgánica: cada partícula del enjambre es
un controlador independiente que debe ser evaluado ejecutando su propia simulación.
Con 30 partículas y 200 iteraciones, eso son **6.000 simulaciones por experimento**.
Distribuirlas en paralelo no es una optimización menor: es la diferencia entre
esperar segundos o minutos.

### 1.3 Objetivo del proyecto

El objetivo es doble:

1. **Implementar y comparar seis estrategias de evaluación paralela** (V0–V5) que
   van desde el bucle secuencial hasta la vectorización NumPy completa, midiendo el
   speedup real en un problema computacionalmente significativo.

2. **Resolver el problema del cohete** de forma rigurosa: simulador físico fiel,
   función de fitness bien diseñada, visualización del comportamiento del controlador
   optimizado y tests que verifican correctitud física y numérica.

---

## 2. Caso de uso: simulador de aterrizaje de cohete

### 2.1 Modelo físico

El cohete opera en un plano 2D. Su estado queda descrito por seis variables:

```
estado = [x, y, vx, vy, theta, omega]
```

| Variable | Unidad | Descripción |
|----------|--------|-------------|
| `x` | m | Posición horizontal (la plataforma está en x = 0) |
| `y` | m | Altitud (el suelo es y = 0) |
| `vx` | m/s | Velocidad horizontal |
| `vy` | m/s | Velocidad vertical (negativa = descendiendo) |
| `theta` | rad | Inclinación respecto a la vertical (0 = perfectamente erguido) |
| `omega` | rad/s | Velocidad angular |

**Condiciones iniciales:**
- Posición: x₀ = 5 m (5 metros a la derecha de la plataforma), y₀ = 50 m
- Velocidad: vx₀ = −1 m/s (deriva lateral), vy₀ = 0 m/s
- Orientación: theta₀ = 0.1 rad (ligera inclinación inicial), omega₀ = 0

**Integración numérica** (Euler semi-implícito, dt = 0.05 s, t_max = 12 s):

```
# Fuerzas en el marco del mundo
Fx = −T · sin(theta)          # componente horizontal del empuje
Fy =  T · cos(theta) − m·g   # componente vertical neta

# Aceleraciones
ax = Fx / m
ay = Fy / m
alpha = tau / I               # torque → aceleración angular

# Actualización de estado (semi-implícito: velocidad primero, posición después)
vx += ax · dt;   vy += ay · dt;   omega += alpha · dt
x  += vx · dt;   y  += vy · dt;   theta += omega · dt
```

Donde `T` es el empuje (limitado a [0, 25 N]), `tau` el torque (limitado a
[−5, 5 N·m]), `m = 1 kg`, `I = 0.05 kg·m²`, `g = 9.81 m/s²`.

**Límites de la simulación:** si el cohete supera 200 m en cualquier dirección,
se considera fuera de área y se penaliza. El combustible inicial es de 100 unidades
y se consume proporcionalmente al empuje aplicado.

### 2.2 El controlador PD

El controlador recibe el estado actual del cohete y decide cuánto empuje aplicar
y qué torque ejercer. Es un **controlador proporcional-derivativo (PD)** de dos
lazos:

**Lazo de empuje** (controla la velocidad vertical):

```
T_nom  = thrust_gain · g · m          # empuje nominal para vuelo estacionario

# Fase de frenado (por debajo de braking_altitude):
T_cmd  = T_nom + vertical_damping · vy
T_cmd  = clip(T_cmd, 0, T_max)
```

Cuando `thrust_gain = 1`, el empuje nominal equilibra exactamente la gravedad:
el cohete queda en vuelo estacionario. Valores menores hacen que descienda
gradualmente. El término `vertical_damping · vy` añade frenado: como `vy < 0`
al descender, un `vertical_damping` negativo genera empuje adicional que frena
la caída.

**Lazo de orientación** (controla la trayectoria horizontal):

```
# Ángulo objetivo: inclinarse hacia la plataforma
target_angle = clip(−arctan(horiz_corr · x), −π/4, π/4)

# Torque PD: corrige el error de ángulo amortiguando omega
tau_cmd = rotation_gain · (target_angle − theta) − angular_damping · omega
tau_cmd = clip(tau_cmd, −tau_max, tau_max)
```

Al inclinar el cohete hacia la plataforma, la componente horizontal del empuje
genera una fuerza lateral que lo desplaza en esa dirección. La ganancia
`horiz_corr` determina cuánto se inclina (y por tanto con qué agresividad se
corrige el error lateral). La ganancia `rotation_gain` determina la rapidez con
que se alcanza el ángulo objetivo.

### 2.3 Espacio de optimización (5 dimensiones)

Cada partícula del enjambre PSO es un vector de 5 parámetros del controlador.
Estos parámetros tienen rangos físicamente significativos:

| # | Parámetro | Rango | Efecto si es muy bajo | Efecto si es muy alto |
|---|-----------|-------|-----------------------|------------------------|
| 0 | `thrust_gain` | [0.5, 3.0] | Cohete cae libre, exceso de velocidad al aterrizar | Cohete sube y nunca aterriza |
| 1 | `vertical_damping` | [−2.0, 2.0] | Sin frenado, impacto duro | Sobrefrena, puede detenerse en el aire y caer |
| 2 | `horiz_correction` | [0.0, 5.0] | No corrige la posición lateral, aterriza lejos del pad | Oscilaciones laterales violentas |
| 3 | `rotation_gain` | [0.0, 10.0] | Respuesta de orientación lenta, inestabilidad | Oscilaciones de alta frecuencia en theta |
| 4 | `braking_altitude` | [0.5, 10.0] | Frena demasiado tarde, velocidad excesiva | Frena demasiado pronto, pierde combustible |

**Ejemplo de controlador inviable:** `thrust_gain = 3.0` genera un empuje nominal
de `3 × 9.81 = 29.4 N`, que supera el máximo (`T_max = 25 N`). El empuje efectivo
(25 N) excede el peso del cohete (9.81 N), por lo que el cohete **acelera hacia
arriba** y nunca aterriza. Fitness = 1000 + altitud_final ≈ 1020 (penalización
de crash).

**Ejemplo de controlador bueno:** `thrust_gain ≈ 0.78`, `vertical_damping ≈ −2.0`,
`horiz_correction ≈ 0.005`, `rotation_gain ≈ 8.1`, `braking_altitude ≈ 9.6`.
Con estos parámetros el PSO (V0, 30 partículas, 200 iter, seed=42) encuentra
fitness ≈ 14.4, que corresponde a un aterrizaje exitoso con ≈1.4 m de error
lateral y baja velocidad de impacto.

### 2.4 Función de fitness

La función mide la calidad del aterrizaje en el **instante exacto de tocar suelo**
(no al final de la simulación):

```
fitness = 10.0 · |x|          (distancia al pad)
        +  5.0 · √(vx²+vy²)  (velocidad de impacto)
        +  3.0 · |theta|      (inclinación al aterrizar)
        +  0.1 · combustible_consumido
        +  2.0 · mean(|omega|) normalizado
```

Los pesos reflejan la importancia relativa de cada criterio: aterrizar en el pad
(×10) es más importante que la estabilidad angular (×2), que a su vez supera el
consumo de combustible (×0.1).

**Penalizaciones adicionales:**
- `+1000 + |y_final|` si el cohete nunca alcanza y = 0 (crash por timeout o por
  salir disparado hacia arriba).
- `+200` si el combustible se agota antes de aterrizar.
- `+500` si el cohete sale de la zona de simulación (|x| > 200 m o y > 200 m).

**Diseño crítico — snapshot al aterrizar:** la versión vectorizada
`rocket_landing_vec` simula *n* partículas con una matriz de estados `(n, 6)`.
Si se usara el estado al final de la simulación, los cohetes ya aterrizados
seguirían acumulando deriva gravitacional (ya sin empuje) durante los pasos
restantes, corrompiendo el fitness. Para evitarlo, el simulador mantiene tres
buffers de instantánea (`final_states`, `final_fuel`, `final_instab`) que
registran el estado exacto en el paso en que cada partícula aterriza:

```python
newly_landed = active & (states[:, 1] <= 0.0)
if np.any(newly_landed):
    final_states[newly_landed] = states[newly_landed]   # snapshot
    final_fuel[newly_landed]   = fuel[newly_landed]
    final_instab[newly_landed] = instability_acc[newly_landed]
active &= ~newly_landed  # desactivar partículas aterrizadas
```

Esto garantiza concordancia exacta entre la versión escalar (que usa `break` al
aterrizar) y la vectorizada (que continúa el bucle para las partículas restantes).

### 2.5 Por qué este problema justifica el paralelismo

La siguiente tabla muestra por qué el cohete es un banco de pruebas ideal para
medir el beneficio real de cada estrategia:

| Característica | Benchmarks clásicos | Cohete |
|----------------|--------------------|----|
| Tiempo por evaluación | ~1 µs (Sphere) | ~0.3 ms |
| Overhead de IPC amortizable | No (muy barato) | Sí (coste significativo) |
| Vectorizable con NumPy | Sí (trivial) | Sí (no trivial: 240 pasos de integración) |
| Picklable para V2/V5 | Sí | Sí (función a nivel de módulo) |
| Independencia entre partículas | Sí | Sí (paralelismo perfecto) |

Con **30 partículas y 200 iteraciones**, la evaluación secuencial (V0) tarda
aproximadamente 7–8 segundos. V4, al simular las 30 partículas simultáneamente
con una sola matriz NumPy en cada iteración, reduce ese tiempo a menos de 0.5
segundos: un speedup real de ~20×.

### 2.6 Implementación: dos versiones del simulador

Para poder usar el cohete con todos los evaluadores, se implementan dos versiones
del simulador en `pso/objectives/rocket_landing.py`:

**Versión escalar** — `rocket_landing_objective(params: ndarray) -> float`:
Evalúa un único controlador. Ejecuta el bucle de simulación paso a paso, sale
con `break` en cuanto el cohete aterriza, y devuelve un float. Es picklable y
compatible con V0, V1, V2, V3 y V5.

**Versión vectorizada** — `rocket_landing_vec(params_matrix: ndarray) -> ndarray`:
Recibe una matriz `(n, 5)` y simula los *n* controladores **simultáneamente**.
La clave es que el estado del enjambre completo es una matriz `(n, 6)`, y cada
paso de integración aplica operaciones NumPy sobre todas las filas a la vez:

```python
# Empuje para los n controladores a la vez
nom_thrust = thrust_gain * cfg.gravity * cfg.mass    # (n,)
braking    = active & (y < brake_alt)                # (n,) booleano
thrust_cmd = np.clip(nom_thrust + braking * (vert_damp * vy), 0, T_max)

# Ángulo objetivo y torque para todos
target_angle = np.clip(-np.arctan(horiz_corr * x), -π/4, π/4)
torque_cmd   = np.clip(rot_gain*(target_angle - theta) - damp*omega, -tau_max, tau_max)

# Integración Euler vectorizada
states[:, 2] += (Fx / m) * dt   # vx para todos
states[:, 3] += (Fy / m) * dt   # vy para todos
# ...
```

Esta versión, usada por V4, elimina completamente el bucle Python sobre partículas.
Todo el cómputo ocurre en el núcleo C de NumPy, que puede además explotar
instrucciones SIMD (AVX2/SSE4) del procesador.

### 2.7 Registro en el sistema de benchmarks

El cohete se integra en el framework exactamente igual que Sphere o Rastrigin:

```python
ROCKET_BENCHMARK = Benchmark(
    name            = "rocket",
    func            = rocket_landing_objective,   # escalar → V0/V1/V2/V3/V5
    vec_func        = rocket_landing_vec,          # matricial → V4
    lb=0.0, ub=10.0,                              # fallback (no usado)
    bounds_override = ROCKET_BOUNDS,               # límites por dimensión
    known_minimum   = 0.0,                         # aterrizaje perfecto teórico
)
```

El campo `bounds_override` es necesario porque cada parámetro del controlador
tiene un rango diferente (e incluso `vertical_damping` puede ser negativo). La
clase `Benchmark` devuelve la matriz `(5, 2)` de límites reales al llamar a
`bounds_array(d)`, ignorando el argumento `d`.

---

## 3. El algoritmo PSO

PSO es un método de optimización estocástica inspirado en el comportamiento de
bandadas de pájaros. Se mantiene un *enjambre* de partículas, cada una con
posición **x** y velocidad **v** en el espacio de búsqueda (en nuestro caso, el
espacio de 5 parámetros del controlador). En cada iteración:

```
v ← w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)
x ← x + v
```

- `w` (inercia): cuánto se mantiene la trayectoria anterior. Valores altos
  favorecen la exploración; bajos, la explotación.
- `c1` (cognitivo): atracción hacia la mejor posición propia de cada partícula.
- `c2` (social): atracción hacia la mejor posición global del enjambre.
- `r1`, `r2`: vectores aleatorios en [0,1] que añaden estocasticidad.

Aplicado al cohete: cada partícula es un controlador. En la primera iteración,
los 30 controladores se distribuyen aleatoriamente por el espacio de parámetros.
Tras evaluar todos, los que aterrizan mejor influyen en la dirección de búsqueda
del resto del enjambre. Tras pocas decenas de iteraciones, el enjambre converge
hacia la región de parámetros que produce aterrizajes óptimos.

### 3.1 Criterio de parada

El algoritmo se detiene cuando se alcanza `max_iter` o cuando la mejora en el
mejor fitness es inferior a `tol = 1e-8` durante `stagnation_iter = 50`
iteraciones consecutivas.

### 3.2 Gestión de límites

Cuando una partícula propone parámetros fuera de rango (p. ej. `thrust_gain = 3.5`
o `vertical_damping = −3.0`), la estrategia **clamp** los recorta al borde del
dominio y niega y reduce a la mitad la velocidad correspondiente. Esto simula una
"pared absorbente" que frena la partícula sin hacerla rebotar.

### 3.3 Topologías disponibles

- **Global (gBest)**: todas las partículas conocen el mejor controlador global.
  Converge rápido hacia las soluciones buenas.
- **Ring (lBest)**: cada partícula sólo comparte información con sus *k* vecinos.
  Converge más lento pero explora mejor el espacio (útil si la función es muy
  multimodal).

---

## 4. Estrategias de evaluación paralela (V0–V5)

El cuello de botella del PSO es la evaluación del fitness: con el cohete, simular
30 partículas secuencialmente lleva ~0.3 ms × 30 = ~9 ms por iteración, o ~1.8 s
por 200 iteraciones, más el overhead del intérprete. Las seis estrategias abordan
esto de formas radicalmente distintas.

La interfaz común es:

```python
class FitnessEvaluator(ABC):
    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        # positions: (n_partículas, d)  →  fitness: (n_partículas,)
```

El motor PSO no sabe ni le importa qué estrategia usa internamente el evaluador.

---

### V0 — Secuencial (baseline)

```python
return np.array([self.objective(pos) for pos in positions])
```

Un bucle Python puro. Evalúa los 30 controladores uno a uno. Es el punto de
referencia para calcular el speedup de las demás versiones. Para el cohete, cada
llamada a `rocket_landing_objective(pos)` ejecuta el bucle de 240 pasos completo
antes de pasar al siguiente controlador. Tiempo total: ~7–8 s para 30 partículas
× 200 iteraciones.

---

### V1 — Hilos (`ThreadPoolExecutor`)

```python
with ThreadPoolExecutor(max_workers=n) as pool:
    results = list(pool.map(self.objective, positions))
```

Lanza las 30 evaluaciones en *n* hilos concurrentes. El **GIL** (Global
Interpreter Lock) de CPython impide que dos hilos ejecuten bytecode Python
simultáneamente, por lo que el paralelismo real sólo se produce durante las
operaciones NumPy (que liberan el GIL). Para el cohete, cuya simulación es
mayoritariamente NumPy, V1 consigue cierto speedup, pero limitado porque el GIL
se recompite con frecuencia. El overhead de crear y destruir el pool en cada
iteración también se nota.

---

### V2 — Procesos (`ProcessPoolExecutor` + batching)

```python
with ProcessPoolExecutor(max_workers=n) as pool:
    results = list(pool.map(_eval_batch, batch_args))
```

Crea *n* procesos Python independientes, cada uno con su propio intérprete y su
propio GIL. El paralelismo es real: varios simuladores del cohete corren
simultáneamente en distintos núcleos de la CPU. El coste es la **serialización
IPC**: cada controlador (array de 5 floats) se serializa con `pickle`, se envía
al worker por pipe, se evalúa, y el resultado se envía de vuelta.

Para el cohete, donde cada evaluación lleva ~0.3 ms, este overhead IPC es
amortizable. El parámetro `batch_size` agrupa varios controladores por tarea,
reduciendo el número de round-trips.

**Limitación importante:** `ProcessPoolExecutor` **crea y destruye el pool de
procesos en cada llamada a `evaluate()`**, es decir, en cada iteración del PSO.
Eso son 200 creaciones de pool. V5 resuelve este problema.

**Requisito de picklabilidad:** `rocket_landing_objective` está definida a nivel
de módulo (no es un lambda ni una función local), por lo que `pickle` puede
serializarla sin problemas. En Windows, donde `multiprocessing` usa el método
*spawn*, todos los scripts de entrada incluyen la guarda
`if __name__ == '__main__': multiprocessing.freeze_support()`.

---

### V3 — Asyncio (`asyncio.gather`)

```python
async def _gather(self, positions, latencies):
    tasks = [self._evaluate_one(pos, lat) for pos, lat in zip(positions, latencies)]
    results = await asyncio.gather(*tasks)
```

Concurrencia cooperativa en un solo hilo. Asyncio **no paraliza trabajo
CPU-bound** (todas las coroutines comparten el mismo núcleo), por lo que para
el cohete no ofrece ninguna ventaja computacional sobre V0.

Su valor didáctico es modelar el escenario en que la "evaluación de fitness" es
en realidad una llamada a un servicio externo (simulador remoto, API REST, base
de datos). Cada evaluación sufre una latencia I/O aleatoria (`asyncio.sleep`).
Con V0, el tiempo total sería Σ(latencias); con V3, es max(latencias), lo que
puede suponer un speedup de 10–30× cuando hay muchas evaluaciones con latencias
de decenas de milisegundos.

---

### V4 — NumPy vectorizado ⭐

```python
return np.asarray(self.vec_objective(positions), dtype=float)
```

Pasa la matriz de posiciones completa `(30, 5)` directamente a
`rocket_landing_vec`, que simula los 30 controladores **en paralelo implícito**:
una sola llamada NumPy que internamente ejecuta kernels BLAS/SIMD sobre toda la
matriz de estados `(30, 6)`.

**Por qué es tan rápido:** no hay bucle Python sobre partículas, no hay IPC, no
hay creación de hilos ni procesos. Toda la computación ocurre en una secuencia
de operaciones matriciales en C. Los 240 pasos de integración se ejecutan con
operaciones `states[:, 2] += ax * dt` que el procesador puede vectorizar con
instrucciones SIMD sobre los 30 valores simultáneamente.

El speedup medido sobre el cohete es **~20× respecto a V0** (de ~7.7 s a ~0.4 s
para 30 partículas × 200 iteraciones). V4 es consistentemente la estrategia más
rápida para funciones que pueden expresarse con NumPy.

**Precisión numérica:** al evaluar 30 partículas a la vez con operaciones SIMD,
el orden de las operaciones de punto flotante puede diferir del escalar (que
procesa una partícula). Para trayectorias estables esto produce resultados
bit-a-bit idénticos; para trayectorias caóticas puede haber divergencias de
redondeo. Los tests verifican concordancia a 1e-8 usando parámetros estables.

---

### V5 — Joblib (pool persistente)

```python
results = Parallel(n_jobs=-1, backend="loky", batch_size="auto")(
    delayed(self.objective)(pos) for pos in positions
)
```

Como V2 (procesos reales, paralelismo verdadero), pero con dos diferencias clave:

1. **Pool persistente:** el backend `loky` mantiene los workers vivos entre
   llamadas a `evaluate()`. Con V2, cada iteración del PSO crea y destruye un
   pool (spawn de procesos, importación de módulos, etc.). Con V5, ese coste se
   paga una sola vez: la primera llamada. Las siguientes 199 iteraciones reutilizan
   los mismos procesos, enviando sólo los datos.

2. **Gestión automática:** joblib decide internamente cómo agrupar tareas
   (`batch_size="auto"`), gestiona crashes de workers y proporciona mejor
   balance de carga que `ProcessPoolExecutor` para tareas de duración variable.

Para el cohete, donde cada evaluación lleva ~0.3 ms, V5 es significativamente
más eficiente que V2: elimina el overhead de spawn repetido que en V2 puede
dominar sobre el trabajo real.

---

### Comparativa de estrategias aplicadas al cohete

| Estrategia | Mecanismo | Esperado en cohete | Caso ideal de uso |
|------------|-----------|---------------------|-------------------|
| V0 | Bucle Python | Baseline (~7.7 s) | Depuración, funciones baratas |
| V1 | Hilos | Ligera mejora (GIL) | Funciones que liberan GIL |
| V2 | Procesos | Mejora significativa, overhead IPC | Funciones costosas (>1 ms) |
| V3 | Asyncio | Sin mejora CPU-bound | Llamadas I/O externas |
| V4 | NumPy vectorizado | ~20× speedup | Simuladores expresables con NumPy |
| V5 | Joblib loky | >V2, pool persistente | Funciones costosas, muchas iteraciones |

---

## 5. Arquitectura del sistema

### 5.1 Estructura del proyecto

```
p1/
├── pso/
│   ├── core/
│   │   ├── pso.py              Motor PSO (bucle, actualización v/x, parada)
│   │   ├── swarm.py            SwarmState, IterationRecord, PSOResult
│   │   ├── bounds.py           ClampStrategy, ReflectStrategy
│   │   └── topology.py         GlobalTopology, RingTopology
│   ├── objectives/
│   │   ├── benchmarks.py       Sphere/Rosenbrock/Rastrigin/Ackley + *_vec + BENCHMARKS
│   │   ├── rocket_landing.py   Simulador físico + rocket_landing_vec + ROCKET_BENCHMARK
│   │   └── __init__.py         Registra rocket en BENCHMARKS al importar
│   ├── parallel/
│   │   ├── base.py             FitnessEvaluator (ABC)
│   │   ├── v0_sequential.py    SequentialEvaluator
│   │   ├── v1_threading.py     ThreadingEvaluator
│   │   ├── v2_multiprocessing.py  MultiprocessingEvaluator
│   │   ├── v3_asyncio.py       AsyncioEvaluator
│   │   ├── v4_numpy.py         VectorizedEvaluator
│   │   └── v5_joblib.py        JoblibEvaluator
│   ├── experiments/
│   │   ├── runner.py           RunConfig + run_experiment()
│   │   └── grid_search.py      grid_search() con barra de progreso
│   ├── io/
│   │   └── persistence.py      save_result / load_result (YAML+JSON+CSV+NPZ)
│   └── viz/
│       ├── visualizer.py       convergence_plot, swarm_animation_2d/3d, speedup_plot
│       └── rocket_viz.py       plot_trajectory, animate_landing
├── scripts/
│   ├── run_pso.py              CLI de un experimento
│   ├── run_benchmarks.py       Suite completa (todas funciones × dims × seeds × estrategias)
│   ├── run_grid_search.py      Grid search de hiperparámetros
│   ├── run_rocket.py           Demo del cohete: tabla de speedup V0–V5
│   └── make_viz.py             Generador de visualizaciones
├── tests/
│   ├── test_benchmarks.py      23 tests
│   ├── test_pso.py             26 tests
│   └── test_rocket.py          22 tests (71 en total)
├── config/default.yaml
└── pyproject.toml
```

### 5.2 Principio de diseño central

El motor PSO (`pso.py`) **no sabe qué estrategia de evaluación usa**. Sólo llama
a `evaluator.evaluate(positions)` y recibe un array de fitness. Esto permite:

- Cambiar la estrategia sin tocar el algoritmo.
- Probar nuevas estrategias implementando sólo `FitnessEvaluator`.
- Tests de regresión que verifican que todas las estrategias producen los mismos
  resultados matemáticos para la misma semilla.

De igual forma, el evaluador **no sabe qué función objetivo usa**. Recibe un
callable; si es `rocket_landing_objective` o `sphere`, da igual.

### 5.3 Flujo de una ejecución

```
run_pso.py --benchmark rocket --strategy v4 --seed 42
      │
      ▼
RunConfig(benchmark="rocket", strategy="v4", seed=42, ...)
      │
      ▼
run_experiment(cfg)
   ├─ get_benchmark("rocket")  →  ROCKET_BENCHMARK
   ├─ bench.bounds_array(5)    →  ndarray (5,2) con límites reales
   ├─ build_evaluator("v4", rocket_landing_objective,
   │                  vec_objective=rocket_landing_vec)
   │      →  VectorizedEvaluator
   └─ PSO(bounds, evaluator=VectorizedEvaluator, seed=42).run()
         │
         ├── inicializar 30 partículas aleatoriamente en [0.5,3]×[-2,2]×...
         ├── for each iteration:
         │     ├── actualizar v y x (NumPy, siempre secuencial)
         │     ├── evaluator.evaluate(positions)
         │     │     └── rocket_landing_vec(positions)  ← 1 llamada, 30 sims
         │     └── actualizar pbest, gbest
         └── PSOResult(best_position, best_fitness, history, ...)
      │
      ▼
save_result(result, "results/")
   ├── config.yaml  (parámetros + git hash + timestamp)
   ├── result.json  (mejor controlador encontrado)
   └── history.csv  (convergencia por iteración)
```

### 5.4 Persistencia

Cada ejecución genera un directorio `results/<run_id>/`:

| Archivo | Formato | Contenido |
|---------|---------|-----------|
| `config.yaml` | YAML | Todos los parámetros + git hash + timestamp |
| `result.json` | JSON | Mejor posición (5 parámetros), fitness, tiempo, iteraciones |
| `history.csv` | CSV | best_fitness, mean_fitness, eval_s, update_s por iteración |
| `trajectories.npz` | NPZ | Posiciones de todas las partículas (sólo con `--trajectories`) |

---

## 6. Visualización del caso de uso

### 6.1 Trayectoria estática (`plot_trajectory`)

Genera un PNG con:
- **Panel izquierdo**: trayectoria de vuelo en el plano (x, y) coloreada por
  tiempo (azul → rojo), flechas de velocidad cada N pasos, plataforma de
  aterrizaje en (0, 0), marcadores de lanzamiento y aterrizaje.
- **Paneles derechos**: cuatro series temporales (altitud, velocidad total,
  inclinación en grados, combustible restante).

Un aterrizaje exitoso con el controlador óptimo muestra: curva de altitud
suave, velocidad que se anula cerca de y=0, inclinación que converge a 0°,
y combustible que disminuye de forma proporcional al empuje aplicado.

### 6.2 Animación del aterrizaje (`animate_landing`)

Genera un GIF que anima, fotograma a fotograma:
- El **polígono del cohete** (rectángulo + punta cónica) rotado por el ángulo
  real `theta` en cada instante.
- Una **flecha de empuje** cuya longitud es proporcional al thrust real aplicado
  y que apunta en la dirección opuesta al eje del cohete.
- El **rastro de trayectoria** que se va dibujando progresivamente.
- Un **panel de métricas** con tiempo, altitud, velocidad, inclinación y
  combustible actualizados en tiempo real.

Ambas visualizaciones llaman internamente a `simulate_trajectory(params)`, que
ejecuta el mismo simulador físico que la función objetivo, garantizando que lo
que se visualiza es exactamente lo que el PSO optimizó.

---

## 7. Tests

**71 tests unitarios**, todos pasando, distribuidos en tres módulos:

### `test_benchmarks.py` (23 tests)
- Mínimos conocidos exactos (Sphere en 0, Rosenbrock en (1,…,1), etc.).
- Que las funciones aceptan dimensiones arbitrarias sin errores.
- Que no generan NaN para entradas aleatorias.
- Registro `BENCHMARKS` y generación de instancias reproducibles.

### `test_pso.py` (26 tests)
- **Reproducibilidad V0–V5**: mismo seed → mismo fitness final para todos los
  evaluadores (verifica que el cambio de estrategia no altera el resultado
  matemático).
- **Límites**: las posiciones nunca salen del dominio con clamp y con reflect.
- **Monotonicidad**: el mejor global nunca empeora entre iteraciones.
- **Convergencia**: Sphere d=5 llega a fitness < 1e-4.
- **Parada temprana**: el criterio de estancamiento se activa correctamente.
- **Contrato de evaluadores**: todos devuelven arrays `(n,)` idénticos para las
  mismas posiciones de entrada.

### `test_rocket.py` (22 tests)
- **Contrato de salida**: escalar devuelve `float`, vectorizada devuelve `(n,)`,
  todos los valores finitos y no negativos.
- **Concordancia escalar/vectorizada**: para parámetros de trayectoria estable,
  diferencia < 1e-8 entre la versión escalar y la vectorizada con n=1.
- **Picklabilidad**: `rocket_landing_objective` y `rocket_landing_vec` se
  serializan y deserializan con `pickle` sin pérdida de funcionalidad (requerido
  por V2 y V5).
- **Registro**: `get_benchmark("rocket")` devuelve bounds correctos, `vec_func`
  presente y funcional.
- **Sanidad física**: parámetros bien ajustados obtienen mejor fitness que
  parámetros que hacen subir el cohete indefinidamente.
- **Integración PSO**: con V0 (20 partículas, 50 iter), V4 y V5, el PSO
  encuentra controladores que aterrizan (fitness < 1000 = penalización de crash).

---

## 8. Decisiones de diseño y trade-offs

| Decisión | Alternativa | Razonamiento |
|----------|-------------|--------------|
| Snapshot al aterrizar en `rocket_landing_vec` | Estado al final de sim | Sin snapshot, la gravedad post-aterrizaje corrompe el fitness vectorizado vs escalar |
| Euler semi-implícito para la física | Runge-Kutta 4 | Suficientemente preciso a dt=0.05 s; reproducible y más simple |
| `bounds_override` por dimensión en `Benchmark` | Bounds simétricos | El cohete tiene rangos distintos por parámetro (uno incluso negativo) |
| V4: `vec_func` separada de la función escalar | Reescribir la función original | Mantiene V0–V3/V5 compatibles; el escalar es más legible |
| V5: joblib loky vs `multiprocessing.Pool` | Pool manual | Loky gestiona crashes, serialización y balance de carga; más robusto |
| V2: pool recréado en cada `evaluate()` | Pool persistente en V2 | Overhead educativamente valioso para comparar con V5; código más simple |
| `clamp` como bounds strategy por defecto | `reflect` | Comportamiento predecible, sin riesgo de oscilaciones |
| Topología global por defecto | Ring | Mayor velocidad de convergencia en espacios de parámetros de baja dimensión |
| CSV para historial de convergencia | Parquet | Sin dependencias adicionales; legible directamente con cualquier herramienta |

**Limitaciones conocidas:**

- V3 no acelera trabajo CPU-bound como el cohete: su ventaja se limita a I/O.
- V2 recrea el pool en cada iteración; para 200 iteraciones, el overhead puede
  dominar sobre el cómputo para funciones rápidas.
- V4 puede dar resultados ligeramente distintos a V0 en trayectorias caóticas
  del cohete por diferencias de redondeo SIMD. No afecta a la calidad de
  la optimización.
- El simulador usa Euler semi-implícito, que puede acumular error para dt
  grandes, aunque a dt=0.05 s es suficientemente preciso para el problema.

---

## 9. Reproducción de experimentos

```bash
# Instalar (incluye joblib para V5)
pip install -e ".[dev]"

# Tests completos
pytest -v   # 71 tests

# --- Caso de uso del cohete ---

# Demo completo: tabla de speedup V0–V5 + trayectoria del mejor controlador
python scripts/run_rocket.py

# Un único experimento (cohete, V4 vectorizado)
python scripts/run_pso.py --benchmark rocket --strategy v4 --n-particles 30 --max-iter 200 --seed 42

# Visualizar trayectoria y animación del controlador óptimo
python scripts/make_viz.py --mode rocket_traj

# Grid search de hiperparámetros PSO sobre el cohete
python scripts/run_grid_search.py --benchmark rocket --strategy v0 --seeds 3

# --- Benchmarks clásicos ---

# Suite completa: todas las funciones × d=2/10/30 × 5 seeds × V0–V5
python scripts/run_benchmarks.py --strategies v0 v1 v2 v3 v4 v5

# Animación del enjambre en 2D sobre Sphere
python scripts/run_pso.py --benchmark sphere --dim 2 --strategy v0 --trajectories
python scripts/make_viz.py --mode swarm2d --run-dir results/<run_id>

# Curvas de convergencia y speedup a partir de resultados guardados
python scripts/make_viz.py --mode convergence --runs-dir results
python scripts/make_viz.py --mode speedup     --runs-dir results
```
