"""PSO — Particle Swarm Optimization with parallel strategies (V0-V3)."""
import logging

from pso.core.pso import PSO
from pso.core.swarm import PSOResult

__version__ = "0.3.0"
__all__ = ["PSO", "PSOResult"]

# Configure library-level logger (NullHandler so it's silent unless the
# application configures logging — best practice for library code).
logging.getLogger("pso").addHandler(logging.NullHandler())
