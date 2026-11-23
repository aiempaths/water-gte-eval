"""water-gte-eval: reproducible evaluation harness for a Geometric Trust Engine (GTE).

The engine itself is a black box accessed only through
:class:`water_gte_eval.engine.base.EngineAdapter`. The harness ships with a
deterministic MockEngine so experiments run end-to-end before the real engine
is plugged in.
"""

__version__ = "0.1.0"
