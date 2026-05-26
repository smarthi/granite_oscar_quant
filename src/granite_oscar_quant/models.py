"""Model identifiers used by the Granite/OScaR command-line tools.

Keeping the default model id in one module avoids a slow kind of drift where
the README, CLI, tests, and examples silently point at different baselines.
"""

DEFAULT_GRANITE_MODEL_ID = "ibm-granite/granite-4.0-1b-base"
