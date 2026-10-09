"""prl: infrastructure for the Practical Reinforcement Learning workshop.

Environments, logged datasets, evaluation, plotting, checkpoints, runtime
detection, QUICK settings and the run recorder. No learning algorithms:
participants write those in the notebooks.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("prl")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

#: Version of the notebook-facing API. A notebook built for a different API
#: version refuses to run (see prl.lab.init).
API = 1

__all__ = ["API", "__version__"]
