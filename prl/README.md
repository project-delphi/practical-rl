# prl

Infrastructure for the **Practical Reinforcement Learning** workshop
(<https://project-delphi.github.io/practical-rl/>).

`prl` holds what every lab shares: environments, logged datasets, evaluation
(seeded runs, bootstrap confidence intervals, IQM), plotting, checkpoint
functions, runtime detection, QUICK settings and the run recorder.

It deliberately contains **no learning algorithms**. Participants write those
in the notebooks; later labs include a reference copy of earlier lab code.

Install on Colab (the notebooks do this for you):

```bash
pip install "git+https://github.com/project-delphi/practical-rl@<ref>#subdirectory=prl"
```

Code: MIT. See the repository for the workshop text license (CC BY 4.0).
