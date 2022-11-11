
# Research Project - The Effect of Noise on the Performance of Variational Quantum Eigensolvers

A Qiskit implementation of the experiment depicted in the research paper - [*Research Project - The Effect of Noise on the Performance of Variational Quantum Eigensolvers*](paper.pdf).

A full guide through may be found within the research paper.


## Requirements

As detailed in [requirements.txt](requirements.txt):

```
qiskit==0.39.2
matplotlib==3.5.3
notebook==6.4.12
pylatexenc==2.10
pandas==1.4.4
qiskit-nature==0.4.5
```


Installation of *PyQuante* (`pyquante2_pure==2.0`) is also required but not supported in *pip* (and therefore not listed above), see more at [pyquante2 GitHub repo](https://github.com/rpmuller/pyquante2).

It is possible to avoid installation of *PyQuante* and *Qiskit Nature* by following the instructions within the [hamiltonians.py](hamiltonians.py) module.
