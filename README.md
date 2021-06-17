# Post-selection inference with HSIC-Lasso

This repository implements a post-selection inference procedure for feature selection with HSIC-Lasso based on the polyhedral lemma.

### Versions and Usage
`master_thesis_version`: This code implements the procedure, simulations and real-world experiments of my Master thesis *Model uncertainty in statistical inference*. It is not entirely tested and might contain errors.

`paper_version`: This folder contains code and experiments from the paper [Post-selection inference with HSIC-Lasso](https://arxiv.org/abs/2010.15659) (to be published in ICML 2021).

The procedure is implemented with a focus on demonstrating its feasibility and analysing its behaviour. Therefore, some methods contain attributes that practitioners do not need when analysing data, naming is not always concise, the division of code blocks into classes and methods can be improved etc. For this reason, we ask the user to be cautious when applying our code.

### Requirements
`numpy`, `matplotlib`, `pandas`, `scipy`, `sklearn`, `joblib`, `covar`,
[`mskernel`](https://github.com/jenninglim/multiscale-features)
