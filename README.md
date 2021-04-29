# POD-RBF NIROM

A collection of high-fidelity (HFM) snapshots are generated that sufficiently capture the time-dynamics of the simulation. POD is adopted to define a reduced basis space for the high-fidelity snaphosts. Radial basis function (RBF) interpolation is employed to approximate the time dynamics in the reduced space spanned by the POD modes. A family of greedy algorithms are proposed to select an optimal set of RBF collocation points for the construction of the interpolant. 

### Citation

If you find this repo useful for your research, please consider citing:

**A greedy non-intrusive reduced order model for shallow water equations**,

S. Dutta, M. Farthing, E. Perracchione, G. Savant, M. Putti, (2021),

*Journal of Computational Physics, [Journal Link](https://doi.org/10.1016/j.jcp.2021.110378)*

[ArXiV preprint](https://arxiv.org/abs/2002.11329).

```
@article{DUTTA2021,
title = {A greedy non-intrusive reduced order model for shallow water equations},
journal = {Journal of Computational Physics},
pages = {110378},
year = {2021},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2021.110378},
author = {Sourav Dutta and Matthew W. Farthing and Emma Perracchione and Gaurav Savant and Mario Putti},
}
```


## Getting Started

The core modules are in the *podrbf* subdirectory. The *notebooks* subdirectory contains a jupyter notebook that implements two different shallow water examples using AdH as the high-fidelity solver. The *data* subdirectory contains high-fidelity snapshots and mesh files for the shallow water example problems. Some additional data files are available at [Link](https://drive.google.com/drive/folders/1yhudg8RPvwV9SJx9CTqANEnyN55Grzem?usp=sharing).

### Dependencies

* python 2.x or 3.x
* matplotlib
* numpy
* scipy
* jupyter

### Installation

* Install the dependencies
* Clone this repo

```
git clone git@github.com:erdc/podrbf_nirom.git
cd podrbf_nirom
```

### Examples
```
cd notebooks
jupyter notebook
```


Shallow water equations (Red River)
:-----:
<p align="center">
    <img align = 'center' height="400" src="figures/red_river_h_t1.778_psr.png?raw=true">
</p>

p-greedy centers
:-----:
<img align = 'center' height="100" src="figures/red_river_p_greedy_centers_600.png?raw=true">

f-greedy centers
:-----:
<img align = 'center' height="100" src="figures/red_river_f_greedy_centers_600.png?raw=true">

psr-greedy centers
:-----:
<img align = 'center' height="100" src="figures/red_river_psr_greedy_centers_600.png?raw=true">




## Authors

* **Sourav Dutta** - *sourav.dutta@erdc.dren.mil* - ERDC-CHL
* **Matthew Farthing** - *matthew.w.farthing@erdc.dren.mil* - ERDC-CHL


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thank you to all the co-authors, Dr. Stacey Howington, Dr. Hwai-Ping Cheng, Dr. Corey Trahan, and the reviewers for the valuable suggestions. 
* Thank you to ERDC-HPC facilities for support with valuable computational infrastructure
* Thank you to ORISE for support with appointment to the Postgraduate Research Participation Program.
