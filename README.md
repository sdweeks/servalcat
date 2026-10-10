# Servalcat
[![Build](https://github.com/keitaroyam/servalcat/workflows/CI/badge.svg)](https://github.com/keitaroyam/servalcat/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/servalcat?color=blue)](https://pypi.org/project/servalcat/)

**S**tructur**e** **r**efinement and **val**idation for **c**rystallography and single p**a**r**t**icle analysis

This fork is reduced to **X-ray crystallographic refinement without the REFMAC5 binary**
(it can also refine against neutron or electron diffraction data).

Main commands:
  * `servalcat refine_xtal`: crystallographic refinement (maximum likelihood; coordinates, isotropic ADPs or isotropic residual B + TLS, anisotropic ADPs, occupancies, twinning); `refine_xtal_norefmac` is kept as an alias
  * `servalcat refine_geom`: geometry optimisation
  * `servalcat sigmaa`: ML parameter estimation and map calculation
  * `servalcat fw`: French-Wilson conversion of intensities
  * `servalcat util`: utilities

## Installation

```shell
# PyPI
pip install servalcat

# Conda
conda install conda-forge::servalcat
```
will install the stable version.

Starting from ver. 0.4.146, a pre-release version of GEMMI is bundled with Servalcat by default.
The current version in use is 0.7.6-dev ([v0.7.5-181-g92e8cfa8](https://github.com/project-gemmi/gemmi/commit/92e8cfa80cf10bd0d0927946da9f6a3aaf998c85)).

CCP4 and REFMAC5 are not needed. You need [the monomer library](https://github.com/MonomerLibrary/monomers) (or CCP4's copy; set `$CLIBD_MON` or use `--monlib`) for restrained refinement.

### Installing the latest (unstable) version

1. Go to https://github.com/keitaroyam/servalcat/actions/workflows/ci.yml and click on the latest (top) one
2. Download "wheels2" from the Artifacts section (note: a GitHub account is required)
3. Unzip the file and provide the path to the directory using the `--find-links` option: `python -mpip install -U --find-links=wheels2 servalcat`

## Usage
Please read the documentation: https://servalcat.readthedocs.io/en/latest/

## References
* [Yamashita, K., Wojdyr, M., Long, F., Nicholls, R. A., Murshudov, G. N. (2023) "GEMMI and Servalcat restrain REFMAC5" *Acta Cryst.* D**79**, 368-373](https://doi.org/10.1107/S2059798323002413)
* [Yamashita, K., Palmer, C. M., Burnley, T., Murshudov, G. N. (2021) "Cryo-EM single particle structure refinement and map calculation using Servalcat" *Acta Cryst. D***77**, 1282-1291](https://doi.org/10.1107/S2059798321009475)
