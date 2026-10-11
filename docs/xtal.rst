Crystallographic refinement
===========================

Getting started
---------------
To perform crystallographic refinement with Servalcat, it is necessary to specify an input model (PDB, mmCIF or smCIF), diffraction data (MTZ or CIF format) and radiation source type (xray, neutron or electron). For example:

.. code-block:: console

    $ servalcat refine_xtal \
     --model input.pdb --hklin ../data.mtz \
     -s xray \
     [-o prefix]

Output files:
   * ``prefix.pdb``: refined model (legacy PDB format)
   * ``prefix.mmcif``: refined model (mmCIF format)
   * ``prefix.mtz``: 2Fo-Fc and Fo-Fc maps which can be auto-opened with Coot.

Output logs:
   * ``servalcat.log``
   * ``prefix_stats.json``: refinement statistics per cycle in JSON format

Frequently used options
-----------------------

   * ``--ligand [LIGAND ...]``: Restraint dictionary CIF file(s)
   * ``--ncycle NCYCLE``: Number of refinement cycles. Default: 10
   * ``--weight WEIGHT``: Starting value of the weight for the experimental data term (default: automatically determined from resolution). By default, the weight is further adjusted to achieve bond length rmsZ in the range between 0.5 and 1.0. This can be changed using the option ``--target_bond_rmsz_range MIN_RMSZ MAX_RMSZ``.
   * ``--ncsr``: Use local restraints for non-crystallographic symmetry
   * ``--tls``, ``--tls_per_chain``, ``--tls_file FILE``, ``--tls_ncycle N``: TLS refinement (see below)
   * ``--jellybody``: Use jelly body restraints
   * ``--adp {iso,aniso,fix}``: Atomic displacement parameter (isotropic: 1 parameter per atom, anisotropic: 6 parameters per atom, fixed B-values)
   * ``-d D_MIN, --d_min D_MIN``: High-resolution limit (in Å)
   * ``--d_max D_MAX``: Low-resolution limit (in Å)
   * ``--free FREE``: flag number for test set
   * ``--hydrogen {all,yes,no}``: Hydrogen atoms - ``all``: (re)generate riding hydrogen atoms, ``yes``: use hydrogen atoms if present in input structure model, ``no``: remove hydrogen atoms in input structure model. Default: all.
   * ``--hout``: Write hydrogen atoms in the output model
   * ``--twin``: Twin refinement
   * ``--randomize RANDOMIZE``: Shake coordinates with a specified rms (in Å)
   * ``--bfactor BFACTOR``: Reset all atomic B values to specified value
   * ``--keywords KEYWORDS [KEYWORDS ...]``: Keyword(s) in REFMAC5 syntax. See :doc:`supported Refmac keywords <refmac_keywords>`
   * ``--keyword_file KEYWORD_FILE [KEYWORD_FILE ...]``: File with keyword(s) in REFMAC5 syntax  (e.g. ProSMART restraint file)


TLS refinement
--------------

Translation/libration/screw (TLS) refinement models the collective motion of rigid groups of atoms. The total atomic displacement is

.. math::

    U_{\rm total} = U_{\rm TLS}(\mathbf{r}; T, L, S) + \frac{B_{\rm res}}{8\pi^2} I, \qquad
    U_{\rm TLS} = T + A L A^{\rm T} + A S + S^{\rm T} A^{\rm T},

where :math:`\mathbf{r} = (x, y, z)` is the position relative to the group origin and
:math:`A = \begin{pmatrix} 0 & z & -y \\ -z & 0 & x \\ y & -x & 0 \end{pmatrix}`.
The units are the same as in the PDB, Refmac and Phenix: T in Å\ :sup:`2`, L in deg\ :sup:`2` and S in Å·deg. The origin of each group is its centre of mass and is kept fixed. S has 8 free parameters (its trace has no effect on U).

TLS groups are given in the config file (``--config``), using the same selection syntax as for occupancy groups:

.. code-block:: yaml

   refine:
     tls:
       groups:
         - id: 1
           selections: ["//A/4-80/*"]
         - id: 2
           selections: ["//A/81-164/*"]
       ncycle: 1   # TLS steps per refinement cycle

and TLS refinement is turned on by the groups or with ``--tls``. If no groups are in the config, those stored in the input model (``REMARK 3`` or ``_pdbx_refine_tls``) are used, including their T, L and S values; ``--tls_per_chain`` defines one group per chain. Atoms that are not in any group (e.g. waters) keep their isotropic B.

.. code-block:: console

    $ servalcat refine_xtal --model input.pdb --hklin data.mtz -s xray --config tls.yaml --ncycle 20

How it works (alternating, as in ``refi tlsc`` of Refmac): in each cycle, coordinates, the isotropic *residual* B and occupancies are refined with the TLS parameters fixed; then T, L and S of every group are refined with the atomic parameters fixed. In the second step the gradient and Fisher matrix of the likelihood target with respect to the anisotropic displacement of each atom are projected onto the TLS parameters, and a damped Newton step is taken, followed by a line search. TLS steps that do not decrease the target are rejected. T and L are kept positive semi-definite (``enforce_psd: true`` under ``refine.tls``, the default) and the total U of every atom must stay positive definite. Phenix requires the latter only; with ``enforce_psd: false`` T and L may become unphysical (negative eigenvalues are reported in the log). In a test on lysozyme data at 1.5 Å, the constrained version gave a stable improvement of R-free (0.268 vs 0.278 without TLS) whereas the unconstrained one drifted back towards the isotropic result. Early cycles may report that no TLS step decreased the target: the isotropic B is still too high for TLS to lower it. The ADP restraints act only on the residual B.

Output: ``ANISOU`` records (and ``_atom_site_anisotrop``) contain the **total** U, the B-factor column is its isotropic equivalent, and the TLS groups are written to the ``REMARK 3`` records and the ``_pdbx_refine_tls`` categories ("U values: with TLS added"). If an input model with TLS groups has the *residual-only* convention of Refmac, this is detected from ``U VALUES : RESIDUAL ONLY``; otherwise the B-factors in the input are taken as total.

Continuing after editing the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Together with the model, ``prefix_tls.yaml`` is written. It contains the residue ranges of every TLS group and the refined origin, T, L and S:

.. code-block:: yaml

   refine:
     tls:
       input_b: total
       enforce_psd: true
       groups:
         - id: 1
           selections: ["//A/4-80/*"]
           origin: [17.08, 8.88, 62.87]
           T: [..6 values..]   # T11 T22 T33 T12 T13 T23 (A^2)
           L: [..6 values..]   # L11 L22 L33 L12 L13 L23 (deg^2)
           S: [..9 values..]   # row-major (A deg)

The TLS groups are also written in the model file (``REMARK 3`` and ``_pdbx_refine_tls``) and read from it if the input model has them, but not every program that edits models keeps those records. After rebuilding, continue with

.. code-block:: console

    $ servalcat refine_xtal --model edited.pdb --hklin data.mtz -s xray --tls_file prefix_tls.yaml

This restarts from the previous T, L and S. The B-factors of the model are taken as total (``input_b: total``, as written by Servalcat; ``residual`` if the model has residual B only), so the TLS part is subtracted to get the residual B. Residues that were added, deleted or renumbered are handled by the ranges: new residues inside a range join the group, residues outside every range (ligands, waters) keep an isotropic B, and a group that has lost all its atoms is dropped with a warning. The origin of a group is moved to its current centre of mass if the group has changed (T, L and S are transformed so that the TLS model is exactly the same). ``--tls_file`` replaces ``refine.tls.groups`` of ``--config``. In a test, a restart from a model whose TLS records had been removed started at Rwork 0.0207, where the previous run had stopped (0.0207); without the yaml it started at 0.064.

Keep the groups fixed during a refinement campaign: changing them between rounds makes R-values and the likelihood of different rounds incomparable. Select again only after a major rebuild or if the data change.

Notes and limitations:

   * The residual B is isotropic, so TLS cannot be combined with ``--adp aniso``. Use ``--adp iso`` (default) or ``--adp fix`` (TLS parameters only).
   * TLS refinement usually needs more cycles than default; 20 or more are recommended. Monitor the R-free and the warnings about large libration (L > 20 deg\ :sup:`2`) in the log.
   * Groups are not determined automatically. Rigid domains, or segments of several residues in flexible chains, are common choices.
   * The overall anisotropic scale B is determined by the scaling step as before; the common anisotropic component of the model is not moved into it when TLS is used.
   * The statistics of each TLS step are in ``prefix_stats.json`` (``tls_step`` and ``tls``).

Automated selection of TLS groups
---------------------------------

.. code-block:: console

    $ servalcat util tls_select --model refined.pdb --hklin data.mtz -s xray [-o prefix]

The model should be an isotropic model refined with ``refine_xtal``. Data and model options are those of ``refine_xtal``. Every polymer chain is divided into contiguous segments; groups never combine different chains, and waters and ligands are not included.

**Scan.** For N = 1, 2, ... groups per chain (up to ``--max_groups``, default 8; segments have at least ``--min_residues``, default 10, residues), the best division of each chain is found by dynamic programming. The score of a segment is the decrease of the likelihood target that TLS is predicted to give, a quantity that is cheap to calculate for any segment from the per-atom gradient and Fisher matrix that the TLS refinement uses. Each N is then evaluated by a short refinement (``--tls_cycles``, default 8; coordinates fixed, residual B and TLS refined), with exactly the code of ``refine_xtal``; N = 0 is the same refinement without TLS. The log shows R-values as a function of N:

.. code-block:: text

     N  groups  params  Rwork  Rfree  d_Rfree     -LL  dp_score
     0       0       0 0.0556 0.0650      NaN 89327.5    0.0000
     1       1      20 0.0345 0.0414   0.0236 76419.3 2439.7
     2       2      40 0.0234 0.0284   0.0130 66786.3 3016.4
     3       3      60 0.0230 0.0280   0.0004 66441.3 3120.2
     4       4      80 0.0229 0.0278   0.0002 66254.0 3206.3

(an example with simulated data from two rigid domains), with a bar graph of R-free versus N, a table for CCP4 loggraph, and the files ``prefix_scan.csv``, ``prefix_scan.json`` (also with the residue ranges of every N) and, if matplotlib is installed, ``prefix_scan.png``. ``dp_score`` is used only to rank divisions with the same N; it is not comparable with the change of -LL, because the real refinement is much more effective than the single Newton step it is derived from.

**Recommendation.** The selection is conservative. With R(N) being Rfree (or R if there is no test set) and d(N) = R(N-1) - R(N):

   * TLS is recommended at all only if one group per chain improves R by ``--min_delta`` (default 0.001).
   * The number of groups is the lowest N >= 2 whose jump d(N) is within ``--tie`` (default 25%, since R-differences between neighbouring N are noisy) of the biggest jump among N >= 2, i.e. the lowest number of groups that gives the biggest jump in improvement.
   * If no further group gives a jump of at least ``--min_delta``, one group per chain is recommended.

Residue ranges of the recommendation are written to ``prefix_tls.yaml`` (use it with ``refine_xtal --tls_file``), and those of every other N to ``prefix_tls_N<N>.yaml``. The files of the trial refinements are in ``prefix_trials/``.

**Choosing the number yourself.** With ``--ngroups`` nothing is scanned: the program determines the best residue ranges for the requested numbers and writes them,

.. code-block:: console

    $ servalcat util tls_select ... --ngroups 4            # 4 groups in every polymer chain
    $ servalcat util tls_select ... --ngroups A:4 B:2      # per chain (chains not mentioned: 1 group)

The R-values for this choice are calculated as above unless ``--no_eval`` is given. If a chain is too short for the requested number of groups of ``--min_residues`` residues, the largest possible number is used (with a warning).

The scan uses the same number of groups for every chain (a chain that cannot be divided that far gets as many as possible), so it cannot tell that chains need different numbers; if they do, look at the curve and set the numbers per chain with ``--ngroups``.

Notes: the residue ranges are written as gemmi selections (``//A/4-80/*``), so a non-polymer residue numbered inside a range is included in the group. The scan costs about (N groups + 1) times ``--tls_cycles`` cycles of TLS refinement in total, but the trials are independent and single-threaded, so they run in parallel: one process per available CPU by default, the next trial starting whenever one finishes (``-j``/``--jobs`` sets the number; ``-j 1`` runs them one after another; every process keeps its own copy of the data, so reduce it if memory is short). The results do not depend on the number of processes. Use ``--max_groups`` or ``--tls_cycles`` to shorten the scan. The recommendation is based on an R-value difference measured on the same test set, but test sets are small, so look at the whole curve and at ``--ngroups`` results before deciding, and prefer fewer groups at lower resolution.

Sequence assignment and mismatch feedback
-----------------------------------------

By default SEQRES and ``label_seq_id`` are generated from the modelled residues only. With ``--seq`` (FASTA file(s) with protein, DNA and/or RNA sequences) the best matching sequence is assigned to every polymer chain, so that unmodelled residues, tags and gaps are represented in SEQRES and the mmCIF numbering. A sequence is accepted if at least ``--seq_min_identity`` (default 0.8) of the modelled residues agree with it; sequences with ``U`` are not used for DNA and sequences with ``T`` are not used for RNA.

The log lists the assignment and every residue that differs from the sequence (chain/number, model residue, sequence residue). Modified residues such as MSE count as the residue they replace and unknown (UNK) residues are not counted. A stretch built out of register is reported as a warning: sequence skipped between bonded residues, modelled residues without a counterpart in the sequence, or a stretch that fits the sequence after a shift.

Input columns for diffraction data
----------------------------------

In the required ``--hklin`` option, it is possible to provide merged or unmerged diffraction data (MTZ or CIF format).
If there are multiple columns available in the input file, mean amplitudes (Friedel pairs averaged) are used by default. Columns named ``F``, ``SIGF`` and ``FreeR_flag`` are preferred when present; otherwise the first suitable columns are chosen by column type.

To specify which columns to use, use the ``--labin`` option. For example, the file ``data_merged.mtz`` contains the following columns with **merged** diffraction data:

``H K L FreeR_flag IMEAN SIGIMEAN I(+) SIGI(+) I(-) SIGI(-) FP SIGFP F(+) SIGF(+) F(-) SIGF(-)``

Servalcat would select to use the ``FP, SIGFP,FreeR_flag`` columns by default (refinement against mean structure factor amplitudes).
Anyway, we can specify to use intensities or separate Friedel pairs as follows:

  * ``--labin I(+),SIGI(+),I(-),SIGI(-),FreeR_flag`` (refinement against intensities, separate Friedel pairs)
  * ``--labin IMEAN,SIGIMEAN,FreeR_flag`` (refinement against mean intensities)
  * ``--labin F(+),SIGF(+),F(-),SIGF(-),FreeR_flag`` (refinement against amplitudes, separate Friedel pairs)
  * ``--labin FP, SIGFP,FreeR_flag`` (refinement against mean amplitudes, selected by default)

If the separate Friedel pairs are specified, anomalous difference density map (``FAN`` and ``PHAN``) columns will be present in the output MTZ file.
Note that the anomalous signal is used only for the map calculation but not for the actual refinement.

If the column for **unmerged** intensities is specified, Servalcat merges the data internally and refines against merged intensities. In the log files, the CC* statistic is also available (described below).
An MTZ or CIF file with free flags can be specified with the ``--hklin_free`` option. A particular column for free flags in this file can be specified with the ``--labin_free`` option.


Logs and statistics
-------------------

Refinement progress is monitored by several statistics that quantify the agreement of the refined model with the experimental diffraction data as well as its quality in respect to expected geometry.
The statistics are written in the log file ``servalcat.log`` and are also available in the JSON format ``output_prefix_stats.json``, which is updated every cycle.


Model agreement with data
~~~~~~~~~~~~~~~~~~~~~~~~~

Which model quality statistics are calculated depends on the nature of the input reflection data.

When amplitudes are used for refinement and free R flags are provided for the test set, the conventional ``Rwork`` and ``Rfree`` values are provided. Moreover, the correlation coefficients ``CCFwork`` and ``CCFfree`` between experimentally observed amplitudes and amplitudes calculated based on the refined model are calculated. If free R flags are not available, only ``R`` (sometimes called ``Rall``) and ``CCF`` are reported.

When intensities (or unmerged diffraction data) are given and free R flags are provided, ``R1work`` and ``R1free`` values are provided, as is common in crystallography of small molecules. These statistics are calculated as R-values between the square roots of observed and calculated intensities, considering only reflections with an intensity-to-sigma ratio above 2. The correlation coefficients ``CCIwork`` and ``CCIfree`` between experimentally observed intensities and intensities calculated based on the refined model are also calculated. If free R flags are not provided, only ``R1`` and ``CCI`` are available.

If unmerged data are used, ``CC*`` statistic is also reported. It estimates the data quality and represents an upper limit for the ``CCI`` statistics. See `Karplus and Diederichs (2012) <https://doi.org/10.1126/science.1218231>`_ or `Diederichs and Karplus (2013) <https://doi.org/10.1107/S0907444913001121>`_.


Model agreement with ideal geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The geometry of the refined model is assessed against the restraint dictionary (monomer library).
Root mean square deviations (RMSD) from expected bond lengths and angles are calculated, as well as  their Z-scores (RMSZ), which represent how many standard deviations the observed geometry deviates from the ideal values. 

Importantly, various kinds of individual model geometry outliers with a Z-score greater than 5 are reported in the last cycle of refinement in ``servalcat.log`` and ``output_prefix_stats.json``. It is recommended to inspect these outliers as they can indicate where the model may need improvement.


Radiation sources
-----------------

The radiation source is X-ray by default. Other sources are selected with the ``-s`` or ``--source`` option (neutron, electron, custom); see :doc:`source`.

When performing refinement against neutron diffraction data, it is possible to refine deuterium fraction using the option ``--refine_dfrac``.
In this case, an extra output file ``output_prefix_expanded.mmcif`` is created for the purpose of deposition to the PDB. The bond lengths and their sigmas from ``_chem_comp_bond.value_dist_nucleus`` and ``_chem_comp_bond.value_dist_nucleus_esd`` are used.


Small molecules
---------------

Servalcat can also refine small molecules against crystallographic data. A generally suggested protocol is as follows:

.. code-block:: console

    $ servalcat refine_xtal \
     --model small_molecule.cif --hklin small_molecule.hkl \
     -s xray --unrestrained --hydrogen no --adp aniso --no_solvent \
     [-o prefix]

For ``--model`` and ``--hklin``, common formats for small molecule crystallography (.hkl, .cif, .res, .ins) can be used. This will run unrestrained refinement with anisotropic ADPs without bulk solvent correction. Note that the Servalcat does not currently support riding hydrogen atoms for the unrestrained refinement. If you require riding hydrogen atoms, you will need to use restrained refinement using dictionaries generated by AceDRG.

Servalcat provides a helper command which converts files between the small molecule formats to .pdb, .mmcif or .mtz:

.. code-block:: console

    $ servalcat util sm2mm structure_and_data.cif -o output


Complete list of options
------------------------

.. code-block:: console

    $ servalcat refine_xtal --help
    usage: servalcat refine_xtal [-h] --hklin HKLIN [--hklin_free HKLIN_FREE] [-d D_MIN] [--d_max D_MAX] [--nbins NBINS]
                                        [--nbins_ml NBINS_ML] [--labin LABIN] [--labin_free LABIN_FREE] [--free FREE] --model MODEL
                                        [--monlib MONLIB] [--ligand [LIGAND ...]] [--newligand_continue] [--hydrogen {all,yes,no}] [--hout]
                                        [--jellybody] [--jellybody_params sigma dmax] [--jellyonly] [--find_links]
                                        [--keywords KEYWORDS [KEYWORDS ...]] [--keyword_file KEYWORD_FILE [KEYWORD_FILE ...]]
                                        [--randomize RANDOMIZE] [--ncycle NCYCLE] [--weight WEIGHT] [--no_weight_adjust]
                                        [--target_bond_rmsz_range TARGET_BOND_RMSZ_RANGE TARGET_BOND_RMSZ_RANGE] [--ncsr]
                                        [--adpr_weight ADPR_WEIGHT] [--occr_weight OCCR_WEIGHT] [--bfactor BFACTOR] [--fix_xyz]
                                        [--adp {fix,iso,aniso}] [--refine_all_occ] [--max_dist_for_adp_restraint MAX_DIST_FOR_ADP_RESTRAINT]
                                        [--adp_restraint_power ADP_RESTRAINT_POWER] [--adp_restraint_exp_fac ADP_RESTRAINT_EXP_FAC]
                                        [--adp_restraint_no_long_range] [--adp_restraint_mode {diff,kldiv}] [--unrestrained] [--refine_h]
                                        [--refine_dfrac] [--twin] [-s {electron,xray,neutron,custom}] [--no_solvent] [--use_in_est {all,work,test}]
                                        [--keep_charges] [--keep_entities] [--allow_unusual_occupancies] [-o OUTPUT_PREFIX]
                                        [--write_trajectory] [--vonmises] [--prefer_intensity] [--use_fw] [--config CONFIG]

    program to refine crystallographic structures

    optional arguments:
    -h, --help            show this help message and exit
    --hklin HKLIN
    --hklin_free HKLIN_FREE
                            Input MTZ file for test flags
    -d D_MIN, --d_min D_MIN
    --d_max D_MAX
    --nbins NBINS         Number of bins for statistics (default: auto)
    --nbins_ml NBINS_ML   Number of bins for ML parameters (default: auto)
    --labin LABIN         F,SIGF,FREE input
    --labin_free LABIN_FREE
                            MTZ column of --hklin_free
    --free FREE           flag number for test set
    --model MODEL         Input atomic model file
    --monlib MONLIB       Monomer library path. Default: $CLIBD_MON
    --ligand [LIGAND ...]
                            restraint dictionary cif file(s)
    --newligand_continue  Make ad-hoc restraints for unknown ligands (not recommended)
    --hydrogen {all,yes,no}
                            all: add riding hydrogen atoms, yes: use hydrogen atoms if present, no: remove hydrogen atoms in input. Default:
                            all
    --hout                write hydrogen atoms in the output model
    --jellybody           Use jelly body restraints
    --jellybody_params sigma dmax
                            Jelly body sigma and dmax (default: [0.01, 4.2])
    --jellyonly           Jelly body only (experimental, may not be useful)
    --find_links          Automatically add links
    --keywords KEYWORDS [KEYWORDS ...]
                            refmac keyword(s)
    --keyword_file KEYWORD_FILE [KEYWORD_FILE ...]
                            refmac keyword file(s)
    --randomize RANDOMIZE
                            Shake coordinates with specified rmsd
    --ncycle NCYCLE       number of CG cycles (default: 10)
    --weight WEIGHT       refinement weight (default: auto)
    --no_weight_adjust    Do not adjust weight during refinement
    --target_bond_rmsz_range TARGET_BOND_RMSZ_RANGE TARGET_BOND_RMSZ_RANGE
                            Bond rmsz range for weight adjustment (default: [0.5, 1.0])
    --ncsr                Use local NCS restraints
    --adpr_weight ADPR_WEIGHT
                            ADP restraint weight (default: 1.000000)
    --occr_weight OCCR_WEIGHT
                            Occupancy restraint weight (default: 0.000000)
    --bfactor BFACTOR     reset all atomic B values to specified value
    --fix_xyz
    --adp {fix,iso,aniso}
    --refine_all_occ
    --max_dist_for_adp_restraint MAX_DIST_FOR_ADP_RESTRAINT
    --adp_restraint_power ADP_RESTRAINT_POWER
    --adp_restraint_exp_fac ADP_RESTRAINT_EXP_FAC
    --adp_restraint_no_long_range
    --adp_restraint_mode {diff,kldiv}
    --unrestrained        No positional restraints
    --refine_h            Refine hydrogen (default: restraints only)
    --refine_dfrac        Refine deuterium fraction (neutron only)
    --twin                Turn on twin refinement
    -s {electron,xray,neutron,custom}, --source {electron,xray,neutron,custom}
                          Radiation source (default: xray)
    --no_solvent          Do not consider bulk solvent contribution
    --use_in_est {all,work,test}
                            Which set of reflections to use for the ML parameter estimation. Default: 'work' if --twin is set; otherwise
                            'test'.
    --keep_charges        Use scattering factor for charged atoms. Use it with care.
    --keep_entities       Do not override entities
    --allow_unusual_occupancies
                            Allow negative or more than one occupancies
    -o OUTPUT_PREFIX, --output_prefix OUTPUT_PREFIX
    --write_trajectory    Write all output from cycles
    --vonmises            Experimental: von Mises type restraint for angles
    --prefer_intensity
    --use_fw              For debugging purpose; use F&W-converted amplitudes but use intensity for stats
    --config CONFIG       Config file (.yaml)

    $ servalcat --version
    Servalcat 0.4.123 with Python 3.9.18 (gemmi 0.7.3, scipy 1.7.3, numpy 1.19.5, pandas 1.3.5)
