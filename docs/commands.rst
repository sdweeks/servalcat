Servalcat commands
==================

Run ``servalcat <command> -h`` to see all arguments of a command.

refine_xtal
-----------
Crystallographic refinement (see :doc:`xtal`). ``refine_xtal_norefmac`` is accepted as an alias.

refine_geom
-----------
Geometry optimisation without experimental data.

sigmaa
------
Maximum-likelihood parameter estimation and map calculation from a model and diffraction data.

fw
--
French-Wilson conversion of intensities to amplitudes.

util
----
Utilities (``servalcat util -h``): ``tls_select``, ``tls_bfit`` and ``tls_nma`` (automated selection of TLS groups from the likelihood, from a fit to the B-factors and from normal-mode domains, see :doc:`xtal`), ``occ_groups`` (occupancy groups from alternative conformations, see :doc:`occ_refine`), ``h_add``, ``geom``, ``adp``, ``fcalc``, ``sm2mm``, ``mm2ins``, ``fix_link``, ``merge_models`` and others.
