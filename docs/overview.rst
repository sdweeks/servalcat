Overview
========

Servalcat refines atomic models against X-ray (and neutron / electron) crystallographic data, without the REFMAC5 binary.

* refinement (``refine_xtal``): amplitude or intensity based maximum-likelihood refinement of
  coordinates, isotropic or anisotropic ADPs, TLS groups, occupancies and (for neutron data) deuterium fractions.
  The old name ``refine_xtal_norefmac`` still works.
* ML parameter estimation and map calculation (``sigmaa``)
* French-Wilson conversion of intensities (``fw``)
* geometry optimisation (``refine_geom``)
* utilities (``util``)

The basic usage is:

.. code-block:: console

    $ servalcat <command> <args>

To see all arguments for a specific command, run:

.. code-block:: console

    $ servalcat <command> -h


Refinement
----------
To perform crystallographic refinement with Servalcat, it is necessary to specify an input model (PDB, mmCIF or smCIF), diffraction data (MTZ or CIF format) and type radiation source (xray, neutron or electron).

.. code-block:: console

    $ servalcat refine_xtal \
     --model input.pdb --hklin ../data.mtz \
     -s xray --ncycle 10 \
     [-o output_prefix]

Output files:
   * ``prefix.pdb``: refined model (legacy PDB format)
   * ``prefix.mmcif``: refined model (mmCIF format)
   * ``prefix.mtz``: 2Fo-Fc and Fo-Fc maps which can be auto-opened with coot.

More details about crystallographic refinement with Servalcat can be found `on a separate page <xtal.html>`_.
