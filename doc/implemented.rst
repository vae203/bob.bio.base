.. _bob.bio.base.implemented:

=================================
Tools implemented in bob.bio.base
=================================

Summary
-------

Base Classes
~~~~~~~~~~~~

.. autosummary::
   bob.bio.base.preprocessor.Preprocessor
   bob.bio.base.extractor.Extractor
   bob.bio.base.algorithm.Algorithm
   bob.bio.base.grid.Grid
   bob.bio.base.annotator.Annotator
   bob.bio.base.baseline.Baseline


Implementations
~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.base.preprocessor.Filename
   bob.bio.base.preprocessor.SequentialPreprocessor
   bob.bio.base.preprocessor.ParallelPreprocessor
   bob.bio.base.preprocessor.CallablePreprocessor
   bob.bio.base.extractor.SequentialExtractor
   bob.bio.base.extractor.ParallelExtractor
   bob.bio.base.extractor.CallableExtractor
   bob.bio.base.extractor.Linearize
   bob.bio.base.algorithm.Distance
   bob.bio.base.algorithm.PCA
   bob.bio.base.algorithm.LDA
   bob.bio.base.algorithm.PLDA
   bob.bio.base.algorithm.BIC
   bob.bio.base.database.BioFile
   bob.bio.base.database.BioDatabase
   bob.bio.base.database.ZTBioDatabase
   bob.bio.base.database.FileListBioDatabase
   bob.bio.base.annotator.FailSafe
   bob.bio.base.annotator.Callable

Preprocessors
-------------

.. automodule:: bob.bio.base.preprocessor

Extractors
----------

.. automodule:: bob.bio.base.extractor

Algorithms
----------

.. automodule:: bob.bio.base.algorithm

Databases
---------

.. automodule:: bob.bio.base.database

Grid Configuration
------------------

.. automodule:: bob.bio.base.grid

.. data:: PREDEFINED_QUEUES

   A dictionary of predefined queue keywords, which are adapted to the Idiap_ SGE.


   .. adapted from http://stackoverflow.com/a/29789910/3301902 to ge a nice dictionary content view

Annotators
----------

.. automodule:: bob.bio.base.annotator


Baselines
---------

.. automodule:: bob.bio.base.baseline


.. include:: links.rst
