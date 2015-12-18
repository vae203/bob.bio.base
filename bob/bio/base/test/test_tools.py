import bob.bio.base
import tempfile
import os
import shutil

def _test_file_selector(command_line, subdir, all_files, train_files, train_clients):
  tempdir = tempfile.mkdtemp(prefix="bobtest_")
  resdir = tempfile.mkdtemp(prefix="bobtest_")

  try:
    base_command_line = [
      '--database', 'dummy',
      '--preprocessor', 'dummy',
      '--extractor', 'dummy',
      '--algorithm', 'dummy',
      '--sub-directory', 'dummy',
      '--temp-directory', tempdir,
      '--result-directory', resdir,
      '--dry-run', '-vvv'
    ]

    command_line = base_command_line + command_line

    parsers = bob.bio.base.tools.command_line_parser()
    args = bob.bio.base.tools.initialize(parsers, command_line)

    fs = bob.bio.base.tools.FileSelector.instance()

    assert isinstance(fs.database, bob.bio.base.database.DatabaseBob)

    # check all files
    groups = bob.bio.base.tools.groups(args)
    files = fs.original_data_list(groups)
    annots = fs.annotation_list(groups)
    assert len(files) == all_files, len(files)
    assert len(annots) == all_files

    preprocessed = fs.preprocessed_list(groups)
    assert len(preprocessed) == all_files
    assert all(p.startswith(os.path.join(tempdir, 'dummy', 'preprocessed')) and p.endswith('.hdf5') for p in preprocessed)

    extracted = fs.extracted_list(groups)
    assert len(extracted) == all_files
    assert all(p.startswith(os.path.join(tempdir, 'dummy', subdir, 'extracted')) and p.endswith('.hdf5') for p in extracted), extracted[0]

    projected = fs.projected_list(groups)
    assert len(extracted) == all_files
    assert all(p.startswith(os.path.join(tempdir, 'dummy', subdir, 'projected')) and p.endswith('.hdf5') for p in projected), projected[0]

    # check training files
    training_simple = fs.training_list('extracted', 'train_projector', False)
    assert len(training_simple) == train_files
    training_by_client = fs.training_list('projected', 'train_enroller', True)
    assert len(training_by_client) == train_clients

    # check enroll files
    model_ids = fs.model_ids('dev')
    assert len(model_ids) == 20
    assert all(len(fs.enroll_files(m, 'dev', 'extracted')) == 5 for m in model_ids)
    assert all(fs.model_file(m, 'dev').startswith(os.path.join(tempdir, 'dummy', 'Default', 'models', 'dev')) for m in model_ids)

    # check client ids
    assert all(fs.client_id(m, 'dev') == m for m in model_ids)

    # check probe files
    assert len(fs.probe_objects('dev')) == 100
    assert all(len(fs.probe_objects_for_model(m, 'dev')) == 100 for m in model_ids)

    # check ZT-norm
    t_model_ids = fs.t_model_ids('dev')
    assert len(t_model_ids) == 20
    assert all(len(fs.t_enroll_files(m, 'dev', 'extracted')) == 5 for m in t_model_ids)
    assert all(fs.t_model_file(m, 'dev').startswith(os.path.join(tempdir, 'dummy', 'Default', 'tmodels', 'dev')) for m in t_model_ids)

    assert all(fs.client_id(m, 'dev', True) == m for m in t_model_ids)
    assert len(fs.z_probe_objects('dev')) == 100

    # check result files
    assert all(fs.no_norm_file(m, 'dev').startswith(os.path.join(resdir, 'dummy', 'Default', 'nonorm', 'dev')) for m in model_ids)
    assert all(fs.zt_norm_file(m, 'dev').startswith(os.path.join(resdir, 'dummy', 'Default', 'ztnorm', 'dev')) for m in t_model_ids)

    assert fs.no_norm_result_file('dev') == os.path.join(resdir, 'dummy', 'Default', 'nonorm', 'scores-dev')
    assert fs.zt_norm_result_file('dev') == os.path.join(resdir, 'dummy', 'Default', 'ztnorm', 'scores-dev')

  finally:
    if os.path.exists(tempdir):
      shutil.rmtree(tempdir)
    if os.path.exists(resdir):
      shutil.rmtree(resdir)


def test_file_selector_training():
  # Tests the file selector, in case we train only on the training set (the default)
  _test_file_selector([], '.', 400, 200, 20)

def test_file_selector_models():
  # Tests the file selector in case we train only on the enroll files
  _test_file_selector(['--train-on-enroll', 'only'], 'Default', 200, 100, 20)

def test_file_selector_both():
  # Tests the file selector in case we train only on both training and enroll files
  _test_file_selector(['--train-on-enroll', 'add'], 'Default', 400, 300, 40)
