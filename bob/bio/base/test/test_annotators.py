import tempfile
import os
import shutil
from click.testing import CliRunner
from bob.bio.base.script.annotate import annotate
from bob.db.base import read_annotation_file


def test_annotate():

    try:
        tmp_dir = tempfile.mkdtemp(prefix="bobtest_")
        runner = CliRunner()
        result = runner.invoke(annotate, args=(
            '-d', 'dummy', '-a', 'dummy', '-o', tmp_dir))
        assertion_error_message = (
            'Command exited with this output: `{}\' \n'
            'If the output is empty, you can run this script locally to see '
            'what is wrong:\n'
            'bin/bob bio annotate -vvv --force -d dummy -a dummy -o /tmp/temp_annotations'
            ''.format(result.output))
        assert result.exit_code == 0, assertion_error_message

        # test if annotations exist
        for dirpath, dirnames, filenames in os.walk(tmp_dir):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                annot = read_annotation_file(path, 'json')
                assert annot['topleft'] == [0, 0]
                # size of atnt images
                assert annot['bottomright'] == [112, 92]
    finally:
        shutil.rmtree(tmp_dir)