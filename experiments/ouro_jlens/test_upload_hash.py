"""Check archival Git object hashes independently of HF credentials/network."""
import ast
import hashlib
import subprocess
from pathlib import Path


def test_git_blob_header_uses_a_nul_and_matches_git():
    tree = ast.parse(Path(__file__).with_name('upload_new_lenses.py').read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'git_blob_sha1')
    namespace = {'hashlib': hashlib}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<hash function>', 'exec'), namespace)
    assert namespace['git_blob_sha1'](b'') == 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'
    payload = b'hello\n'
    expected = subprocess.check_output(['git', 'hash-object', '--stdin'], input=payload).decode().strip()
    assert namespace['git_blob_sha1'](payload) == expected
