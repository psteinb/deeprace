
from deeprace.cli import main


def test_main():
    assert main([]) == 1


def test_main_version():
    assert main(['--version']) != 0
