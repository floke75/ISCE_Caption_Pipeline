import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

# Add repo root to path
sys.path.append(os.getcwd())

from scripts import install

class TestInstallScript(unittest.TestCase):
    @patch('scripts.install.run_command')
    @patch('scripts.install.ensure_python_version')
    @patch('scripts.install.build_virtualenv')
    @patch('scripts.install.venv_bin')
    @patch('scripts.install.install_spacy_model')
    @patch('scripts.install.install_frontend_dependencies')
    @patch('scripts.install.check_ffmpeg')
    @patch('scripts.install.summarize')
    @patch('scripts.install.parse_args')
    def test_batched_installation(self, mock_parse_args, mock_summarize, mock_check_ffmpeg,
                                  mock_install_frontend, mock_install_spacy, mock_venv_bin,
                                  mock_build_venv, mock_ensure_python, mock_run_command):

        # Setup mocks
        mock_args = MagicMock()
        mock_args.venv = Path(".venv")
        mock_args.recreate_venv = False
        mock_args.gpu = False
        mock_args.skip_frontend = True
        mock_parse_args.return_value = mock_args

        mock_venv_bin.return_value = Path("/mock/venv/bin/pip")

        # Mock requirements file existence
        with patch('pathlib.Path.exists', return_value=True):
            # Run main
            install.main()

        # Verify run_command calls for requirements
        # We expect calls like: ["/mock/venv/bin/pip", "install", "-r", ".../requirements/core.txt"]

        calls = mock_run_command.call_args_list

        # Helper to find requirement installation calls
        req_calls = []
        for call in calls:
            args, _ = call
            cmd = args[0]
            if len(cmd) > 1 and "install" in cmd and "-r" in cmd:
                req_file = str(cmd[-1])
                req_calls.append(req_file)

        self.assertEqual(len(req_calls), 4, f"Should have installed 4 requirement groups, found {len(req_calls)}: {req_calls}")
        self.assertTrue(req_calls[0].endswith("core.txt"), f"Expected core.txt, got {req_calls[0]}")
        self.assertTrue(req_calls[1].endswith("speech.txt"), f"Expected speech.txt, got {req_calls[1]}")
        self.assertTrue(req_calls[2].endswith("nlp.txt"), f"Expected nlp.txt, got {req_calls[2]}")
        self.assertTrue(req_calls[3].endswith("web.txt"), f"Expected web.txt, got {req_calls[3]}")

if __name__ == '__main__':
    unittest.main()
