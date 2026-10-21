"""Exercise the shell sampler using tiny files; no image/model dependencies."""

from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_frames.sh"


class ExtractFramesTests(unittest.TestCase):
    def run_script(self, *arguments, cwd=None):
        return subprocess.run(
            ["bash", str(SCRIPT), *map(str, arguments)],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=10,
        )

    def test_help_does_not_require_source(self):
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SOURCE_DIR", result.stdout)

    def test_source_is_required(self):
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source directory is required", result.stderr)

    def test_rejects_missing_or_invalid_option_values(self):
        cases = [
            ("-o",), ("--output", ""), ("-o", "--help"),
            ("-n",), ("-n", "0"), ("-n", "-2"), ("-n", "1.5"),
            ("-n", "abc"), ("-n", "999999999999999999999999"),
            ("--unknown",),
        ]
        with tempfile.TemporaryDirectory() as source:
            for arguments in cases:
                with self.subTest(arguments=arguments):
                    result = self.run_script(source, *arguments)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Error:", result.stderr)

    def test_rejects_missing_and_empty_source_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            empty = Path(temporary)
            for source in (empty, empty / "missing"):
                with self.subTest(source=source):
                    result = self.run_script(source)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Error:", result.stderr)

    def test_uniform_sampling_preserves_order_and_file_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source frames"
            source.mkdir()
            names = ["00.png", "01.JPG", "02.jpeg", "03.jpg", "04.PNG"]
            for name in names:
                (source / name).write_bytes(name.encode())
            (source / "ignore.txt").write_text("not an image")
            output = base / "output frames"
            result = self.run_script(source, "--output", output, "-n", "03")
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = {f"frame_{index:03d}_{name}": name.encode() for index, name in enumerate(names[::2])}
            actual = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(actual, expected)

    def test_one_frame_uses_middle_and_count_is_capped(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "input"
            nested = source / "nested"
            nested.mkdir(parents=True)
            for index in range(3):
                (nested / f"{index}.jpeg").write_text(str(index))
            one = base / "one"
            result = self.run_script(source, "-o", one, "-n", "1")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((one / "frame_000_1.jpeg").read_text(), "1")
            all_frames = base / "all"
            result = self.run_script(source, "-o", all_frames, "-n", "10")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(list(all_frames.iterdir())), 3)

    def test_filename_newlines_and_dash_prefixed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "-input"
            source.mkdir()
            name = "line\nbreak.png"
            (source / name).write_bytes(b"image bytes")
            result = self.run_script("-n", "1", "--", "-input", cwd=base)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((base / "extracted_frames" / f"frame_000_{name}").read_bytes(), b"image bytes")


if __name__ == "__main__":
    unittest.main()
