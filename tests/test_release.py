from __future__ import annotations

import unittest

from ura_xlaw.config import PATHS
from ura_xlaw.dataset.validate_release import validate_release


class ReleaseTests(unittest.TestCase):
    def test_release_integrity_and_grounding(self) -> None:
        validate_release(PATHS.release, sample_size=0)


if __name__ == "__main__":
    unittest.main()
