from __future__ import annotations

import unittest

from ura_xlaw.corpus.map_citations import parse_citation
from ura_xlaw.preprocessing.clean_judgments import clean_body_text


class CoreTests(unittest.TestCase):
    def test_parse_article_citation(self) -> None:
        citation = parse_citation("D188:K3:Luật Đất đai 2013")
        self.assertEqual(citation["article"], "188")
        self.assertEqual(citation["clause"], "3")

    def test_clean_judgment_text(self) -> None:
        self.assertEqual(clean_body_text(" A   B\n\n\n C "), "A B\n\nC")


if __name__ == "__main__":
    unittest.main()
