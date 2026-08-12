from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
RAG2 = HERE.parent
if str(RAG2) not in sys.path:
    sys.path.insert(0, str(RAG2))

from hybrid_index import DEFAULT_MODEL, HybridIndex, Rag2Error, WordPieceTokenizer


class Rag2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = RAG2 / "pack"
        cls.dense_available = all(
            (DEFAULT_MODEL / name).is_file()
            for name in (
                "bge-small-zh-v1.5.dynamic-uint8.onnx",
                "tokenizer.json",
            )
        )
        cls.index = HybridIndex(
            pack_dir=cls.pack,
            dense_model_dir=DEFAULT_MODEL,
            enable_dense=cls.dense_available,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.index.close()

    def test_pack_thresholds_and_zero_authority(self) -> None:
        manifest = json.loads((self.pack / "manifest.v2.json").read_text("utf-8"))
        self.assertGreaterEqual(manifest["counts"]["chunks"], 40)
        self.assertGreaterEqual(manifest["counts"]["gold"], 60)
        self.assertGreaterEqual(manifest["counts"]["forbidden"], 30)
        self.assertFalse(any(manifest["authority"].values()))

    def test_pure_python_tokenizer_contract(self) -> None:
        if not self.dense_available:
            self.skipTest("optional upstream BGE tokenizer is not redistributed")
        tokenizer = WordPieceTokenizer(DEFAULT_MODEL / "tokenizer.json")
        encoded = tokenizer.encode_batch(["为这个句子生成表示"])
        # [CLS], 为, 这, 个, 句, 子, 生, 成, 表, 示, [SEP]
        self.assertEqual(encoded["input_ids"].shape, (1, 11))
        self.assertEqual(encoded["input_ids"][0, 0], 101)
        self.assertEqual(encoded["input_ids"][0, -1], 102)
        self.assertEqual(int(encoded["attention_mask"].sum()), 11)

    def test_default_is_bm25_and_allowlisted(self) -> None:
        hits = self.index.search("固定舱为什么不需要定位建图", limit=5)
        self.assertTrue(hits)
        self.assertTrue(all(hit.backend == "SQLITE_FTS5_BM25" for hit in hits))
        self.assertTrue(all(hit.citation_id in self.index.allowlist for hit in hits))

    def test_dense_challenger_executes(self) -> None:
        if not self.dense_available:
            self.skipTest("optional upstream BGE ONNX model is not redistributed")
        hits = self.index.search(
            "出水后怎样确认目标根区真的变湿", limit=5, backend="dense"
        )
        self.assertEqual(len(hits), 5)
        self.assertTrue(all(hit.backend == "BGE_SMALL_ZH_V1_5_ONNX_UINT8" for hit in hits))

    def test_rrf_executes_without_new_citations(self) -> None:
        if not self.dense_available:
            self.skipTest("optional upstream BGE ONNX model is not redistributed")
        hits = self.index.search(
            "解释模型能否直接控制水泵", limit=5, backend="rrf"
        )
        self.assertEqual(len(hits), 5)
        self.assertTrue(all(hit.citation_id in self.index.allowlist for hit in hits))

    def test_invalid_queries_fail_closed(self) -> None:
        for query in ("", "x" * 257):
            with self.assertRaises(Rag2Error):
                self.index.search(query)

    def test_frozen_dense_selection_is_not_eligible(self) -> None:
        selection = json.loads(
            (RAG2 / "deploy_selection.v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(selection["selected_default"], "SQLITE_FTS5_BM25_V2")
        self.assertFalse(selection["dense_challenger"]["eligible"])
        self.assertEqual(
            selection["x5_status"],
            "FINAL_CANDIDATE_ACCEPTANCE_PENDING",
        )


if __name__ == "__main__":
    unittest.main()
