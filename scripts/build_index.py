from __future__ import annotations

import argparse
from pathlib import Path

from src.config import SAMPLE_DOCS_DIR, load_config
from src.knowledge_base import KnowledgeBase


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 RAG 知识库的 FAISS + BM25 索引。")
    parser.add_argument("--doc-dir", type=Path, default=SAMPLE_DOCS_DIR, help="待入库文档目录。")
    args = parser.parse_args()

    kb = KnowledgeBase(load_config())
    stats = kb.build(doc_dir=args.doc_dir, reset=True)
    print("索引构建成功")
    print(f"文件数: {stats['files']}")
    print(f"原始文档数: {stats['documents']}")
    print(f"文本块数: {stats['chunks']}")
    print(f"来源文件: {stats['sources']}")


if __name__ == "__main__":
    main()
