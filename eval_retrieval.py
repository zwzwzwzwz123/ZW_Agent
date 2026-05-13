from __future__ import annotations

from src.config import load_config
from src.evaluation import evaluate_retrieval
from src.knowledge_base import KnowledgeBase


def main() -> None:
    config = load_config()
    kb = KnowledgeBase(config)
    kb.load()
    metrics = evaluate_retrieval(kb)
    print(f"评估样例数: {metrics['cases']:.0f}")
    print(f"Recall@K: {metrics['recall_at_k']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")


if __name__ == "__main__":
    main()
