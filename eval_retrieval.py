from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import load_config
    from src.evaluation import evaluate_retrieval
    from src.knowledge_base import KnowledgeBase
except ModuleNotFoundError as exc:
    missing = exc.name or str(exc)
    raise SystemExit(
        f"缺少依赖模块：{missing}\n"
        "请确认已激活正确的 conda 环境，并在项目根目录运行：pip install -r requirements.txt"
    ) from exc


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
