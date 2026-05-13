from pathlib import Path


def test_default_reranker_is_bge_cross_encoder() -> None:
    env_text = Path(".env.example").read_text(encoding="utf-8")
    config_text = Path("src/config.py").read_text(encoding="utf-8")

    assert "RERANKER_MODE=cross_encoder" in env_text
    assert "RERANKER_MODEL=BAAI/bge-reranker-base" in env_text
    assert 'reranker_mode: str = "cross_encoder"' in config_text
    assert 'os.getenv("RERANKER_MODE", "cross_encoder")' in config_text


def test_multi_turn_history_reaches_retrieval_and_generation() -> None:
    app_text = Path("app.py").read_text(encoding="utf-8")
    agent_text = Path("src/agent.py").read_text(encoding="utf-8")
    generator_text = Path("src/generator.py").read_text(encoding="utf-8")

    assert "st.chat_input" in app_text
    assert "st.session_state.messages" in app_text
    assert "chat_history=history" in app_text
    assert "standalone_question = self._condense_question(question, chat_history)" in agent_text
    assert "self.knowledge_base.retrieve(standalone_question)" in agent_text
    assert '"history": self._format_history(chat_history)' in generator_text
