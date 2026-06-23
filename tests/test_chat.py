from chat import DataAssistant


def test_ask_without_api_key_returns_explanatory_message():
    assistant = DataAssistant()
    answer = assistant.ask("What's going on?", {}, api_key=None)
    assert "ANTHROPIC_API_KEY" in answer


def test_ask_without_api_key_still_appends_both_messages_to_history():
    # Regression test: an earlier version only appended user+assistant messages to history inside the
    # live-API branch, so the no-key path's "needs a key" reply was returned directly but never
    # recorded. That was invisible while the caller displayed the return value directly, but became a
    # real bug once the dashboard switched to rendering purely from history after a rerun -- the
    # message would silently vanish instead of appearing.
    assistant = DataAssistant()
    assistant.ask("What's going on?", {}, api_key=None)
    assert len(assistant.history) == 2
    assert assistant.history[0]["role"] == "user"
    assert assistant.history[0]["content"] == "What's going on?"
    assert assistant.history[1]["role"] == "assistant"
    assert "ANTHROPIC_API_KEY" in assistant.history[1]["content"]


def test_multiple_questions_without_a_key_each_append_their_own_pair():
    assistant = DataAssistant()
    assistant.ask("First question?", {}, api_key=None)
    assistant.ask("Second question?", {}, api_key=None)
    assert len(assistant.history) == 4
    assert assistant.history[0]["content"] == "First question?"
    assert assistant.history[2]["content"] == "Second question?"
