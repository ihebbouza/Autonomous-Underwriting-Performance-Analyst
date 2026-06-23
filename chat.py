"""
DataAssistant answers free-form questions about the current week's data. It is deliberately NOT owned
by AnalystAgent and not used by run.py -- answering open-ended questions has no equivalent in an
unattended cron job, so it doesn't belong inside the class the unattended path depends on. Only app.py
knows this class exists.
"""
from pathlib import Path

import config


class DataAssistant:
    def __init__(self, prompt_dir=None):
        self.prompt_dir = Path(prompt_dir or config.PROMPT_DIR)
        self.system_prompt = self._load_prompt("chat_system_prompt.txt")
        self.history = []

    def _load_prompt(self, filename):
        lines = (self.prompt_dir / filename).read_text().splitlines()
        body, started = [], False
        for line in lines:
            if not started and line.strip().startswith("#"):
                continue
            started = True
            body.append(line)
        return "\n".join(body)

    def ask(self, question, summary, api_key=None):
        self.history.append({"role": "user", "content": question})
        if not api_key:
            answer = "This needs an ANTHROPIC_API_KEY to be set -- the chat assistant only runs against a live model."
            self.history.append({"role": "assistant", "content": answer})
            return answer
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        context = (
            f"Portfolio KPIs: {summary['portfolio_kpis']}\n"
            f"Top concerns: {summary['top_concerns']}\n"
            f"Top opportunities: {summary['top_opportunities']}\n"
            f"All concerns: {summary['all_concerns']}\n"
        )
        response = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=400,
            system=self.system_prompt + "\n\nCurrent data:\n" + context,
            messages=self.history,
        )
        answer = response.content[0].text
        self.history.append({"role": "assistant", "content": answer})
        return answer
