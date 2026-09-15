"""Ragas adapter for Groq's single-completion API and nested reasoning usage metadata."""
from ragas.llms.base import LangchainLLMWrapper
from langchain_core.outputs import LLMResult


class GroqRagasLLM(LangchainLLMWrapper):
    async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        # Groq requires n=1. Calling LangChain with multiple prompts also triggers a
        # dict += dict bug in this pinned adapter when GPT-OSS returns nested token usage.
        # Request each completion separately and combine only the generated text.
        generations = []
        for _ in range(n):
            result = await super().agenerate_text(prompt, n=1, temperature=temperature,
                                                  stop=stop, callbacks=callbacks)
            generations.append(result.generations[0][0])
        return LLMResult(generations=[generations])
