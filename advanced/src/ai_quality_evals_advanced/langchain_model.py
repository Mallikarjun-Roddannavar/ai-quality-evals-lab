"""LangChain chat model backed exclusively by the bounded approved adapter."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field


class ApprovedChat(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    inference: Any = Field(exclude=True)

    @property
    def _llm_type(self):
        return "demoshop-approved-provider"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        roles = {"human": "user", "ai": "assistant", "system": "system"}
        converted = [{"role": roles[m.type], "content": m.content} for m in messages]
        reply = self.inference.generate(converted)
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=reply.text, response_metadata={"model": reply.model}
                    )
                )
            ]
        )
