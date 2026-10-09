"""Typed UI transport only; user messages never acquire tool authority."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ContentPart(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    type: str
    text: str | None = None

    @model_validator(mode="after")
    def require_text_for_text_part(self):
        if self.type in {"text", "input_text"} and self.text is None:
            raise ValueError("text part requires string text")
        return self


class ChatMessage(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    role: Literal["system", "developer", "user", "assistant", "tool", "function"]
    content: str | list[ContentPart] | None = None


class ChatMetadata(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    chat_id: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    messages: list[ChatMessage] = Field(min_length=1)
    metadata: ChatMetadata | None = None
    user: str | None = None
    model: str | None = None
    stream: bool = False


def validate_chat_request(payload):
    try:
        ChatRequest.model_validate(payload)
    except ValidationError as error:
        raise ValueError("Invalid chat request: " + "; ".join(
            ".".join(map(str, row["loc"])) + ": " + row["type"]
            for row in error.errors(include_input=False, include_url=False))) from None


class AssistantMessage(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    role: Literal["assistant"]
    content: str


class ChatChoice(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    index: int
    message: AssistantMessage
    finish_reason: Literal["stop"]


class ChatResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatChoice] = Field(min_length=1)
