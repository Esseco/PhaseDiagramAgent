"""Strict durable boundary contracts, separate from transient runtime objects."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApprovalIdentity(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    invocation_id: str = Field(min_length=1)
    config_version: str | None
    proposal_hash: str = Field(min_length=1)
    revision: int = Field(ge=0)

    @field_validator("invocation_id", "proposal_hash")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("identity must not be blank")
        return value


class ExecutionIdentity(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    invocation_id: str = Field(min_length=1)
    config_version: str = Field(min_length=1)
    action_hash: str = Field(min_length=1)
    tool: str = Field(min_length=1)

    @field_validator("invocation_id", "config_version", "action_hash", "tool")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("identity must not be blank")
        return value
