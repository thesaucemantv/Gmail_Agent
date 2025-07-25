from dataclasses import dataclass


@dataclass(kw_only=True)
class AgentConfigurable:
    """The configurable fields for the chatbot."""

    user_id: str = "default-user"