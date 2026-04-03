from dataclasses import dataclass, field


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    candidates_tokens: int = 0
    total_tokens: int = 0
    thoughts_tokens: int = 0

    @classmethod
    def from_response(cls, response) -> "TokenUsage":
        m = response.usage_metadata
        return cls(
            prompt_tokens=m.prompt_token_count or 0,
            candidates_tokens=m.candidates_token_count or 0,
            total_tokens=m.total_token_count or 0,
            thoughts_tokens=getattr(m, "thoughts_token_count", 0) or 0,
        )

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "candidates_tokens": self.candidates_tokens,
            "total_tokens": self.total_tokens,
            "thoughts_tokens": self.thoughts_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TokenUsage":
        return cls(**d)


@dataclass
class PolicyTokenLog:
    policy: str
    video_id: str
    build_usage: list[TokenUsage] = field(default_factory=list)
    query_usage: list[TokenUsage] = field(default_factory=list)

    @property
    def total_build_tokens(self) -> int:
        return sum(u.total_tokens for u in self.build_usage)

    @property
    def total_query_tokens(self) -> int:
        return sum(u.total_tokens for u in self.query_usage)

    @property
    def total_tokens(self) -> int:
        return self.total_build_tokens + self.total_query_tokens

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "video_id": self.video_id,
            "build_usage": [u.to_dict() for u in self.build_usage],
            "query_usage": [u.to_dict() for u in self.query_usage],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyTokenLog":
        return cls(
            policy=d["policy"],
            video_id=d["video_id"],
            build_usage=[TokenUsage.from_dict(u) for u in d["build_usage"]],
            query_usage=[TokenUsage.from_dict(u) for u in d["query_usage"]],
        )
