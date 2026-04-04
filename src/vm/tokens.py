from dataclasses import dataclass, field


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    candidates_tokens: int = 0
    total_tokens: int = 0
    thoughts_tokens: int = 0
    # Wall time for the generate_content await (0 when served from cache).
    latency_s: float = 0.0

    @classmethod
    def from_response(cls, response, *, latency_s: float = 0.0) -> "TokenUsage":
        m = response.usage_metadata
        return cls(
            prompt_tokens=m.prompt_token_count or 0,
            candidates_tokens=m.candidates_token_count or 0,
            total_tokens=m.total_token_count or 0,
            thoughts_tokens=getattr(m, "thoughts_token_count", 0) or 0,
            latency_s=latency_s,
        )

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "candidates_tokens": self.candidates_tokens,
            "total_tokens": self.total_tokens,
            "thoughts_tokens": self.thoughts_tokens,
            "latency_s": self.latency_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TokenUsage":
        return cls(
            prompt_tokens=d.get("prompt_tokens", 0),
            candidates_tokens=d.get("candidates_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
            thoughts_tokens=d.get("thoughts_tokens", 0),
            latency_s=float(d.get("latency_s", 0.0)),
        )


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
