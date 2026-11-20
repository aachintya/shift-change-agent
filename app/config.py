"""All settings in one place, read from environment variables.

Why env vars: the same image runs in docker, in tests and on a laptop; only the
environment changes. Secrets (LLM key, JWT secret) never live in the code.
"""
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/transport")

# JWT: HS256 with a shared secret is enough for a single service.
# The default is for local development only; docker-compose / .env overrides it.
JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-secret-change-me")
JWT_TTL_MINUTES = int(os.getenv("JWT_TTL_MINUTES", "720"))

# LLM: any OpenAI-compatible chat endpoint (Groq, Gemini, Ollama...).
# Tests and the offline eval inject a FakeLLM instead, so they need no key.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "3.0"))

# Hard ceiling of 4 LLM calls per message. Our normal path uses 1 call; a malformed answer
# gets 1 repair attempt, so a message never uses more than 2 (< 4).
MAX_LLM_ATTEMPTS = 2
# Total time for ALL LLM calls of one message (each call also has its own
# LLM_TIMEOUT_SECONDS). Bounds LLM time only; database work and checkpointing are
# extra (normally ~15 ms), so the 5 s p95 target is measured, not enforced.
LLM_BUDGET_SECONDS = 4.0

# Optional fixed clock, e.g. DEMO_TIME=17:00 -> "now" is today 17:00 IST.
# Makes the demo repeatable (the cut-off story always plays out the same way).
DEMO_TIME = os.getenv("DEMO_TIME", "")

# Shown to employees when the bot cannot help.
DESK_PHONE = os.getenv("DESK_PHONE", "080-4000-1234")
