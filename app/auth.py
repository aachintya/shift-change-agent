"""JWT issue and verify.

The ONLY way the system learns who is talking is the signed token. The chat
message, the request body and the LLM output are never asked "who are you".
Every function below that touches data receives a Principal built from here.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Header, HTTPException

from app import config


@dataclass(frozen=True)
class Principal:
    employee_id: str
    company_id: str
    role: str  # "employee" | "manager"


def issue_token(p: Principal) -> str:
    """Mock identity provider: signs a token for an employee that exists in the DB."""
    claims = {
        "sub": p.employee_id,
        "company_id": p.company_id,
        "role": p.role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=config.JWT_TTL_MINUTES),
    }
    return jwt.encode(claims, config.JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> Principal:
    # algorithms=["HS256"] pins the algorithm, so a token with alg="none" is rejected.
    claims = jwt.decode(token, config.JWT_SECRET, algorithms=["HS256"],
                        options={"require": ["sub", "company_id", "role", "exp"]})
    return Principal(claims["sub"], claims["company_id"], claims["role"])


def _from_header(authorization: str | None) -> Principal:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Please sign in again (missing token).")
    try:
        return decode_token(authorization.removeprefix("Bearer "))
    except jwt.PyJWTError:
        raise HTTPException(401, "Please sign in again (invalid or expired token).")


# FastAPI dependencies --------------------------------------------------------

def employee(authorization: str | None = Header(None)) -> Principal:
    p = _from_header(authorization)
    if p.role != "employee":
        raise HTTPException(403, "This chat is for employees.")
    return p


def manager(authorization: str | None = Header(None)) -> Principal:
    p = _from_header(authorization)
    if p.role != "manager":
        raise HTTPException(403, "Only a transport manager can do this.")
    return p
