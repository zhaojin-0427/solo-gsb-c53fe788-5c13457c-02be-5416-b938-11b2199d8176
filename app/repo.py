from __future__ import annotations

from typing import Any

import asyncpg

from . import audit


def unique_constraint_name(exc: BaseException) -> str | None:
    """SQLSTATE 23505（unique_violation）冲突的**实际约束名**；否则返回 None。

    asyncpg 把 SQLSTATE 放在 ``sqlstate``（``pgcode`` 是 psycopg 风格属性，
    asyncpg 异常上并不存在——只读 ``pgcode`` 会把唯一冲突误判为未知错误
    原样上抛，既丢确定性 4xx，也漏记拒绝审计）；部分唯一索引冲突时
    ``constraint_name`` 即索引名。调用方必须同时核对 SQLSTATE 与约束名，
    只翻译已知的占位约束，其余一律原样上抛，绝不误判。
    """
    sqlstate = getattr(exc, "sqlstate", None) or getattr(exc, "pgcode", None)
    if sqlstate != "23505":
        return None
    return getattr(exc, "constraint_name", None)


async def audit_denied(
    pool: asyncpg.Pool,
    *,
    tenant_id: str | None,
    actor_type: str,
    actor_id: str | None,
    action: str,
    resource: str | None,
    error_code: str,
    basis: dict[str, Any],
) -> None:
    """Record a rejected action in its own transaction (never blocks the rollback)."""
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await audit.append_audit(
                    conn,
                    tenant_id=tenant_id,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    action=action,
                    resource=resource,
                    result="denied",
                    error_code=error_code,
                    basis=basis,
                )
    except Exception:  # auditing failure must not mask the original error
        pass


async def resolve_custodian_ids(conn: asyncpg.Connection, ids: list[str]) -> None:
    rows = await conn.fetch("SELECT id FROM custodians WHERE id = ANY($1::text[])", ids)
    found = {r["id"] for r in rows}
    missing = sorted(set(ids) - found)
    if missing or len(ids) != len(set(ids)):
        from .errors import ApiError, E_BAD_REQUEST

        raise ApiError(
            E_BAD_REQUEST,
            basis={"unknown_custodian_ids": missing, "duplicate": len(ids) != len(set(ids))},
        )


async def get_policy_version(
    conn: asyncpg.Connection, policy_id: str, version: int
) -> asyncpg.Record | None:
    return await conn.fetchrow("SELECT * FROM policies WHERE id = $1 AND version = $2", policy_id, version)


async def get_head_version(conn: asyncpg.Connection, policy_id: str) -> int | None:
    row = await conn.fetchrow("SELECT current_version FROM policy_heads WHERE policy_id = $1", policy_id)
    return row["current_version"] if row else None
