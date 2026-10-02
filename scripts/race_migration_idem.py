"""Regression harness: concurrent idempotent creation of migration sessions (v1.6.2).

Run (needs KEYVAULT_FAULT_INJECTION=1 for the gate-pinned scenarios; the docker
compose ``demo`` profile sets it automatically):

    python -m scripts.race_migration_idem

It boots the *real* FastAPI app in-process (ASGI transport, real PostgreSQL,
real unique indexes) and pins the exact ``POST /migrations`` interleaving that
previously surfaced as an unhandled database error: one tenant reuses a single
``Idempotency-Key`` to concurrently start migrations for **two different,
valid** source custody records. The two custody row locks do not overlap, so
both requests pass the idempotency pre-check (pinned at
``migration.after_idem_check``); the later placeholder insert then hits
``uq_migration_idem_open``. The old branch only read ``pgcode`` — an attribute
asyncpg exceptions do not have — so the ``UniqueViolationError`` was re-raised
raw: the caller got an unhandled 500 and the promised denial audit was never
recorded.

Expected behavior under test:

  I1  pinned pre-check interleaving (winner commits first): exactly one 201
      (queryable open session), the loser gets a stable 409
      IDEMPOTENCY_CONFLICT naming the winner; the loser leaves no session row
      and no success audit, and exactly one separately-committed denial audit
      matches the 409. The same key + the loser's request stays a
      deterministic 409, the same key + the winner's request replays the
      winner (200), and the loser immediately re-starts with a fresh key.
  I2  same key + same request fired concurrently: the custody row lock
      serializes them — 201 then a 200 replay of the same session; exactly
      one session row and one success audit.
  I3  both gates released simultaneously: the loser's insert waits on the
      unique index until the winner commits, then resolves to the same
      deterministic outcome — the result set is always {201, 409} and exactly
      one session row exists for the key.
  I4  different keys on the same source custody stay MIGRATION_CONFLICT
      (open_migration_exists_for_custody).
  I5  a unique violation on a NON-target constraint (injected at
      ``migration.insert``) surfaces as a plain 500 — never mistranslated
      into a 409 — leaves no session row and no idempotency denial audit,
      and the same key retries cleanly.
  I6  a generic database error at the insert point likewise surfaces as a
      500 (never a 409), rolls back wholesale, and the same key retries.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import sys

# The gates must be armed before app construction reads settings.
os.environ.setdefault("KEYVAULT_FAULT_INJECTION", "1")

import httpx

from app.config import get_settings
from app.crypto import sha256_hex
from app.db import Database, wait_for_database
from app.faults import FaultBoard
from app.main import app
from app.pending import PendingShareStore
from app.seed import seed, demo_credentials

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail and not ok else ""))


def body_of(r) -> dict:
    try:
        return r.json()
    except Exception:  # noqa: BLE001
        return {}


class _Raw500:
    """Stand-in for the 500 a real HTTP client gets on an unhandled exception.

    Under httpx's ASGI transport, Starlette's ServerErrorMiddleware sends the
    500 response and then re-raises the original exception (uvicorn itself
    swallows that re-raise); we surface it as a plain 500 here.
    """

    def __init__(self, exc: BaseException):
        self.status_code = 500
        self.text = f"{type(exc).__name__}: {exc}"

    def json(self) -> dict:
        return {}


EPOCH = {"valid_from": "2026-01-01T00:00:00Z",
         "valid_until": "2035-01-01T00:00:00Z", "rotation_interval": 0}


@contextlib.asynccontextmanager
async def server():
    settings = get_settings()
    await wait_for_database(settings.database_url)
    await seed()
    database = Database(settings.database_url)
    await database.connect()
    app.state.db = database
    app.state.pending = PendingShareStore()
    app.state.fault_enabled = True
    board = FaultBoard()
    app.state.fault_board = board
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        try:
            yield http, database.pool, board
        finally:
            await database.close()


def auth(key: str) -> dict:
    return {"X-API-Key": key}


async def wait_lock_waiters(pool, expect: int, timeout: float = 10.0) -> int:
    """Wait until ``expect`` backends wait on an ungranted custody row lock."""
    sql = """
        SELECT count(*)
        FROM pg_locks l
        JOIN pg_stat_activity a ON a.pid = l.pid
        WHERE l.locktype = 'transactionid'
          AND NOT l.granted
          AND a.query LIKE '%key_custodies%FOR UPDATE%'
    """
    deadline = asyncio.get_event_loop().time() + timeout
    seen = 0
    while asyncio.get_event_loop().time() < deadline:
        seen = await pool.fetchval(sql)
        if seen >= expect:
            return seen
        await asyncio.sleep(0.05)
    return seen


class Harness:
    def __init__(self, http, pool, creds):
        self.http = http
        self.pool = pool
        self.tenant_key = creds["tenant"]["api_key"]
        self.secret = b"migration-idem-race-regression-0123456789"

    async def make_policies(self, tag: str):
        """Fresh 2-of-3 source policy and a distinct target policy."""
        src = f"pol_isrc_{tag}"
        tgt = f"pol_itgt_{tag}"
        for pid in (src, tgt):
            r = await self.http.post("/policies", headers=auth(self.tenant_key), json={
                "policy_id": pid, "threshold": 2,
                "custodian_ids": ["cus_demo_alice", "cus_demo_bob", "cus_demo_carol"],
                "epoch": EPOCH,
            })
            assert r.status_code == 201, r.text
        return src, tgt

    async def make_custody(self, policy: str) -> str:
        r = await self.http.post("/custodies", headers=auth(self.tenant_key), json={
            "policy_id": policy, "expected_version": 1, "secret_hex": self.secret.hex(),
        })
        assert r.status_code == 201, r.text
        return r.json()["custody_id"]

    async def start(self, custody_id: str, target: str, idem: str,
                    extra_headers: dict | None = None):
        try:
            return await self.http.post(
                "/migrations",
                headers={**auth(self.tenant_key), "Idempotency-Key": idem,
                         **(extra_headers or {})},
                json={"source_custody_id": custody_id, "target_policy_id": target,
                      "target_version": 1},
            )
        except Exception as exc:  # ASGI transport re-raises the 500 path exception
            return _Raw500(exc)

    async def key_rows(self, idem: str) -> int:
        return await self.pool.fetchval(
            "SELECT count(*) FROM migration_sessions WHERE idempotency_key_hash=$1",
            sha256_hex(idem.encode()))

    async def custody_rows(self, custody_id: str) -> int:
        return await self.pool.fetchval(
            "SELECT count(*) FROM migration_sessions WHERE source_custody_id=$1",
            custody_id)

    async def start_audits(self, custody_id: str, result: str) -> list:
        return await self.pool.fetch(
            "SELECT error_code, basis FROM audit_log "
            "WHERE action='migration.start' AND result=$1 "
            "AND (resource=$2 OR basis->>'source_custody_id'=$2)",
            result, custody_id)


async def scenario_i1(hs: Harness, board: FaultBoard, tag: str):
    """Same key x two different valid sources, both passing the pre-check."""
    label = "I1(idem-key-race:conflict)"
    src, tgt = await hs.make_policies(tag)
    cid_a = await hs.make_custody(src)
    cid_b = await hs.make_custody(src)
    key = f"i1-{tag}"
    gate_a = {"X-Fault-Key": f"{label}-a",
              "X-Fault-Gate": "migration.after_idem_check"}
    gate_b = {"X-Fault-Key": f"{label}-b",
              "X-Fault-Gate": "migration.after_idem_check"}

    ta = asyncio.create_task(hs.start(cid_a, tgt, key, gate_a))
    tb = asyncio.create_task(hs.start(cid_b, tgt, key, gate_b))
    # 钉住交错：两个请求都已通过幂等预查（都未发现既有会话）。
    await board.wait_reached(f"migration.after_idem_check:{label}-a")
    await board.wait_reached(f"migration.after_idem_check:{label}-b")

    # 先放行 A：完整提交 -> 201 + 可查询的 open 会话。
    await board.release_gate(f"migration.after_idem_check:{label}-a")
    ra = await ta
    ma = body_of(ra)
    check(f"{label}: first request -> 201 open session",
          ra.status_code == 201 and ma.get("status") == "open"
          and ma.get("source_custody_id") == cid_a,
          f"{ra.status_code} {ra.text[:200]}")
    mid_a = ma.get("migration_id")

    # 再放行 B：插入撞 uq_migration_idem_open -> 整体回滚 -> 确定性 409。
    await board.release_gate(f"migration.after_idem_check:{label}-b")
    rb = await tb
    eb = body_of(rb).get("error", {})
    check(f"{label}: loser -> stable 409 IDEMPOTENCY_CONFLICT (not 500)",
          rb.status_code == 409 and eb.get("code") == "IDEMPOTENCY_CONFLICT"
          and eb.get("basis", {}).get("reason")
          == "concurrent_open_with_same_idempotency_key"
          and eb.get("basis", {}).get("migration_id") == mid_a,
          f"{rb.status_code} {rb.text[:200]}")

    # 胜方会话可查询。
    r = await hs.http.get(f"/migrations/{mid_a}", headers=auth(hs.tenant_key))
    g = body_of(r)
    check(f"{label}: winner session queryable (open, bound to its custody)",
          r.status_code == 200 and g.get("status") == "open"
          and g.get("source_custody_id") == cid_a, r.text[:200])

    # 败方不留迁移会话：该幂等键全库恰一行（胜方的），败方 custody 无任何会话。
    check(f"{label}: loser left no session row",
          await hs.key_rows(key) == 1 and await hs.custody_rows(cid_b) == 0,
          f"key_rows={await hs.key_rows(key)} loser_rows={await hs.custody_rows(cid_b)}")

    # 败方不留成功审计；恰一条与实际结果一致的拒绝审计（独立事务提交）。
    ok_audits = await hs.start_audits(cid_b, "ok")
    ok_winner = await hs.start_audits(cid_a, "ok")
    check(f"{label}: no success audit for the loser, winner audited exactly once",
          len(ok_audits) == 0 and len(ok_winner) == 1,
          f"loser_ok={len(ok_audits)} winner_ok={len(ok_winner)}")
    denied = await hs.start_audits(cid_b, "denied")
    check(f"{label}: exactly one denial audit matching the 409",
          len(denied) == 1 and denied[0]["error_code"] == "IDEMPOTENCY_CONFLICT",
          str([dict(d) for d in denied]))

    # 同键再发败方请求 -> 预查路径的确定性 409；同键同胜方请求 -> 200 重放。
    r = await hs.start(cid_b, tgt, key)
    check(f"{label}: same key + loser request again -> 409 (pre-check path)",
          r.status_code == 409
          and body_of(r).get("error", {}).get("code") == "IDEMPOTENCY_CONFLICT",
          f"{r.status_code} {r.text[:150]}")
    r = await hs.start(cid_a, tgt, key)
    check(f"{label}: same key + winner request -> 200 replay of winner session",
          r.status_code == 200 and body_of(r).get("migration_id") == mid_a,
          f"{r.status_code} {r.text[:150]}")

    # 败方换新幂等键即可安全重试（败方不占位）。
    r = await hs.start(cid_b, tgt, f"i1-retry-{tag}")
    check(f"{label}: loser retries with a fresh key -> 201",
          r.status_code == 201 and body_of(r).get("source_custody_id") == cid_b,
          f"{r.status_code} {r.text[:150]}")


async def scenario_i2(hs: Harness, board: FaultBoard, tag: str):
    """Same key + same request concurrently: serialized replay of one session."""
    label = "I2(same-request-concurrent-retry)"
    src, tgt = await hs.make_policies(tag)
    cid = await hs.make_custody(src)
    key = f"i2-{tag}"
    gate = {"X-Fault-Key": label, "X-Fault-Gate": "migration.after_idem_check"}

    t1 = asyncio.create_task(hs.start(cid, tgt, key, gate))
    await board.wait_reached(f"migration.after_idem_check:{label}")
    # 同键同请求的第二个请求在源 custody 行锁上排队（锁不重叠才需要唯一索引
    # 兜底；同源的并发由行锁串行化后经预查重放）。
    t2 = asyncio.create_task(hs.start(cid, tgt, key))
    queued = await wait_lock_waiters(hs.pool, 1)
    check(f"{label}: same-request retry queues on the custody row lock",
          queued == 1, f"q={queued}")
    await board.release_gate(f"migration.after_idem_check:{label}")
    r1 = await t1
    r2 = await t2
    check(f"{label}: 201 then 200 replay of the same session",
          r1.status_code == 201 and r2.status_code == 200
          and body_of(r1).get("migration_id")
          and body_of(r2).get("migration_id") == body_of(r1).get("migration_id"),
          f"{r1.status_code}/{r2.status_code} {r2.text[:150]}")
    check(f"{label}: exactly one session row and one success audit",
          await hs.key_rows(key) == 1 and len(await hs.start_audits(cid, "ok")) == 1,
          f"rows={await hs.key_rows(key)} ok={len(await hs.start_audits(cid, 'ok'))}")


async def scenario_i3(hs: Harness, board: FaultBoard, tag: str):
    """Both gates released together: the index wait resolves to {201, 409}."""
    label = "I3(simultaneous-release)"
    src, tgt = await hs.make_policies(tag)
    cid_a = await hs.make_custody(src)
    cid_b = await hs.make_custody(src)
    key = f"i3-{tag}"
    gate_a = {"X-Fault-Key": f"{label}-a",
              "X-Fault-Gate": "migration.after_idem_check"}
    gate_b = {"X-Fault-Key": f"{label}-b",
              "X-Fault-Gate": "migration.after_idem_check"}
    ta = asyncio.create_task(hs.start(cid_a, tgt, key, gate_a))
    tb = asyncio.create_task(hs.start(cid_b, tgt, key, gate_b))
    await board.wait_reached(f"migration.after_idem_check:{label}-a")
    await board.wait_reached(f"migration.after_idem_check:{label}-b")
    # 同时放行：败方的插入在唯一索引上等待胜方提交，解析后确定性 409。
    await board.release_gate(f"migration.after_idem_check:{label}-a")
    await board.release_gate(f"migration.after_idem_check:{label}-b")
    ra, rb = await asyncio.gather(ta, tb)
    codes = sorted(r.status_code for r in (ra, rb))
    check(f"{label}: exactly one 201 and one 409", codes == [201, 409], str(codes))
    if codes == [201, 409]:
        win, lose = (ra, rb) if ra.status_code == 201 else (rb, ra)
        check(f"{label}: 409 carries IDEMPOTENCY_CONFLICT",
              body_of(lose).get("error", {}).get("code") == "IDEMPOTENCY_CONFLICT",
              lose.text[:150])
        mid = body_of(win).get("migration_id")
        r = await hs.http.get(f"/migrations/{mid}", headers=auth(hs.tenant_key))
        check(f"{label}: winner session queryable",
              r.status_code == 200 and body_of(r).get("status") == "open",
              r.text[:150])
        check(f"{label}: still exactly one session row for the key",
              await hs.key_rows(key) == 1, str(await hs.key_rows(key)))
        loser_cid = cid_b if ra.status_code == 201 else cid_a
        check(f"{label}: loser custody left untouched",
              await hs.custody_rows(loser_cid) == 0,
              str(await hs.custody_rows(loser_cid)))


async def scenario_i4(hs: Harness, tag: str):
    """Different keys on the same source custody stay MIGRATION_CONFLICT."""
    label = "I4(different-key-same-source)"
    src, tgt = await hs.make_policies(tag)
    cid = await hs.make_custody(src)
    r1 = await hs.start(cid, tgt, f"i4a-{tag}")
    check(f"{label}: first key -> 201", r1.status_code == 201, r1.text[:150])
    r2 = await hs.start(cid, tgt, f"i4b-{tag}")
    err = body_of(r2).get("error", {})
    check(f"{label}: different key same source -> 409 MIGRATION_CONFLICT",
          r2.status_code == 409 and err.get("code") == "MIGRATION_CONFLICT"
          and err.get("basis", {}).get("reason") == "open_migration_exists_for_custody",
          f"{r2.status_code} {r2.text[:200]}")


async def scenario_i5(hs: Harness, tag: str):
    """A unique violation on a NON-target constraint is never mistranslated."""
    label = "I5(non-target-constraint)"
    src, tgt = await hs.make_policies(tag)
    cid = await hs.make_custody(src)
    key = f"i5-{tag}"
    r = await hs.start(cid, tgt, key, {
        "X-Fault-Key": label,
        "X-Fault-Fail": "migration.insert:PG23505:uq_migration_sessions_pkey"})
    check(f"{label}: non-target unique violation surfaces as 500, not 409",
          r.status_code == 500, f"{r.status_code} {r.text[:150]}")
    check(f"{label}: no session row after non-target violation",
          await hs.key_rows(key) == 0, str(await hs.key_rows(key)))
    denied = await hs.start_audits(cid, "denied")
    check(f"{label}: no idempotency denial audit misrecorded",
          len(denied) == 0, str([dict(d) for d in denied]))
    r = await hs.start(cid, tgt, key)
    check(f"{label}: same key retries cleanly -> 201",
          r.status_code == 201 and body_of(r).get("status") == "open",
          f"{r.status_code} {r.text[:150]}")


async def scenario_i6(hs: Harness, tag: str):
    """A generic database error at the insert point is never mistranslated."""
    label = "I6(generic-error-not-conflict)"
    src, tgt = await hs.make_policies(tag)
    cid = await hs.make_custody(src)
    key = f"i6-{tag}"
    r = await hs.start(cid, tgt, key, {
        "X-Fault-Key": label, "X-Fault-Fail": "migration.insert:RAISE"})
    check(f"{label}: generic failure surfaces as 500, not 409",
          r.status_code == 500, f"{r.status_code} {r.text[:150]}")
    check(f"{label}: no session row after generic failure",
          await hs.key_rows(key) == 0, str(await hs.key_rows(key)))
    r = await hs.start(cid, tgt, key)
    check(f"{label}: same key retries cleanly -> 201",
          r.status_code == 201 and body_of(r).get("status") == "open",
          f"{r.status_code} {r.text[:150]}")


async def amain() -> int:
    async with server() as (http, pool, board):
        creds = demo_credentials()
        hs = Harness(http, pool, creds)
        suffix = os.urandom(4).hex()
        await scenario_i1(hs, board, f"{suffix}1")
        await scenario_i2(hs, board, f"{suffix}2")
        await scenario_i3(hs, board, f"{suffix}3")
        await scenario_i4(hs, f"{suffix}4")
        await scenario_i5(hs, f"{suffix}5")
        await scenario_i6(hs, f"{suffix}6")

        r = await http.get("/audit/replay", headers=auth(hs.tenant_key))
        check("final: audit hash chain valid across idempotency-race scenarios",
              r.json().get("valid") is True)

    failed = [n for n, ok, _ in results if not ok]
    print("\n" + "=" * 70)
    print(f"{len(results) - len(failed)}/{len(results)} migration-idempotency-race checks passed")
    if failed:
        print("FAILED:")
        for n in failed:
            print(" -", n)
        return 1
    print("MIGRATION IDEMPOTENCY RACE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
