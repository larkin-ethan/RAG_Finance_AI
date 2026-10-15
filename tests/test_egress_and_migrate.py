import asyncio

from finrag.security.egress_proxy import EgressProxy, is_public, serve
from finrag.store.migrate import grants


async def _request(port: int, request: bytes) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(request)
    await writer.drain()
    data = await asyncio.wait_for(reader.read(1024), 5)
    writer.close()
    return data


def _run(coro):
    return asyncio.run(coro)


def test_is_public():
    assert is_public("93.184.216.34")
    for ip in ("10.0.0.1", "127.0.0.1", "169.254.169.254", "192.168.1.5", "::1"):
        assert not is_public(ip)


def test_proxy_denies_non_connect_wrong_port_and_unlisted_hosts():
    async def scenario():
        async def resolve(host, port):
            return ["93.184.216.34"]

        server = await serve(EgressProxy({"www.sec.gov"}, resolve=resolve), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server:
            get = await _request(port, b"GET http://www.sec.gov/ HTTP/1.1\r\nHost: x\r\n\r\n")
            other = await _request(port, b"CONNECT evil.example:443 HTTP/1.1\r\n\r\n")
            port80 = await _request(port, b"CONNECT www.sec.gov:80 HTTP/1.1\r\n\r\n")
            return get, other, port80

    get, other, port80 = _run(scenario())
    assert get.startswith(b"HTTP/1.1 405")
    assert other.startswith(b"HTTP/1.1 403")
    assert port80.startswith(b"HTTP/1.1 403")


def test_proxy_blocks_allowlisted_name_resolving_to_private_ip():
    async def scenario():
        async def rebind(host, port):
            return ["169.254.169.254"]  # cloud metadata / internal address

        server = await serve(EgressProxy({"www.sec.gov"}, resolve=rebind), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server:
            return await _request(port, b"CONNECT www.sec.gov:443 HTTP/1.1\r\n\r\n")

    assert _run(scenario()).startswith(b"HTTP/1.1 403")


def test_proxy_tunnels_to_allowed_host():
    async def scenario():
        async def echo(reader, writer):
            writer.write(await reader.read(100))
            await writer.drain()
            writer.close()

        upstream = await asyncio.start_server(echo, "127.0.0.1", 0)
        up_port = upstream.sockets[0].getsockname()[1]

        async def resolve(host, port):
            return ["127.0.0.1"]

        proxy = EgressProxy({"api.example.com"}, ports=frozenset({up_port}), resolve=resolve,
                            public_only=False)  # fmt: skip
        server = await serve(proxy, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with upstream, server:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(f"CONNECT API.example.com:{up_port} HTTP/1.1\r\n\r\n".encode())
            await writer.drain()
            status = await reader.readuntil(b"\r\n\r\n")
            writer.write(b"ping")
            await writer.drain()
            echoed = await asyncio.wait_for(reader.read(4), 5)
            writer.close()
            return status, echoed

    status, echoed = _run(scenario())
    assert status.startswith(b"HTTP/1.1 200") and echoed == b"ping"


def test_migrate_grants_are_dml_only():
    from psycopg import sql

    text = " ".join(
        s.as_string(None) if hasattr(s, "as_string") else str(s) for s in grants("finrag_app")
    )
    assert "SELECT, INSERT, UPDATE, DELETE" in text
    assert "REVOKE CREATE" in text
    assert "DROP" not in text and "ALTER" not in text
    assert isinstance(grants("x")[0], sql.Composed)


def test_auto_migrate_flag(monkeypatch):
    from finrag.store.db import auto_migrate_enabled

    assert auto_migrate_enabled()
    monkeypatch.setenv("DB_AUTO_MIGRATE", "false")
    assert not auto_migrate_enabled()
