"""Minimal allowlist egress proxy (HTTPS CONNECT only) for the private cluster.

Pods have no direct internet access (default-deny NetworkPolicy); outbound
HTTPS goes through this proxy, which only tunnels to allowlisted hostnames:

  * CONNECT only, port 443 only: no plain HTTP, no other ports.
  * Exact hostname match (case-insensitive) against EGRESS_ALLOWLIST.
  * After resolving, refuses private/loopback/link-local/reserved addresses, so
    an allowlisted name can't be pointed at internal services (DNS rebinding).
  * Logs host + decision only (TLS hides paths and payloads).

  EGRESS_ALLOWLIST=www.sec.gov,data.sec.gov,api.tiingo.com,api.stlouisfed.org \
  python -m finrag.security.egress_proxy --port 3128
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import logging
import os
import socket
from collections.abc import Awaitable, Callable

log = logging.getLogger("egress")
DEFAULT_ALLOWLIST = "www.sec.gov,data.sec.gov,api.tiingo.com,api.stlouisfed.org"
MAX_HEADER_BYTES = 8192
TIMEOUT_S = 15

Resolver = Callable[[str, int], Awaitable[list[str]]]


async def system_resolve(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


def is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    return addr.is_global and not addr.is_multicast


class EgressProxy:
    def __init__(
        self,
        allowlist: set[str],
        ports: frozenset[int] = frozenset({443}),
        resolve: Resolver = system_resolve,
        public_only: bool = True,
    ):
        self.allowlist = {h.strip().lower() for h in allowlist if h.strip()}
        self.ports, self.resolve, self.public_only = ports, resolve, public_only

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), TIMEOUT_S)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError):
            writer.close()
            return
        request_line = head.split(b"\r\n", 1)[0].decode("latin-1")
        parts = request_line.split()
        if len(parts) != 3 or parts[0] != "CONNECT":
            return await self._deny(writer, 405, "Method Not Allowed", request_line[:80])
        host, _, port_s = parts[1].rpartition(":")
        host = host.strip("[]").lower()
        if not port_s.isdigit() or int(port_s) not in self.ports or host not in self.allowlist:
            return await self._deny(writer, 403, "Forbidden", parts[1])
        port = int(port_s)
        try:
            ips = await asyncio.wait_for(self.resolve(host, port), TIMEOUT_S)
        except (OSError, TimeoutError):
            return await self._deny(writer, 502, "Bad Gateway", f"{host}: resolve failed")
        if not ips or (self.public_only and not all(is_public(ip) for ip in ips)):
            return await self._deny(writer, 403, "Forbidden", f"{host} -> non-public {ips}")
        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(ips[0], port), TIMEOUT_S
            )
        except (OSError, TimeoutError):
            return await self._deny(writer, 502, "Bad Gateway", f"{host}: connect failed")
        log.info("allow %s:%s", host, port)
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        await asyncio.gather(self._pipe(reader, up_writer), self._pipe(up_reader, writer))

    @staticmethod
    async def _pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
        try:
            while data := await asyncio.wait_for(src.read(65536), 300):
                dst.write(data)
                await dst.drain()
        except (OSError, TimeoutError):
            pass
        finally:
            dst.close()

    @staticmethod
    async def _deny(writer: asyncio.StreamWriter, code: int, reason: str, what: str) -> None:
        log.warning("deny %s (%s)", what, code)
        writer.write(f"HTTP/1.1 {code} {reason}\r\nContent-Length: 0\r\n\r\n".encode())
        await writer.drain()
        writer.close()


async def serve(proxy: EgressProxy, host: str, port: int) -> asyncio.Server:
    return await asyncio.start_server(proxy.handle, host, port, limit=MAX_HEADER_BYTES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3128)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    allow = set(os.environ.get("EGRESS_ALLOWLIST", DEFAULT_ALLOWLIST).split(","))
    proxy = EgressProxy(allow)

    async def run() -> None:
        server = await serve(proxy, args.host, args.port)
        log.info(
            "egress proxy on %s:%s, allowlist=%s", args.host, args.port, sorted(proxy.allowlist)
        )
        async with server:
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
