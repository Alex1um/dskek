import httpx
from aiohttp_socks import ProxyConnector


def get_http_client(proxy: str | None = None) -> httpx.Client:
    return httpx.Client(proxy=proxy)

def get_async_http_client(proxy: str | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(proxy=proxy)

def get_aio_proxy_connector(proxy: str) -> ProxyConnector:
    return ProxyConnector.from_url(proxy)

def get_aio_proxy_connector_checked(proxy: str | None) -> ProxyConnector | None:
    if not proxy:
        return
    return ProxyConnector.from_url(proxy)
