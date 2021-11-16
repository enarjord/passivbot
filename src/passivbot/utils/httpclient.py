import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from typing import Any
from typing import Dict
from typing import Optional

import aiohttp
import websockets

from passivbot.utils.funcs.pure import format_float
from passivbot.utils.funcs.pure import sort_dict_keys

log = logging.getLogger(__name__)


class HTTPRequestError(Exception):
    def __init__(self, url, code, msg=None):
        self.url = url
        self.code = code
        self.msg = msg

    def __str__(self) -> str:
        """
        Convert the exception into a printable string
        """
        return f"Request to {self.url!r} failed. Code: {self.code}; Message: {self.msg}"


class HTTPClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_secret: str,
        endpoints: Dict[str, str],
        session_headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        for name, url in endpoints.items():
            if url.startswith(("wss://", "http://", "https://")):
                continue
            if not url.startswith("/"):
                raise ValueError(f"The endpoint({name}) URL({url}) does not start with '/'")
        self.endpoints = endpoints
        self.session_headers = session_headers
        self.session = aiohttp.ClientSession(headers=session_headers)

    @classmethod
    async def onetime_get(cls, url):
        async with aiohttp.ClientSession().get(url) as response:
            result = await response.text()
        return json.loads(result)

    async def close(self):
        await self.session.close()

    def url_for_endpoint(self, endpoint: str) -> str:
        if endpoint.startswith(("wss://", "http://", "https://")):
            return endpoint
        if endpoint in self.endpoints:
            url = self.endpoints[endpoint]
        else:
            url = endpoint
        if not url.startswith(("wss://", "http://", "https://")):
            url = f"{self.base_url}{url}"
        return url

    def signature_params_key(self) -> str:
        raise NotImplementedError

    def get_signature(self, params: Dict[str, Any]) -> str:
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            urllib.parse.urlencode(params).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def signed_params(self, params: Dict[str, Any]) -> Dict[str, str]:
        params["timestamp"] = f"{int(time.time() * 1000)}"
        for key, value in params.items():
            if isinstance(value, bool):
                params[key] = str(value).lower()
            if isinstance(value, float):
                params[key] = format_float(value)
        params = sort_dict_keys(params)
        params[self.signature_params_key()] = self.get_signature(params)
        return params

    async def get(
        self,
        endpoint: str,
        signed: bool = False,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = self.url_for_endpoint(endpoint)
        if params is None:
            params = {}
        if signed:
            params = self.signed_params(params)
        log.debug("HTTPRequest URL: %s; HEADERS: %s; PARAMS: %s;", url, headers, params)
        async with self.session.get(url, params=params, headers=headers) as response:
            result = await response.text()
        payload: Dict[str, Any] = json.loads(result)
        error = self._get_error_from_payload(url, payload)
        if error:
            raise error
        return payload

    async def post(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = self.url_for_endpoint(endpoint)
        if params is None:
            params = {}
        params = self.signed_params(params)
        async with self.session.post(
            self.url_for_endpoint(endpoint), params=params, headers=headers
        ) as response:
            result = await response.text()
        payload: Dict[str, Any] = json.loads(result)
        error = self._get_error_from_payload(url, payload)
        if error:
            raise error
        return payload

    async def put(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = self.url_for_endpoint(endpoint)
        if params is None:
            params = {}
        params = self.signed_params(params)
        async with self.session.put(
            self.url_for_endpoint(endpoint), params=params, headers=headers
        ) as response:
            result = await response.text()
        payload: Dict[str, Any] = json.loads(result)
        error = self._get_error_from_payload(url, payload)
        if error:
            raise error
        return payload

    async def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = self.url_for_endpoint(endpoint)
        if params is None:
            params = {}
        params = self.signed_params(params)
        async with self.session.delete(
            self.url_for_endpoint(endpoint), params=params, headers=headers
        ) as response:
            result = await response.text()
        payload: Dict[str, Any] = json.loads(result)
        error = self._get_error_from_payload(url, payload)
        if error:
            raise error
        return payload

    def ws_connect(self, endpoint):
        return websockets.connect(self.url_for_endpoint(endpoint))

    def _get_error_from_payload(self, url: str, payload: Dict[str, Any]):
        raise NotImplementedError


class BinanceHTTPClient(HTTPClient):
    def signature_params_key(self) -> str:
        return "signature"

    def signed_params(self, params: Dict[str, Any]) -> Dict[str, str]:
        params["recvWindow"] = 5000
        return super().signed_params(params)

    async def get(
        self,
        endpoint: str,
        signed: bool = False,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if signed is True:
            if headers is None:
                headers = {}
            headers["X-MBX-APIKEY"] = self.api_key
        return await super().get(endpoint, signed, params, headers)

    async def post(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if headers is None:
            headers = {}
        headers["X-MBX-APIKEY"] = self.api_key
        return await super().post(endpoint, params, headers)

    async def put(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if headers is None:
            headers = {}
        headers["X-MBX-APIKEY"] = self.api_key
        return await super().put(endpoint, params, headers)

    async def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if headers is None:
            headers = {}
        headers["X-MBX-APIKEY"] = self.api_key
        return await super().delete(endpoint, params, headers)

    def _get_error_from_payload(
        self,
        url: str,
        payload: Dict[str, Any],
    ) -> Optional[HTTPRequestError]:
        if "code" in payload and "msg" in payload:
            return HTTPRequestError(url, code=payload["code"], msg=payload["msg"])
        return None


class ByBitHTTPClient(HTTPClient):
    def signature_params_key(self) -> str:
        return "sign"

    def signed_params(self, params: Dict[str, Any]) -> Dict[str, str]:
        params["api_key"] = self.api_key
        return super().signed_params(params)

    def _get_error_from_payload(
        self,
        url: str,
        payload: Dict[str, Any],
    ) -> Optional[HTTPRequestError]:
        return None
