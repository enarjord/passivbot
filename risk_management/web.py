"""FastAPI powered web dashboard for Passivbot risk management."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from starlette.middleware.sessions import SessionMiddleware

from .configuration import RealtimeConfig
from .realtime import RealtimeDataFetcher
from .snapshot_utils import build_presentable_snapshot


class AuthManager:
    """Handle authentication for the dashboard."""

    def __init__(
        self,
        secret_key: str,
        users: Mapping[str, str],
        session_cookie_name: str = "risk_dashboard_session",
    ) -> None:
        if not secret_key:
            raise ValueError("Authentication requires a non-empty secret key.")
        if not users:
            raise ValueError("At least one dashboard user must be configured.")
        self.secret_key = secret_key
        self.users = dict(users)
        self.session_cookie_name = session_cookie_name
        self._password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def authenticate(self, username: str, password: str) -> bool:
        hashed = self.users.get(username)
        if not hashed:
            return False
        return self._password_context.verify(password, hashed)


class RiskDashboardService:
    """Wrap a realtime fetcher to expose snapshot data."""

    def __init__(self, fetcher: RealtimeDataFetcher) -> None:
        self._fetcher = fetcher

    async def fetch_snapshot(self) -> Dict[str, Any]:
        return await self._fetcher.fetch_snapshot()

    async def close(self) -> None:
        await self._fetcher.close()


def create_app(
    config: RealtimeConfig,
    *,
    service: RiskDashboardService | None = None,
    auth_manager: AuthManager | None = None,
    templates_dir: Path | None = None,
) -> FastAPI:
    if service is None:
        service = RiskDashboardService(RealtimeDataFetcher(config))
    if config.auth is None and auth_manager is None:
        raise ValueError("Realtime configuration must include authentication details for the web dashboard.")
    if auth_manager is None and config.auth is not None:
        auth_manager = AuthManager(
            config.auth.secret_key,
            config.auth.users,
            session_cookie_name=config.auth.session_cookie_name,
        )
    assert auth_manager is not None  # for mypy/static tools

    app = FastAPI(title="Passivbot Risk Dashboard")
    app.state.service = service
    app.state.auth_manager = auth_manager

    templates_path = templates_dir or Path(__file__).with_name("templates")
    templates = Jinja2Templates(directory=str(templates_path))

    def currency_filter(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "-"
        return f"${number:,.2f}"

    def pct_filter(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "0.00%"
        return f"{number * 100:.2f}%"

    templates.env.filters.setdefault("currency", currency_filter)
    templates.env.filters.setdefault("pct", pct_filter)

    app.add_middleware(
        SessionMiddleware,
        secret_key=auth_manager.secret_key,
        session_cookie=auth_manager.session_cookie_name,
    )

    def get_service(request: Request) -> RiskDashboardService:
        return request.app.state.service

    def require_user(request: Request) -> str:
        user = request.session.get("user")
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return str(user)

    @app.get("/login", response_class=HTMLResponse)
    async def login_form(request: Request) -> HTMLResponse:
        if request.session.get("user"):
            return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        return templates.TemplateResponse("login.html", {"request": request, "error": None})

    @app.post("/login", response_class=HTMLResponse)
    async def login_submit(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ) -> HTMLResponse:
        if not auth_manager.authenticate(username, password):
            context = {"request": request, "error": "Invalid username or password."}
            return templates.TemplateResponse("login.html", context, status_code=status.HTTP_401_UNAUTHORIZED)
        request.session["user"] = username
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    @app.post("/logout")
    async def logout(request: Request) -> RedirectResponse:
        request.session.pop("user", None)
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request, service: RiskDashboardService = Depends(get_service)) -> HTMLResponse:
        user = request.session.get("user")
        if not user:
            return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        snapshot = await service.fetch_snapshot()
        view_model = build_presentable_snapshot(snapshot)
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "user": user,
                "snapshot": view_model,
            },
        )

    @app.get("/api/snapshot", response_class=JSONResponse)
    async def api_snapshot(
        request: Request,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        snapshot = await service.fetch_snapshot()
        view_model = build_presentable_snapshot(snapshot)
        return JSONResponse(view_model)

    @app.on_event("shutdown")
    async def shutdown() -> None:  # pragma: no cover - FastAPI lifecycle
        await service.close()

    return app
