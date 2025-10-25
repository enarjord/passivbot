"""FastAPI powered web dashboard for live risk management.

The application exposes REST endpoints and templated views backed by the
RealtimeDataFetcher utilities.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Mapping
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.sessions import SessionMiddleware
from urllib.parse import quote, urljoin

from .configuration import RealtimeConfig
from .realtime import RealtimeDataFetcher
from .reporting import ReportManager
from .snapshot_utils import build_presentable_snapshot


class AuthManager:
    """Handle authentication for the dashboard."""

    def __init__(
        self,
        secret_key: str,
        users: Mapping[str, str],
        session_cookie_name: str = "risk_dashboard_session",
        https_only: bool = True,
    ) -> None:
        if not secret_key:
            raise ValueError("Authentication requires a non-empty secret key.")
        if not users:
            raise ValueError("At least one dashboard user must be configured.")
        self.secret_key = secret_key
        self.users = dict(users)
        self.session_cookie_name = session_cookie_name
        self.https_only = https_only
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

    async def trigger_kill_switch(
        self, account_name: str | None = None, symbol: str | None = None
    ) -> Dict[str, Any]:
        return await self._fetcher.execute_kill_switch(account_name, symbol)


def create_app(
    config: RealtimeConfig,
    *,
    service: RiskDashboardService | None = None,
    auth_manager: AuthManager | None = None,
    templates_dir: Path | None = None,
    letsencrypt_challenge_dir: Path | None = None,
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
            https_only=config.auth.https_only,
        )
    assert auth_manager is not None  # for mypy/static tools

    app = FastAPI(title="Risk Management Dashboard")
    app.state.service = service
    app.state.auth_manager = auth_manager
    reports_dir = config.reports_dir
    if reports_dir is None:
        base_root = config.config_root or Path.cwd()
        reports_dir = base_root / "reports"
    app.state.report_manager = ReportManager(reports_dir)

    def resolve_grafana_context() -> dict[str, Any]:
        grafana_cfg = config.grafana
        if grafana_cfg is None:
            return {"dashboards": [], "theme": None}

        def resolve_url(raw_url: str) -> str:
            url = raw_url.strip()
            if grafana_cfg.base_url and not url.lower().startswith(("http://", "https://")):
                base = grafana_cfg.base_url.rstrip("/") + "/"
                return urljoin(base, url.lstrip("/"))
            return url

        dashboards: list[dict[str, Any]] = []
        for dashboard in grafana_cfg.dashboards:
            dashboards.append(
                {
                    "title": dashboard.title,
                    "url": resolve_url(dashboard.url),
                    "description": dashboard.description,
                    "height": dashboard.height or grafana_cfg.default_height,
                }
            )

        return {"dashboards": dashboards, "theme": grafana_cfg.theme}

    app.state.grafana_context = resolve_grafana_context()

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

    if auth_manager.https_only:
        app.add_middleware(HTTPSRedirectMiddleware)

    app.add_middleware(
        SessionMiddleware,
        secret_key=auth_manager.secret_key,
        session_cookie=auth_manager.session_cookie_name,
        https_only=auth_manager.https_only,
        same_site="lax",
    )

    if letsencrypt_challenge_dir is not None:
        challenge_dir = Path(letsencrypt_challenge_dir)
        challenge_dir.mkdir(parents=True, exist_ok=True)
        app.mount(
            "/.well-known/acme-challenge",
            StaticFiles(directory=str(challenge_dir), check_dir=False),
            name="acme-challenge",
        )

    def get_service(request: Request) -> RiskDashboardService:
        return request.app.state.service

    def require_user(request: Request) -> str:
        user = request.session.get("user")
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return str(user)

    def get_report_manager(request: Request) -> ReportManager:
        return request.app.state.report_manager

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
        grafana_context: dict[str, Any] = request.app.state.grafana_context
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "user": user,
                "snapshot": view_model,
                "grafana_dashboards": grafana_context.get("dashboards", []),
                "grafana_theme": grafana_context.get("theme"),
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

    @app.post("/api/kill-switch", response_class=JSONResponse)
    async def api_global_kill_switch(
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        result = await service.trigger_kill_switch()
        return JSONResponse(result)

    @app.post("/api/accounts/{account_name}/kill-switch", response_class=JSONResponse)
    async def api_kill_switch(
        request: Request,
        account_name: str,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        target = account_name.strip()
        symbol = request.query_params.get("symbol")
        if symbol:
            symbol = symbol.strip()
            if symbol.lower() == "all":
                symbol = None
        try:
            if not target or target.lower() == "all":
                result = await service.trigger_kill_switch(symbol=symbol)
            else:
                result = await service.trigger_kill_switch(target, symbol=symbol)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.post("/api/accounts/{account_name}/positions/{symbol}/kill-switch", response_class=JSONResponse)
    async def api_position_kill_switch(
        account_name: str,
        symbol: str,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        target_symbol = symbol.strip()
        if not target_symbol:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Symbol is required")
        target_account = account_name.strip()
        if not target_account or target_account.lower() == "all":
            target_account = None
        try:
            result = await service.trigger_kill_switch(target_account, symbol=target_symbol)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.get("/api/accounts/{account_name}/reports", response_class=JSONResponse)
    async def api_list_reports(
        account_name: str,
        manager: ReportManager = Depends(get_report_manager),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        reports = await manager.list_reports(account_name)
        items = []
        for report in reports:
            data = report.to_view()
            data["download_url"] = (
                f"/api/accounts/{quote(account_name, safe='')}/reports/{quote(report.report_id, safe='')}"
            )
            items.append(data)
        return JSONResponse({"account": account_name, "reports": items})

    @app.post("/api/accounts/{account_name}/reports", response_class=JSONResponse)
    async def api_generate_report(
        account_name: str,
        service: RiskDashboardService = Depends(get_service),
        manager: ReportManager = Depends(get_report_manager),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        snapshot = await service.fetch_snapshot()
        view_model = build_presentable_snapshot(snapshot)
        try:
            report = await manager.create_account_report(account_name, view_model)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        data = report.to_view()
        data["download_url"] = (
            f"/api/accounts/{quote(account_name, safe='')}/reports/{quote(report.report_id, safe='')}"
        )
        return JSONResponse(data)

    @app.get("/api/accounts/{account_name}/reports/{report_id}")
    async def api_download_report(
        account_name: str,
        report_id: str,
        manager: ReportManager = Depends(get_report_manager),
        _: str = Depends(require_user),
    ) -> FileResponse:
        path = await manager.get_report_path(account_name, report_id)
        if path is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
        return FileResponse(path, media_type="text/csv", filename=path.name)

    @app.on_event("shutdown")
    async def shutdown() -> None:  # pragma: no cover - FastAPI lifecycle
        await service.close()

    return app
