"""Dash web application for interactive bearing analysis.

Main entry point for running the bearing analysis UI server.
"""

import importlib.util
import os
import pkgutil
from importlib import import_module
from pathlib import Path

celery_app = None


def _require_ui_dependencies() -> None:
    missing = []
    for pkg in ("dash", "plotly"):
        try:
            import_module(pkg)
        except ModuleNotFoundError:
            missing.append(pkg)

    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "UI dependencies are missing: "
            f"{missing_str}. Install with:\n"
            "  uv sync --extra ui\n"
            'or:\n  pip install "openairbearing[ui]"'
        )


def build_background_callback_manager():
    """Build an optional Dash background callback manager from env config.

    Environment variables:
    - OAB_DASH_BACKGROUND_BACKEND: one of ``none``, ``celery``, ``diskcache``.
    - OAB_CELERY_BROKER_URL: Celery broker URL.
    - OAB_CELERY_RESULT_BACKEND: Celery result backend URL.
    - OAB_DISKCACHE_DIR: DiskCache directory path.
    """
    backend = os.getenv("OAB_DASH_BACKGROUND_BACKEND", "auto").strip().lower()
    if backend in {"", "none", "off", "disabled"}:
        return None

    # Auto mode: prefer Celery when configured, otherwise use DiskCache locally.
    if backend == "auto":
        has_celery_env = bool(
            os.getenv("OAB_CELERY_BROKER_URL", "").strip()
            and os.getenv("OAB_CELERY_RESULT_BACKEND", "").strip()
        )
        if has_celery_env:
            backend = "celery"
        else:
            backend = "diskcache"

    if backend == "celery":
        global celery_app
        broker_url = os.getenv("OAB_CELERY_BROKER_URL", "").strip()
        result_backend = os.getenv("OAB_CELERY_RESULT_BACKEND", "").strip()
        if not broker_url or not result_backend:
            raise RuntimeError(
                "Celery background callbacks require OAB_CELERY_BROKER_URL and "
                "OAB_CELERY_RESULT_BACKEND to be set."
            )

        try:
            from celery import Celery
            from dash import CeleryManager
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Celery background callback dependencies are missing. Install with:\n"
                "  uv sync --extra background-celery\n"
                'or:\n  pip install "openairbearing[background-celery]"'
            ) from exc

        celery_app = Celery(
            "openairbearing-dash",
            broker=broker_url,
            backend=result_backend,
        )
        return CeleryManager(celery_app)

    if backend == "diskcache":
        cache_dir = os.getenv("OAB_DISKCACHE_DIR", ".openairbearing-cache").strip()
        try:
            import diskcache
            from dash import DiskcacheManager
        except ModuleNotFoundError as exc:
            if os.getenv("OAB_DASH_BACKGROUND_BACKEND", "").strip().lower() == "auto":
                # In auto mode, fall back to foreground execution if DiskCache
                # is not installed.
                return None
            raise RuntimeError(
                "web UI dependencies are missing. Install with:\n"
                '  uv sync --extra ui\nor:\n  pip install "openairbearing[ui]"'
            ) from exc

        return DiskcacheManager(diskcache.Cache(cache_dir))

    raise RuntimeError(
        "Unsupported OAB_DASH_BACKGROUND_BACKEND value: "
        f"{backend!r}. Expected one of: none, celery, diskcache."
    )


if not hasattr(pkgutil, "find_loader"):

    def _find_loader(fullname):
        spec = importlib.util.find_spec(fullname)
        return spec.loader if spec else None

    pkgutil.find_loader = _find_loader


def build_app():
    """Create and configure the Dash application."""
    _require_ui_dependencies()

    import dash

    from openairbearing.app.ui_callbacks import register_callbacks
    from openairbearing.app.ui_layouts import create_layout
    from openairbearing.bearings import CircularBearing
    from openairbearing.solution_analytic import solve_bearing_analytic

    default_bearing = CircularBearing()

    bearing = default_bearing
    result = solve_bearing_analytic(bearing)
    assets_dir = Path(__file__).resolve().parent / "assets"
    background_callback_manager = build_background_callback_manager()

    app = dash.Dash(
        __name__,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
        ],
        title="OpenAirBearing",
        update_title="Loading...",
        assets_folder=str(assets_dir),
        background_callback_manager=background_callback_manager,
    )

    app.layout = create_layout(default_bearing, bearing, result)

    register_callbacks(app)

    app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            <p> Open Air Bearing is available as open-source software
            under MIT license:
            <a href="https://github.com/Aalto-Arotor/openAirBearing">
            github.com/Aalto-Arotor/openAirBearing</a> - Contact:
            <a href="mailto:mikael.miettinen@iki.fi">
            mikael.miettinen@iki.fi</a></p>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

    return app


try:
    app = build_app()
except RuntimeError:
    app = None

server = app.server if app is not None else None


def main():
    """Run the Dash development server."""
    app_instance = app if app is not None else build_app()
    app_instance.run_server(debug=True)


if __name__ == "__main__":
    main()
