import os

from openairbearing import build_app


def main():
    os.environ.setdefault("OAB_DASH_BACKGROUND_BACKEND", "diskcache")
    app = build_app()
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
