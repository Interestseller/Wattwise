"""
run.py — WattWise Flask application entry point.

Usage:
    python run.py
    flask --app run run --debug
"""

from flask import Flask
import os, sys

# Ensure project root is importable from any working directory
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=os.path.join(ROOT, "flask_app", "templates"),
        static_folder=os.path.join(ROOT, "flask_app", "static"),
    )
    app.secret_key = os.environ.get("SECRET_KEY", "wattwise-dev-secret-2024")

    from flask_app.routes import main
    app.register_blueprint(main)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
