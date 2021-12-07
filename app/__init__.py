from flask import Flask
from flask_bootstrap import Bootstrap
bootstrap = Bootstrap()

def create_app():
    """Construct the core application."""
    app = Flask(__name__,
                instance_relative_config=True)
    app.config.from_object('config.Config')

    with app.app_context():
        from app.uploadFile import routes as uf_route
        app.register_blueprint(uf_route.uf_bp)
        from app.assets import compile_assets
        compile_assets(app)
        bootstrap.init_app(app)
        return app
