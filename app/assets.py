from flask_assets import Environment, Bundle


def compile_assets(app):
    """Configure authorization asset bundles."""
    Environment.auto_build = True
    Environment.debug = False
