from flask import Blueprint

uf_bp = Blueprint('uf_bp', __name__,
                    template_folder='templates',
                    static_folder='static')

from app.uploadFile import routes