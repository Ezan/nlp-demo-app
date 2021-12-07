from flask import request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField

class UploadFile(FlaskForm):
    file = FileField()
    def getData(self):
        # RETURN THE READ DATA
        pass