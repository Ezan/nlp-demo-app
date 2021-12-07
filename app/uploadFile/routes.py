from flask import render_template, url_for, redirect, request, send_file
from app.uploadFile import uf_bp
from app.uploadFile.forms import UploadFile
from werkzeug.utils import secure_filename
import os
from runEngine import Engine
engine = Engine()

@uf_bp.route('/', methods=['GET','POST'])
def uploadFile():
    uploadFileForm = UploadFile()
    if uploadFileForm.validate_on_submit():
        print('VALID UPLOAD')
        filename = secure_filename(uploadFileForm.file.data.filename)
        if not os.path.exists('./uploads'):
            os.mkdir('./uploads')
        uploadFileForm.file.data.save('./uploads/' + filename)
        return redirect(url_for('uf_bp.processUpload', secure_filename=filename))
    return render_template('uploadFile.html', uploadFileForm=uploadFileForm)

@uf_bp.route('/process_upload', methods=['GET'])
def processUpload():
    secure_filename = request.args.get('secure_filename')
    output_file = engine.checkFileType(secure_filename)
    return render_template('result.html', secure_filename=output_file)

@uf_bp.route('/download_result', methods=['GET'])
def downloadResult():
    secure_filename = request.args.get('secure_filename')
    return send_file(os.path.join(os.getcwd(), 'results/{}'.format(secure_filename)),mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', attachment_filename='result.xlsx',as_attachment=True)