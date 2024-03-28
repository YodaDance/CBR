import os
from flask import Flask, flash, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from forecast import Model
import json
import plotly
import plotly.express as px
import pandas as pd

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'xlsx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('Не выбран файл')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path_to_save)
            return redirect(url_for('index', file=path_to_save))
    return '''
    <!doctype html>
    <h1>Прогноз инфляции на 6 месяцев вперед</h1>
    <h2>Загрузите файл ниже</h2>
    <p>Убедитесь, что файл скачан с официального сайта Росстата в формате xlsx.</p>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Загрузить>
    </form>
    '''

# Идею и код почерпнул тут https://github.com/alanjones2/Flask-Plotly
@app.route('/callback', methods=['POST', 'GET'])
def cb():
    return gm(request.args.get('data')) #возвращает json графика
   
@app.route('/index/<file>')
def index(file):
    graphJSONs, table = gm(file)
    return render_template('index.html',  graphJSON=graphJSONs[0], graphJSON2=graphJSONs[1], graphJSON3 = graphJSONs[2], table = table)

def gm(file):
    df = Model(file)
    df.make_predictions(6)
    df.transform_forecast()
    
    charts = [
        px.line(df.data, x = "date", y = "value", title = "График динамики ИПЦ - MoM", line_group = "ipc_type", color = "ipc_type"),
        px.line(df.data, x = "date", y = "base", title = "График динамики ИПЦ - Base", line_group = "ipc_type", color = "ipc_type"),
        px.line(df.data, x = "date", y = "MoM_sa_final", title = "График ИПЦ и прогноза", line_group = "ipc_type", color = "ipc_type")
    ]

    graphJSONs = list(map(lambda fig: json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder), charts))
    data = df.data[["date", "ipc_type", "value", "base", "sa", "MoM_sa_final"]]
    data.rename({"value": "MoM", "sa": "base_sa", "MoM_sa_final": "MoM_sa with forecast"})
    del df
    
    return graphJSONs, data.to_html(header=True)
