import os
from flask import Flask, flash, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from forecast import Model
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import datetime
import numpy as np

UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'downloads'
ALLOWED_EXTENSIONS = {'xlsx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["DOWNLOAD_FOLDER"] = DOWNLOAD_FOLDER

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
# @app.route('/callback', methods=['POST', 'GET'])
# def cb():
#     return gm(request.args.get('data')) #возвращает json графика
   
@app.route('/index/<file>')
def index(file):
    graphJSONs, table = gm(file)
    return render_template('index.html', graphJSON=graphJSONs[0], graphJSON2=graphJSONs[1], graphJSON3 = graphJSONs[2], table = table)

@app.route('/download')
def download():
    path = 'results.csv'
    return send_from_directory(app.config["DOWNLOAD_FOLDER"], path)

def gm(file):
    df = Model(file)
    df.make_predictions(6)
    df.transform_forecast()
    
    data = df.data[["date", "ipc_type", "value", "base", "sa", "MoM_sa_final"]]
    data["forecast"] = data.apply(lambda row: row["MoM_sa_final"] if np.isnan(row["sa"]) else np.NAN, axis=1)
    data["MoM sa"] = data.apply(lambda row: row["MoM_sa_final"] if np.isnan(row["forecast"]) else np.NAN, axis=1)
    data.rename(columns={"value": "MoM", "sa": "base sa", "MoM_sa_final": "MoM sa with forecast"}, inplace = True)
    data = data[data["date"].apply(lambda x: x.year >= max(data["date"]).year - 3)]
    path = os.path.join(app.config['DOWNLOAD_FOLDER'], "results.csv")
    df.data.to_csv(path)
    
    # график с прогнозом
    dates = data["date"].unique()
    ipc_mom = data[data["ipc_type"] == "ИПЦ"]["MoM sa"]
    prod_ipc_mom = data[data["ipc_type"] == "Продовольственный ИПЦ"]["MoM sa"]
    nprod_ipc_mom = data[data["ipc_type"] == "Непродовольственный ИПЦ"]["MoM sa"]
    services_ipc_mom = data[data["ipc_type"] == "ИПЦ услуг"]["MoM sa"]
    ipc_forecast = data[data["ipc_type"] == "ИПЦ"]["MoM sa with forecast"]
    prod_ipc_forecast = data[data["ipc_type"] == "Продовольственный ИПЦ"]["MoM sa with forecast"]
    nprod_ipc_forecast = data[data["ipc_type"] == "Непродовольственный ИПЦ"]["MoM sa with forecast"]
    services_ipc_forecast = data[data["ipc_type"] == "ИПЦ услуг"]["MoM sa with forecast"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=ipc_mom, name='ИПЦ',
                            line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=dates, y=prod_ipc_mom, name = 'Продовольственный ИПЦ',
                            line=dict(color='red', width=1)))
    fig.add_trace(go.Scatter(x=dates, y=nprod_ipc_mom, name='Непродовольственный ИПЦ',
                            line=dict(color='green', width=1)))
    fig.add_trace(go.Scatter(x=dates, y=services_ipc_mom, name='ИПЦ услуг',
                            line=dict(color='purple', width=1)))
    fig.add_trace(go.Scatter(x=dates, y=ipc_forecast, name='ИПЦ прогноз',
                            line = dict(color='blue', width=1, dash='dash')))
    fig.add_trace(go.Scatter(x=dates, y=prod_ipc_forecast, name='Прод. ИПЦ прогноз',
                            line = dict(color='red', width=1, dash='dash')))
    fig.add_trace(go.Scatter(x=dates, y=nprod_ipc_forecast, name='Непрод. ИПЦ прогноз',
                            line=dict(color='green', width=1, dash='dash')))
    fig.add_trace(go.Scatter(x=dates, y=services_ipc_forecast, name='ИПЦ услуг прогноз',
                            line=dict(color='purple', width=1, dash='dash')))

    # Edit the layout
    fig.update_layout(title='Динамика ИПЦ и прогноз',
                    xaxis_title='Дата',
                    yaxis_title='MoM sa')
    
    fig.update_traces(connectgaps=True)
    
    
    charts = [
        px.line(df.data, x = "date", y = "value", title = "График динамики ИПЦ - MoM", line_group = "ipc_type", color = "ipc_type"),
        px.line(df.data, x = "date", y = "base", title = "График динамики ИПЦ - Base", line_group = "ipc_type", color = "ipc_type"),
        fig
        #px.line(data, x = "date", y = ["MoM sa", "forecast"], title = "График ИПЦ и прогноза", line_group = "ipc_type", color = "ipc_type")
    ]

    graphJSONs = list(map(lambda fig: json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder), charts))
    del df

    return graphJSONs, data.to_html(header = True, index = False, na_rep = '')


if __name__ == '__main__':
    app.run(debug=False)
