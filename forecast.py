import pandas as pd
import numpy as np
from datetime import date, datetime

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SARIMAX, AutoReg
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics  import mean_squared_error, median_absolute_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

import warnings
warnings.filterwarnings("ignore")


class Model:
    '''
    Класс "модель" принимает на вход файл.
    Имеет следующие атрибуты:
        file_name - наименование файла;
        sheets - листы внутри файла;
        excel_file - pd.ExcelFile, необходимо для считывания листов в файле и сбора данных с них;
        data - pd.DataFrame, получены методом preprocess_excel, туда же кладутся результаты прогнозов
        models_scores - оценка моделей по MAE на тестовой выборке (равно количеству точек для прогноза)
        ts_series - dlog_sa ряды не в виде df, в виде словаря тип ИПЦ - dlog_sa
        forecast - прогноз моделей на выбранное количество точек вперед
        chosen_forecast - те из forecast, которые выбраны по минимальной MAE models_scores
    Обладает следующими методами:
        preprocess_excel - обработка файла, возвращает готовый pd.DataFrame в атрибуте self.data для моделирования;
        make_ts - создает time series выбранной перменной и показателя
        evaluate_models - обучение моделей на train выборке и подсчет MAE на тестовой для заданного числа точек прогноза
        make_predictions - прогноз на выбранное число точек вперед
        transform_forecast - добавление данных прогноза в self.data
    '''
    
    months_names = {
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель":  4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12
    }
    
    models_specs = {"ИПЦ": {
                                "ARIMA": {"order": (2, 0, 1)}, 
                                "AutoReg": {"lags": 2, "trend": "c"}, 
                                "GradientBoostingRegressor": {"max_depth": 5, "min_samples_leaf": 5, "n_estimators": 1000, "learning_rate": 0.1}
                            },
                    "Непродовольственный ИПЦ": {
                               "ARIMA": {"order": (1, 0, 1)}, 
                               "AutoReg": {"lags": 1, "trend": "c"}, 
                               "GradientBoostingRegressor": {"n_estimators": 1000, "min_samples_leaf": 5, "max_depth": 7, "learning_rate": 0.01}
                           },
                    "Продовольственный ИПЦ": {
                               "ARIMA": {"order": (2, 0, 1)}, 
                               "AutoReg": {"lags": 2, "trend": "c"},  
                               "GradientBoostingRegressor": {"n_estimators": 1000, "min_samples_leaf": 1, "max_depth": 5, "learning_rate": 0.1}
                           },
                    "ИПЦ услуг": {
                               "ARIMA": {"order": (2, 1, 1)}, 
                               "AutoReg": {"lags": [1, 2, 4, 5, 6], "trend": "c"}, 
                               "GradientBoostingRegressor": {"n_estimators": 500, "min_samples_leaf": 3, "max_depth": 7, "learning_rate": 0.01}
                           }}
                            
    def __init__(self, file_name: str, data = pd.DataFrame(), sheets = []):
        self.file_name = file_name
        self.sheets = sheets
        self.data = data
        self.excel_file = None
        self.models = {}
        self.models_scores = pd.DataFrame()
        self.ts_series = {}
        self.forecast = pd.DataFrame()
        self.chosen_forecast = pd.DataFrame()

    def __str__(self):
        self.file_name = str(self.file_name)
        return self.file_name
        
    def __read_excel(self):
        '''
        Считывает файл и выводит его листы, за исключением Содержания
        '''
        self.excel_file = pd.ExcelFile(self.file_name, engine = 'openpyxl')
        if not self.sheets:
            self.sheets = list(filter(lambda name: "Содерж" not in name, self.excel_file.sheet_names))
        else:
            self.sheets = list(filter(lambda name: "Содерж" not in name, self.sheets))
        pass
    
    def __extract_sheet(self, sheet):
        '''
        Метод по вытягиванию данных из страницы Excel файла, возвращает DF: значение ИПЦ MoM, дата, тип ИПЦ, base
        Для модели убираем данные до 01-01-2000
        '''
        sheet = pd.read_excel(self.excel_file, sheet)
        # убираем верхние колонки, чтобы добраться до данных
        df = sheet.drop(index=[0, 1, 3])
        df.reset_index(inplace = True, drop = True)
        # переназначаем колнки - потребуется, чтобы превратить таблицу в длинный формат
        df.columns = ["Месяц"] + list(map(lambda x: int(x), df.iloc[0,1:].to_list()))
        df.drop(index = [0], inplace = True)
        df.reset_index(inplace = True, drop = True)
        # уменьшаем размер матрица до необходимых данных
        df = df.iloc[0:12, :]
        # делаем в длинный формат
        df = df.melt(id_vars = ["Месяц"], value_vars = df.columns.to_list().remove("Месяц"))
        df.dropna(axis = 0, inplace = True)
        # формируем дату и сразу делаем индексом
        df["date"] = pd.to_datetime(dict(year = df["variable"], month = df["Месяц"].map(Model.months_names), day = 1))
        df.drop(columns=["Месяц", "variable"], inplace = True)
        # на всякий еще назовем тип ИПЦ
        df["ipc_type"] = sheet.columns[0]
        # Уберем данные инфляции 90-х
        df = df[df["date"] >= datetime(1999, 12, 1)]
        df["base"] = df["value"].apply(lambda x: x / 100).cumprod()
        df["base_log"] = np.log(df["base"])
        df["dlog"] = df["base_log"].diff()
        

        return df

    def make_ts(self, series_label: str, var_type: str):
        '''
        Создает time series для дальнейшего использования в прогнозе
        '''
        if self.data.empty:
            self.preprocess_excel()
        df = self.data[self.data["ipc_type"] == series_label][["date", var_type]].dropna()
        return pd.Series(data = df[var_type].values, index = df["date"].values, name = series_label + "_" + var_type)

    def __sa_decompose(self, ts):
        '''
        Процедура по получению сезонности ряда, возвращает сезонную компоненту и сезонно сглаженный ряд
        '''
        sd = seasonal_decompose(ts)
        df_sa = pd.concat([pd.Series(sd.seasonal, name = "sa_component"), 
                           pd.Series(ts - sd.seasonal, name = "sa"),
                           np.log(pd.Series(ts - sd.seasonal, name = "log_sa")), 
                           np.log(pd.Series(ts - sd.seasonal, name = "dlog_sa")).diff()],
                           axis = 1)
        df_sa["ipc_type"] = ts.name.split("_")[0]
        return df_sa
        
        
    
    def preprocess_excel(self):
        '''
        Метод подготовки данных из эксель для модели, результат записывается в атрибут data/
        '''
        self.__read_excel()
        
        # проверяем, что df пустой, иначе очищаем данные
        if self.data.empty:
            self.data = pd.DataFrame()
        # пробегаемся по листам с данными и возвращаем итоговый датасет
        for sheet in self.sheets:
            self.data =  pd.concat([self.data, self.__extract_sheet(sheet)])

        # features for boosting
        self.data["year"] = self.data["date"].apply(lambda x: x.year)
        self.data["month"] = self.data["date"].apply(lambda x: x.month)
        self.data["quarter"] = self.data["date"].apply(lambda x: x.quarter) 
        
        # упрощаем наименования колонок
        self.data["ipc_type"] = np.where(
            self.data["ipc_type"].str.contains("товары и услуги"), "ИПЦ",
                np.where((self.data["ipc_type"].str.contains("продовольств")) & (self.data["ipc_type"].str.contains("непродовольств") == False), 
                         "Продовольственный ИПЦ",
                         np.where(self.data["ipc_type"].str.contains("непродовольств"), "Непродовольственный ИПЦ", "ИПЦ услуг")
                    )
        )
        
        # добавим данные по seasonal decompose
        df_sd = pd.concat([self.__sa_decompose(self.make_ts(series_label, "base")) for series_label in self.data["ipc_type"].unique()], axis = 0).reset_index().rename(columns = {"index": "date"})
        self.data = pd.merge(left = self.data, right = df_sd, how = "left", left_on = ["date", "ipc_type"], right_on = ["date", "ipc_type"])
        # ts для прогноза - берем base_sa ts
        self.ts_series = {series_label: self.make_ts(series_label, "dlog_sa") for series_label in self.data["ipc_type"].unique()}
        
        return self.data

    def evaluate_models(self, lags: int):
        ''' 
        Оценка лучшей модели, делит выборку на train test по выбранному лагу и оценивает лучшую вариант прогнозирования
        '''
        for series in self.ts_series.keys():
            df = self.data[self.data["ipc_type"] == series].drop(columns = "ipc_type")
            models_params = Model.models_specs[series]
            train, test = df.iloc[:-lags, :].dropna(axis = 0), df.iloc[-lags:, :].dropna(axis = 0)
            ts_train = pd.Series(train["dlog_sa"].values, index = train["date"].values)
            ts_test = pd.Series(test["dlog_sa"].values, index = test["date"].values)
            X_train, X_test, y_train, y_test = train[["year", "month", "quarter"]], test[["year", "month", "quarter"]], train["dlog_sa"], test["dlog_sa"]
                
            # модели для прогноза
            arima_model = ARIMA(ts_train, order = models_params["ARIMA"]["order"]).fit()
            autoreg_model = AutoReg(ts_train, lags = models_params["AutoReg"]["lags"], trend = models_params["AutoReg"]["trend"]).fit()
            gbm_model = GradientBoostingRegressor(n_estimators = models_params["GradientBoostingRegressor"]["n_estimators"],
                                                      min_samples_leaf = models_params["GradientBoostingRegressor"]["min_samples_leaf"],
                                                      max_depth = models_params["GradientBoostingRegressor"]["max_depth"],
                                                      learning_rate = models_params["GradientBoostingRegressor"]["learning_rate"]).\
                                                      fit(X_train, y_train)

                # проверка MAE на тестовой, выбор лучшей
            arima_preds = arima_model.forecast(steps = lags)
            autoreg_preds = autoreg_model.forecast(steps = lags)
            gbm_preds = pd.Series(gbm_model.predict(X_test), index = arima_preds.index)
            ensemble_preds = (arima_preds + autoreg_preds + gbm_preds) / 3
            maes = pd.DataFrame({"models": ["ARIMA", "autoreg_model", "gbm_model", "ensemble"], 
                                 "MAE": [mean_absolute_error(y_test, arima_preds.values), 
                                             mean_absolute_error(y_test, autoreg_preds.values),
                                             mean_absolute_error(y_test, gbm_preds.values),
                                             mean_absolute_error(y_test, ensemble_preds.values)],
                                 "forecast_testv": [arima_preds, autoreg_preds, gbm_preds, ensemble_preds]
                        })
            maes["ipc_type"] = series
            self.models_scores = pd.concat([self.models_scores, maes], axis= 0, ignore_index=True)

            pass

    def make_predictions(self, steps: int):
        '''
        Модель прогноза на n шагов вперед
        '''
        if self.data.empty:
            self.preprocess_excel()

        if self.models_scores.empty:
            self.evaluate_models(steps)
            
        dates = pd.DataFrame({"date": pd.date_range(start = max(self.data["date"]), periods = steps + 1, freq = 'MS')[1: ]})
        dates["month"] = dates["date"].apply(lambda x: x.month)
        dates["year"] = dates["date"].apply(lambda x: x.year)
        dates["quarter"] = dates["date"].apply(lambda x: x.quarter)

        best_models_id = self.models_scores.groupby(["ipc_type"])["MAE"].idxmin()
        best_models = self.models_scores.loc[best_models_id.values, ["models", "ipc_type"]]

        for series in self.ts_series.keys():
            df = self.data[self.data["ipc_type"] == series].drop(columns = "ipc_type").dropna()
            ts = pd.Series(df["dlog_sa"].values, index = df["date"].values)
            X, y = df[["year", "month", "quarter"]], df["dlog_sa"]
            models_params = Model.models_specs[series]
            # заводим модели на всю выборку для вневыборочного прогноза
            arima = ARIMA(ts, order = models_params["ARIMA"]["order"]).fit()
            autoreg = AutoReg(ts, lags = models_params["AutoReg"]["lags"], trend = models_params["AutoReg"]["trend"]).fit()
            gbm = GradientBoostingRegressor(n_estimators = models_params["GradientBoostingRegressor"]["n_estimators"],
                                                      min_samples_leaf = models_params["GradientBoostingRegressor"]["min_samples_leaf"],
                                                      max_depth = models_params["GradientBoostingRegressor"]["max_depth"],
                                                      learning_rate = models_params["GradientBoostingRegressor"]["learning_rate"]).\
                                                      fit(X, y)
            # результаты прогноза
            arima_preds = arima.forecast(steps = steps)
            autoreg_preds = autoreg.forecast(steps = steps)
            gbm_preds = pd.Series(data = gbm.predict(dates[["year", "month", "quarter"]]), index = arima_preds.index)
            ensemble_preds = (arima_preds + autoreg_preds + gbm_preds) / 3
            
            forecast = pd.DataFrame({"models": ["ARIMA", "autoreg_model", "gbm_model", "ensemble"], 
                                     "forecast": [arima_preds.values, autoreg_preds.values, gbm_preds.values, ensemble_preds.values],
                                    "date": [arima_preds.index, autoreg_preds.index, gbm_preds.index, ensemble_preds.index]})
            forecast["ipc_type"] = series
            self.forecast = pd.concat([self.forecast, forecast])
            best_model = best_models[best_models["ipc_type"] == series]["models"].values[0]
            chosen_forecast = forecast[forecast["models"] == best_model]
            self.chosen_forecast = pd.concat([self.chosen_forecast, chosen_forecast], axis = 0, ignore_index = True)

        pass

    def transform_forecast(self):
        '''
        Возвращает предсказания dlog_sa в MoM sa обратно
        '''
        output = pd.DataFrame()
        
        for series in self.chosen_forecast["ipc_type"].values:
            df = self.chosen_forecast[self.chosen_forecast["ipc_type"] == series]
            preds_df = pd.DataFrame({"date": df["date"].values[0], "forecast_dlog_sa": df["forecast"].values[0]})
            preds_df["ipc_type"] = series
            output = pd.concat([output, preds_df], axis = 0, ignore_index = True)

        self.data = pd.merge(left = self.data, right = output, how = "outer", left_on = ["date", "ipc_type"], right_on = ["date", "ipc_type"])

        mom_sa = pd.DataFrame()
        for series in self.data["ipc_type"].unique():
            df_tf = self.data[self.data["ipc_type"] == series].reset_index()
            df_tf["tf"] = df_tf["dlog_sa"].fillna(0) + df_tf["forecast_dlog_sa"].fillna(0)
            df_tf.loc[0, "tf"] = df_tf.loc[0, "log_sa"]
            df_tf["log_sa_final"] = df_tf["tf"].cumsum()
            df_tf["sa_final"]  = df_tf["log_sa_final"].apply(np.exp)
            df_tf["MoM_sa_final"] = df_tf["sa_final"] / df_tf["sa_final"].shift() + 100

            mom_sa = pd.concat([mom_sa, df_tf[["date", "ipc_type", "MoM_sa_final"]]], axis = 0, ignore_index=True)



        self.data = pd.merge(left = self.data, right = mom_sa, 
                                 how = "outer", left_on = ["date", "ipc_type"], right_on = ["date", "ipc_type"])
            

        return self.data
  

    
