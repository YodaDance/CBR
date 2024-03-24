import pandas as pd
import numpy as np
import plotly.express as plt
from datetime import date, datetime


class Model:
    '''
    Класс "модель" принимает на вход файл.
    Имеет следующие атрибуты:
    
    Обладает следующими методами:
    
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
    
    def __init__(self, file_name: str, data = pd.DataFrame(), sheets = []):
        self.file_name = file_name
        self.sheets = sheets # указать конкретный лист
        self.data = data
        self.excel_file = None

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
        df = df[df["date"] >= datetime(2000, 1, 1)]
        df["base"] = df["value"].apply(lambda x: x / 100).cumprod()

        return df
    
    def preprocess_excel(self):
        self.__read_excel()
        
        # проверяем, что df пустой, иначе очищаем данные
        if self.data.empty:
            self.data = pd.DataFrame()
        # пробегаемся по листам с данными и возвращаем итоговый датасет
        for sheet in self.sheets:
            self.data =  pd.concat([self.data, self.__extract_sheet(sheet)])
            
        pass
    

    
