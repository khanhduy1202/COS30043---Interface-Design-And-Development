from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import stock_prediction_1 
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import json

from typing import List
from fastapi import HTTPException
from datetime import timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockParameters(BaseModel):
    company: str
    startDate: str
    endDate: str

class PredictStockParameters(BaseModel):
    company: str
    startDate: str
    priceValue: str
    predictionDays: int
    units: int
    nLayers: int
    dropout: float
    epochs: int
    batchSize: int
    numOfDays: int
    
class ChartParameters(BaseModel):
    company: str
    startDate: str
    endDate: str

@app.post("/api/fetch_stock_data")
async def get_stock_data(parameters: StockParameters):
    # Retrieve parameter data from the request
    DATASOURCE = "Yahoo"
    COMPANY = parameters.company  
    START_DATE = parameters.startDate
    END_DATE = parameters.endDate  

    data = stock_prediction_1.load_data(datasource=DATASOURCE, ticker=COMPANY, start_date=START_DATE, end_date=END_DATE)
    return data['og_df'].to_dict(orient='records')

@app.post("/api/predict_stock_data")
async def predict_stock(parameters: PredictStockParameters):
    # Retrieve parameter data from the request
    DATASOURCE = "Yahoo"
    COMPANY = parameters.company  
    START_DATE = parameters.startDate
    END_DATE = stock_prediction_1.dt.datetime.now()
    PRICE_VALUE = parameters.priceValue
    PREDICTION_DAYS = parameters.predictionDays

    data = stock_prediction_1.load_data(datasource=DATASOURCE, ticker=COMPANY, start_date=START_DATE, end_date=END_DATE, price_value=PRICE_VALUE, prediction_days=PREDICTION_DAYS)
    
    UNITS = parameters.units
    N_LAYERS = parameters.nLayers
    DROPOUT = parameters.dropout
    EPOCHS = parameters.epochs
    BATCHSIZE = parameters.batchSize
    X_TRAIN = data['x_train']
    Y_TRAIN = data['y_train']
    NUMOFDAYS = parameters.numOfDays
    
    model = stock_prediction_1.create_model(units=UNITS, n_layers=N_LAYERS, dropout=DROPOUT, x_train=X_TRAIN)
    model.fit(X_TRAIN, Y_TRAIN, epochs=EPOCHS, batch_size=BATCHSIZE)
    data1 = data["og_df"]
    data2 =  stock_prediction_1.yf.download(COMPANY, start=START_DATE, end=END_DATE)
    test_data = data2.reset_index()
    test_data = test_data[1:]
    total_dataset = stock_prediction_1.pd.concat((data1[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
    model_inputs = model_inputs.reshape(-1, 1)
    scaler = stock_prediction_1.MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data1[PRICE_VALUE].values.reshape(-1, 1))
    model_inputs = scaler.transform(model_inputs)
    real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
    real_data = stock_prediction_1.np.array(real_data)
    real_data = stock_prediction_1.np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    predicted_prices = stock_prediction_1.multivariate_multistep_predict(model, real_data, NUMOFDAYS, scaler)
    predicted_prices_list = predicted_prices.tolist()
    next_day_date = END_DATE + timedelta(days=1)

    # Create a list of dictionaries containing date and predicted price
    predictions_with_dates = [
        {"date": (next_day_date + timedelta(days=i)).strftime("%Y-%m-%d"), "price": price[0]}
        for i, price in enumerate(predicted_prices_list)
    ]
    return predictions_with_dates

@app.post("/api/get_chart_data")
async def get_chart_data(parameters: ChartParameters):
    # Retrieve parameter data from the request
    COMPANY = parameters.company
    START_DATE = parameters.startDate
    END_DATE = parameters.endDate

    data = yf.download(COMPANY, start=START_DATE, end=END_DATE)

    # Create the candlestick chart figure
    boxplot_fig = create_boxplot_chart(data)
    candlestick_data = [{
        "open": row['Open'],
        "high": row['High'],
        "low": row['Low'],
        "close": row['Close'],
    } for _, row in data.iterrows()]

    # Convert the figures to JSON
    boxplot_json = boxplot_fig.to_json()

    response_data = {
        "candlestick_data": candlestick_data,
        "boxplot_data": json.loads(boxplot_json),
    }
    return response_data


def create_candlestick_chart(data):
    fig = make_subplots(rows=1, cols=1)
    candlestick_trace = go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlesticks',
    )
    fig.add_trace(candlestick_trace)

    fig.update_layout(
        title="Candlestick Chart",
        xaxis_rangeslider_visible=False,
    )

    return fig

def create_boxplot_chart(data):
    fig = go.Figure()
    
    for column_name in data.columns:
        fig.add_trace(go.Box(y=data[column_name], name=column_name, boxpoints="all"))
    
    fig.update_layout(title="Boxplot Chart")
    return fig


