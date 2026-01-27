import sys
from pathlib import Path

# Add project root to sys.path to allow running from within 'src/api'
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.inference.predictor import SalesPredictor
from src.inventory.optimization import InventoryOptimizer
from src.ai_advisor.advisor import AIAdvisor
import pandas as pd
import os
import json
import plotly
import plotly.graph_objs as go

app = FastAPI(title="Walmart Sales Forecasting API")

# Setup templates with absolute path
templates = Jinja2Templates(directory=str(root_path / "frontend" / "templates"))

# Initialize components with absolute model path if needed
# (Assuming SalesPredictor handles its own relative path, but let's be safe)
predictor = SalesPredictor(model_dir=str(root_path / "model_artifacts"))
inventory_optimizer = InventoryOptimizer()
ai_advisor = AIAdvisor()

class PredictionRequest(BaseModel):
    store: int
    dept: int
    current_stock: float
    temperature: float = 42.5
    fuel_price: float = 3.1
    is_holiday: bool = False
    markdown1: float = 0
    markdown2: float = 0
    markdown3: float = 0
    markdown4: float = 0
    markdown5: float = 0
    cpi: float = 211.0
    unemployment: float = 8.1
    size: int = 151315
    type: str = 'A'

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def post_predict(
    request: Request,
    store: int = Form(...),
    dept: int = Form(...),
    size: int = Form(...),
    store_type: str = Form(...),
    period: str = Form(...),
    current_stock: float = Form(...),
    lead_time: int = Form(...)
):
    try:
        # Construct input for predictor
        now = pd.Timestamp.now()
        input_data = {
            'Store': store,
            'Dept': dept,
            'IsHoliday': 0, # Default
            'Temperature': 45.0, # Default
            'Fuel_Price': 3.5, # Default
            'MarkDown1': 0, 'MarkDown2': 0, 'MarkDown3': 0, 'MarkDown4': 0, 'MarkDown5': 0,
            'CPI': 212.0,
            'Unemployment': 7.5,
            'Size': size,
            'Year': now.year,
            'Month': now.month,
            'Week': now.isocalendar()[1],
            'Day': now.day,
            'DayOfWeek': now.weekday(),
            'Lag_1': 20000, 'Lag_2': 20000, 'Lag_3': 20000, 'Lag_4': 20000, 
            'Lag_8': 20000, 'Lag_12': 20000, 'Lag_16': 20000, 'Lag_20': 20000, 'Lag_24': 20000,
            'RollingMean_4': 20000, 'RollingStd_4': 1000, 
            'RollingMean_12': 20000, 'RollingStd_12': 1500
        }

        # 1. Forecast Sales
        forecast = predictor.predict(input_data)
        
        # Determine prediction value based on period
        target_sales = forecast['next_week_sales']
        if period == 'month':
            target_sales = forecast['next_month_sales']
        elif period == '3months':
            target_sales = forecast['next_3_month_sales']

        # 2. Inventory Optimization
        inventory = inventory_optimizer.calculate_metrics(
            current_stock, 
            target_sales, 
            historical_std=2000
        )
        
        # 3. AI Suggestions
        ai_suggestion = ai_advisor.get_suggestion(forecast, inventory)

        # Prepare results for template
        predictions = [{
            'Date': f"Forecast for {period}",
            'Predicted_Sales': f"${target_sales:,.2f}",
            'Stock_Status': inventory['stock_status'],
            'Out_of_Stock_Risk': "High" if inventory['stock_status'] != "HEALTHY" else "Low",
            'Reorder_Point': f"{inventory['reorder_point']:.0f} units"
        }]

        # Create Plotly Charts
        # 1. Sales Trend Plot
        sales_fig = go.Figure()
        sales_fig.add_trace(go.Bar(
            x=['Next Week', 'Next Month', 'Next 3 Months'],
            y=[forecast['next_week_sales'], forecast['next_month_sales'], forecast['next_3_month_sales']],
            marker_color='#58A6FF'
        ))
        sales_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e6f0fb',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300
        )
        graphJSON_sales = json.dumps(sales_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # 2. Stock Plot
        stock_fig = go.Figure()
        stock_fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = current_stock,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Current Stock Level"},
            gauge = {
                'axis': {'range': [None, inventory['reorder_point'] * 1.5]},
                'steps': [
                    {'range': [0, inventory['safety_stock']], 'color': "red"},
                    {'range': [inventory['safety_stock'], inventory['reorder_point']], 'color': "yellow"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': inventory['reorder_point']
                }
            }
        ))
        stock_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e6f0fb',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300
        )
        graphJSON_stock = json.dumps(stock_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Prepare Gemini Summary Lines
        summary_lines = [
            "Expected Sales Trend",
            f"The forecast indicates a demand of ${target_sales:,.2f} for the selected period.",
            "Stockout Risk",
            f"Current status is {inventory['stock_status']}.",
            "Safety Stock Need",
            f"Maintain at least {inventory['safety_stock']:.0f} units as safety stock.",
            "Reorder Point Details",
            f"A reorder should be triggered if stock falls below {inventory['reorder_point']:.0f} units.",
            "Final Action Plan",
            ai_suggestion
        ]

        return templates.TemplateResponse("results.html", {
            "request": request,
            "predictions": predictions,
            "graphJSON_sales": graphJSON_sales,
            "graphJSON_stock": graphJSON_stock,
            "summary_lines": summary_lines
        })

    except Exception as e:
        print(f"Error in post_predict: {str(e)}")
        # Fallback error response
        return HTMLResponse(content=f"<h3>Error processing prediction: {str(e)}</h3>", status_code=500)

@app.post("/predict")
async def predict_api(request: PredictionRequest):
    try:
        # API version of prediction
        now = pd.Timestamp.now()
        input_data = {
            'Store': request.store,
            'Dept': request.dept,
            'IsHoliday': int(request.is_holiday),
            'Temperature': request.temperature,
            'Fuel_Price': request.fuel_price,
            'MarkDown1': request.markdown1,
            'MarkDown2': request.markdown2,
            'MarkDown3': request.markdown3,
            'MarkDown4': request.markdown4,
            'MarkDown5': request.markdown5,
            'CPI': request.cpi,
            'Unemployment': request.unemployment,
            'Size': request.size,
            'Year': now.year,
            'Month': now.month,
            'Week': now.isocalendar()[1],
            'Day': now.day,
            'DayOfWeek': now.weekday(),
            'Lag_1': 20000, 'Lag_2': 20000, 'Lag_3': 20000, 'Lag_4': 20000, 
            'Lag_8': 20000, 'Lag_12': 20000, 'Lag_16': 20000, 'Lag_20': 20000, 'Lag_24': 20000,
            'RollingMean_4': 20000, 'RollingStd_4': 1000, 
            'RollingMean_12': 20000, 'RollingStd_12': 1500
        }
        
        forecast = predictor.predict(input_data)
        inventory = inventory_optimizer.calculate_metrics(
            request.current_stock, 
            forecast['next_week_sales'], 
            historical_std=2000
        )
        ai_suggestion = ai_advisor.get_suggestion(forecast, inventory)
        
        return {
            **forecast,
            **inventory,
            "ai_suggestion": ai_suggestion
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
