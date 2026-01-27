import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class AIAdvisor:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        else:
            self.model = None

    def get_suggestion(self, forecast_data, inventory_data):
        if not self.model:
            return "Gemini API key not configured. Please add GEMINI_API_KEY to your .env file."

        prompt = f"""
        Analyze the following sales forecast and inventory data for a Walmart store department:
        
        Forecast:
        - Next Week: {forecast_data['next_week_sales']}
        - Next Month: {forecast_data['next_month_sales']}
        - Next 3 Months: {forecast_data['next_3_month_sales']}
        
        Inventory:
        - Status: {inventory_data['stock_status']}
        - Reorder Point: {inventory_data['reorder_point']}
        - Safety Stock: {inventory_data['safety_stock']}
        - Recommended Order Qty: {inventory_data['recommended_order_qty']}
        
        Provide a concise, actionable business recommendation (2-3 sentences).
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating suggestion: {str(e)}"
