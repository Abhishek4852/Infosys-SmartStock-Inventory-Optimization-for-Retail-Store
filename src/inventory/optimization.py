import math

class InventoryOptimizer:
    def __init__(self, service_level=1.65, lead_time=7):
        """
        service_level: Z-score (1.65 for 95% service level)
        lead_time: Lead time in days
        """
        self.Z = service_level
        self.lead_time = lead_time

    def calculate_metrics(self, current_stock, predicted_sales, historical_std):
        """
        predicted_sales: Forecasted sales for the next week
        historical_std: Standard deviation of demand
        """
        # Average Daily Demand
        avg_daily_demand = predicted_sales / 7
        
        # Safety Stock = Z * Demand_Std * sqrt(Lead_Time)
        # Assuming lead time is in days
        safety_stock = self.Z * historical_std * math.sqrt(self.lead_time)
        
        # Reorder Point = (Average Daily Demand * Lead Time) + Safety Stock
        reorder_point = (avg_daily_demand * self.lead_time) + safety_stock
        
        # Stock Status
        if current_stock <= 0:
            status = "OUT OF STOCK"
        elif current_stock < safety_stock:
            status = "UNDERSTOCK"
        elif current_stock >= reorder_point:
            status = "HEALTHY"
        else:
            status = "REORDER RECOMMENDED" # Between Safety and ROP
            
        # Recommended Order Quantity
        recommended_order_qty = max(0, reorder_point - current_stock) if status != "HEALTHY" else 0
        
        return {
            "reorder_point": round(reorder_point, 2),
            "safety_stock": round(safety_stock, 2),
            "stock_status": status,
            "recommended_order_qty": round(recommended_order_qty, 2)
        }
