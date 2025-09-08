export interface PredictionResponse {
  ticker: string;
  last_actual_price: number;
  next_predicted_price: number;
  error?: string;
}
