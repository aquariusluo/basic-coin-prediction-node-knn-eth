import json
import os
from flask import Flask, Response
from model import download_data, format_data, train_model, get_inference
from config import model_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER, data_base_path, CG_API_KEY, MODEL

app = Flask(__name__)

def update_data():
    print("Starting data update process...")
    training_price_data_path = os.path.join(data_base_path, "price_data.csv")
    
    # Log config values
    print(f"Config: TOKEN={TOKEN}, TRAINING_DAYS={TRAINING_DAYS}, TIMEFRAME={TIMEFRAME}, MODEL={MODEL}, REGION={REGION}, DATA_PROVIDER={DATA_PROVIDER}, CG_API_KEY={CG_API_KEY}")
    
    # Validate critical config values
    if not TRAINING_DAYS or not TRAINING_DAYS.isdigit():
        raise ValueError(f"TRAINING_DAYS must be a number, got: {TRAINING_DAYS}")
    training_days = int(TRAINING_DAYS)  # Convert to integer
    if training_days <= 0:
        raise ValueError(f"TRAINING_DAYS must be positive, got: {training_days}")
    if DATA_PROVIDER not in ["binance", "coingecko"]:
        raise ValueError(f"DATA_PROVIDER must be 'binance' or 'coingecko', got: {DATA_PROVIDER}")
    if DATA_PROVIDER == "coingecko" and not CG_API_KEY:
        raise ValueError("CG_API_KEY is required for CoinGecko but is None")

    # Force deletion of old price_data.csv to ensure fresh data
    if os.path.exists(training_price_data_path):
        print(f"Deleting existing {training_price_data_path}")
        os.remove(training_price_data_path)
    
    try:
        print(f"Downloading BTC data with TRAINING_DAYS={training_days}, REGION={REGION}, DATA_PROVIDER={DATA_PROVIDER}")
        files_btc = download_data("BTC", training_days, REGION, DATA_PROVIDER)
        print(f"Downloading ETH data with TRAINING_DAYS={training_days}, REGION={REGION}, DATA_PROVIDER={DATA_PROVIDER}")
        files_eth = download_data("ETH", training_days, REGION, DATA_PROVIDER)
        
        if not files_btc or not files_eth:
            print("Warning: No new data files downloaded for BTC or ETH")
            if os.path.exists(training_price_data_path):
                print(f"Using existing {training_price_data_path} for training")
                train_model(TIMEFRAME)
                return
            else:
                raise ValueError("No data files downloaded for BTC or ETH, and no existing price_data.csv")
        
        print("Formatting data...")
        format_data(files_btc, files_eth, DATA_PROVIDER)
        
        if not os.path.exists(training_price_data_path):
            raise FileNotFoundError(f"{training_price_data_path} was not created by format_data")
        
        print("Training model...")
        train_model(TIMEFRAME)
    except Exception as e:
        print(f"Error in update_data: {str(e)}")
        raise

@app.route("/inference/<string:token>")
def generate_inference(token):
    if not token or token.upper() != TOKEN:
        error_msg = "Token is required" if not token else "Token not supported"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')
    try:
        inference = get_inference(token.upper(), TIMEFRAME, REGION, DATA_PROVIDER)
        return Response(str(inference), status=200)
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

@app.route("/update")
def update():
    try:
        update_data()
        return "0"
    except Exception as e:
        print(f"Update failed: {str(e)}")
        return "1"

if __name__ == "__main__":
    update_data()  # Regenerate data and train model on startup
    app.run(host="0.0.0.0", port=8000)
