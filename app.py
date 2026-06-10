import os
import json
import pickle
import pandas as pd
from http.server import SimpleHTTPRequestHandler, HTTPServer

PORT = 8000
MODEL_PATH = "model.pkl"
INDEX_PATH = "index.html"

# Load model pipeline at startup
if os.path.exists(MODEL_PATH):
    print(f"Loading trained machine learning pipeline from {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found. Run 'python train_model.py' to generate the model first.")

class PredictorRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve the index.html file for root URL
        if self.path == "/" or self.path == "/index.html":
            if os.path.exists(INDEX_PATH):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                with open(INDEX_PATH, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Frontend file index.html not found.")
        else:
            # Fallback to default handler for other files
            super().do_GET()

    def do_POST(self):
        if self.path == "/predict":
            if model is None:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                response = {"error": "Model pickle file not found on server. Train the model first."}
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return
            
            # Read content length
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            try:
                # Parse inputs
                data = json.loads(post_data.decode("utf-8"))
                
                # Assemble inputs into a pandas DataFrame matching train columns
                input_df = pd.DataFrame({
                    "longitude": [float(data.get("longitude", -122.23))],
                    "latitude": [float(data.get("latitude", 37.88))],
                    "housing_median_age": [float(data.get("housing_median_age", 41.0))],
                    "total_rooms": [float(data.get("total_rooms", 880.0))],
                    "total_bedrooms": [float(data.get("total_bedrooms", 129.0))],
                    "population": [float(data.get("population", 322.0))],
                    "households": [float(data.get("households", 126.0))],
                    "median_income": [float(data.get("median_income", 8.3252))],
                    "ocean_proximity": [str(data.get("ocean_proximity", "NEAR BAY"))]
                })
                
                # Predict
                prediction = model.predict(input_df)
                predicted_val = float(prediction[0])
                
                # Send JSON response
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                
                response = {
                    "status": "success",
                    "predicted_value": predicted_val
                }
                self.wfile.write(json.dumps(response).encode("utf-8"))
                
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                response = {
                    "status": "error",
                    "message": str(e)
                }
                self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found.")

def run(server_class=HTTPServer, handler_class=PredictorRequestHandler):
    server_address = ("", PORT)
    httpd = server_class(server_address, handler_class)
    print(f"\n=======================================================")
    print(f"California Housing Predictor UI Server Running!")
    print(f"URL: http://localhost:{PORT}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    run()
