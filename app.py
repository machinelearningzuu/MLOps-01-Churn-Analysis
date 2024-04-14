from flask_cors import CORS
from flask import Flask,request,jsonify
from src.pipeline.prediction_pipeline import PredictionPipeline

app=Flask(__name__)
# CORS(app)

pipe = PredictionPipeline()
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = pipe.initiate_prediction(data)
    return jsonify({'prediction':prediction})

if __name__ == '__main__':
    app.run(debug=True)