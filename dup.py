from flask_cors import CORS
from flask import Flask,request,jsonify

app=Flask(__name__)
# CORS(app)

print("YESSS")
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("###YESSS")
    return jsonify({'prediction':'yes'})

if __name__ == '__main__':
    app.run(debug=True)