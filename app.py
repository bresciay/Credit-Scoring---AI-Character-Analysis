from flask import Flask, request, jsonify
import pickle
import numpy as np
import sklearn

app = Flask(__name__)

# load the model
model = pickle.load(open('model\RFC_new.pkl','rb')) 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            params = request.json
            input_data = [x for x in params.values()]
            print(input_data)

            final_features = np.array(input_data).reshape(1, -1)
            print(final_features)

            result = model.predict(final_features)[0]

            # criteria
            if result < 34:
                criteria = 'low'
            elif result >= 34 and result < 67:
                criteria = 'medium'
            else:
                criteria = 'high'

            return jsonify({
                'status': True,
                'message': 'success',
                'input': params,
                'result': result,
                'criteria' : criteria
            })
        
    except Exception as e:
        return jsonify({
                'status': False,
                'message': str(e)
            })

if __name__ =='__main__':
    app.run(port=5000, debug=True)