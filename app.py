from flask import Flask, jsonify, request
import pandas as pd
import pickle

file_name = "titanic_survival_prediction.pkl"

# save model
# pickle.dump(best_model, open(file_name, "wb"))

# load model
loaded_model = pickle.load(open(file_name,'rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)
    print(data,"------")

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = loaded_model.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == "__main__":
    app.run(debug=True,port=5000)

