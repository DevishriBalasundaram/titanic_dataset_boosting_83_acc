# from flask import Flask, jsonify, request
# import pandas as pd
# import pickle

# file_name = "titanic_survival_prediction.pkl"

# # save model
# # pickle.dump(best_model, open(file_name, "wb"))

# # load model
# loaded_model = pickle.load(open(file_name,'rb'))

# # app
# app = Flask(__name__)

# # routes
# @app.route('/', methods=['POST'])

# def predict():
#     # get data
#     data = request.get_json(force=True)
#     print(data,"------")

#     # convert data into dataframe
#     data.update((x, [y]) for x, y in data.items())
#     data_df = pd.DataFrame.from_dict(data)

#     # predictions
#     result = loaded_model.predict(data_df)

#     # send back to browser
#     output = {'results': int(result[0])}

#     # return data
#     return jsonify(results=output)

# if __name__ == "__main__":
#     app.run(debug=True,port=5000)



from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# app
app = Flask(__name__)

# routes
@app.route("/")
def home():
    return render_template('index.html')


@app.route("/result", methods=['GET', 'POST'])
def survival_prediction():
    if request.method == 'POST':
        print(request.form)
        Pclass = request.form['Pclass']
        Sex = request.form['Sex']
        Age = request.form['Age']
        SibSp = int(request.form['SibSp'])
        Parch = int(request.form['Parch'])
        Fare = request.form['Fare']
        Cabin = request.form['Cabin']
        Embarked = request.form['Embarked']
        Title = request.form['Title']
        
        feature_values = [Pclass,Sex,Age,SibSp,Parch,Fare,Cabin,Embarked,Title]
        array_args = np.array(feature_values)
        feature_values_arr = array_args.reshape(1, -1)
        model = open("titanic_survival_prediction.pkl","rb")
        model = joblib.load(model)
        model_prediction = model.predict(feature_values_arr)
        model_prediction = round(float(model_prediction), 2)
        if model_prediction == 0.0:
            return render_template('predict.html', prediction = "Person was not survived")
        else:
            return render_template('predict.html', prediction = "Person was survived")
            

    return render_template('predict.html', prediction = model_prediction)


if __name__ == "__main__":
    app.run(debug=True) 