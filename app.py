import yaml
from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import ProcessUserData, Prediction

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

round_digit = config['prediction']['round_digits']
port = config['app']['port']
host = config['app']['host']

application = Flask(__name__)
app = application

# Route for the home page and prediction form
@app.route('/', methods=['GET', 'POST'], endpoint='predict_user_data')
def predict_user_data():
    # Check if it's a POST request (form submission)
    if request.method == 'POST':
        # Process user data from the form
        user_data = ProcessUserData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        
        # Get data frame for prediction
        df_user_input = user_data.get_data_frame()
        
        # Get prediction
        prediction_obj = Prediction()
        results = prediction_obj.get_prediction(df_user_input)
        rounded_results = round(results[0], round_digit)
    
        
        # Render the result in the home page
        return render_template('home.html', 
                               results=rounded_results)
    
    # If GET request, show the form
    return render_template('home.html', results=None)

if __name__ == "__main__":
    app.run(host=host, debug=False, port=port)