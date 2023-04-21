from flask import Flask,request,render_template
from ml1 import svmmodel
import os
app=Flask(__name__)

cf_port = os.getenv("PORT")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit',methods=['POST','GET'])
def submit():
    description=str(request.form["input_text"]).lower()
    description=list(description.split(','))
    output=svmmodel(description)
    return render_template('result1.html',result=output,description=description,length=len(description))


if __name__ == '__main__':
	if cf_port is None:
		app.run(host='0.0.0.0', port=5000, debug=True)
	else:
		app.run(host='0.0.0.0', port=int(cf_port))