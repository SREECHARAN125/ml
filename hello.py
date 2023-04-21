from flask import Flask,request,render_template,redirect,url_for
#from ml import svmmodel
from demo import svm
import os
app=Flask(__name__)

cf_port = os.getenv("PORT")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit',methods=['POST','GET'])
def submit():
    
    description=str(request.form["input_text"])
    description=list(description.split(','))
    
    if description==[""]:
        return redirect(url_for('nodescription'))
    else:
        output=svm(description)
        return render_template('result.html',result=output,description=description,length=len(description))


@app.route('/nodescription')
def nodescription():
    return render_template('index.html')

if __name__ == '__main__':
	if cf_port is None:
		app.run(host='0.0.0.0', port=7000, debug=True)
	else:
		app.run(host='0.0.0.0', port=int(cf_port), debug=True)