'''
#from flask import Flask, render_template

#------------FOR LOGIN
from flask import Flask, render_template, redirect, url_for, request


app = Flask(__name__)

@app.route('/')
def home():
	return "Hello, World!"

@app.route('/welcome')
def welcome():
	return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('home'))
    return render_template('login.html', error=error)
	
if __name__=='__main__':
	app.run(debug=True)
'''
import cv2
import os
import numpy as np
from flask import Flask, render_template, redirect, url_for, request

subjects = ["Rishabh","Sacha Baron Cohen","Manish"]

def detect_face(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
 
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

	if (len(faces) == 0):
		return None, None
 
	(x, y, w, h) = faces[0]

	return gray[y:y+w, x:x+h], faces[0]


def prepare_training_data(data_folder_path):

	data_folder_path = "training-data"
	dirs = os.listdir(data_folder_path)


	faces = []
	labels = []
 
	for dir_name in dirs:
		if not dir_name.startswith("s"):
			continue
	
		label = int(dir_name.replace("s", ""))

		subject_dir_path = data_folder_path + "/" + dir_name
 
		subject_images_names = os.listdir(subject_dir_path)

		for image_name in subject_images_names:
			if image_name.startswith("."):
				continue

			image_path = subject_dir_path + "/" + image_name

			image = cv2.imread(image_path)

			face, rect = detect_face(image)

			if face is not None:
				faces.append(face)
				labels.append(label)
 
	return faces, labels

faces, labels = prepare_training_data("training-data")

face_recognizer = cv2.face.LBPHFaceRecognizer_create() 

face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img):
	img = test_img.copy()
	face, rect = detect_face(img)
	
	if(face is None):
		return img,1000
	
	label,predicted_confidence= face_recognizer.predict(face)
	label_text = subjects[label]
	
	if(predicted_confidence<60):
		return img,label
 
	return img,1000

End_of_Video = False


app = Flask(__name__)

@app.route('/')
def home():
	return "Hello, World!"

@app.route('/welcome')
def welcome():
	return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
	error = "NO"
	render_template('login.html', error=error)
	if request.method == 'POST':
		cap = cv2.VideoCapture(0)
		while(1):
			ret, img = cap.read() 	
			predicted_img,guy = predict(img)
		
			if guy<3:
				print "WELCOME " + subjects[guy]
				End_of_Video = True
				return redirect(url_for('home'))
				break
	
			#cv2.imshow("VIDEO",predicted_img)
	
			k = cv2.waitKey(1)
			if k == 27:
				break
		
	return render_template('login.html', error=error)
	
if __name__=='__main__':
	app.run(debug=True)