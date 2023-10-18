# Virtual-Autonomous-Driving-car-Project-


Autonomous Self-Driving Car using Machine Learning

# Introduction
This project focuses on the implementation of an autonomous self-driving car using advanced machine learning techniques, particularly Convolutional Neural Networks (CNNs). The system mimics the human brain's visual cortex to make driving decisions based on the input from onboard cameras.

Features
Real-time data processing from vehicle-mounted cameras
Advanced image recognition for immediate response to driving conditions
Data exploration and visualization using Python libraries

# Technologies Used

Python
Pandas
Matplotlib
Scikit-learn
Convolutional Neural Networks (CNNs)

# Dataset
The project uses a dataset comprising images from the vehicle's cameras and corresponding steering angles. This data undergoes extensive preprocessing and augmentation to train the machine learning model effectively.

#System Architecture
The system employs a CNN that takes in processed camera images and predicts the required steering angle to navigate the road. The CNN architecture is designed to automatically learn feature representation from the raw pixels of the images without manual feature extraction.

#Implementation
The CNN model is trained using a dataset of driving scenarios. The training process adjusts the model weights to minimize the difference between the predicted steering command and the actual human driver command. The system uses various Python libraries for image processing and model training.

#Usage
The trained model is deployed in a simulated environment where it receives real-time camera images, processes these images, and sends control commands (steering and throttle) to navigate the car autonomously.

#Environment Setup and Dependencies
This project is implemented using Python and requires certain libraries and modules. Follow the steps below to set up your environment and run the project.

#Prerequisites
Python 3.x
pip (Python package installer)

Dependencies
#Here is a list of Python libraries used in this project:

pandas: Data manipulation and analysis
numpy: Numerical computations
scikit-learn: Machine learning library
keras: Neural network library
opencv-python (cv2): Computer vision tasks, used for image processing
Pillow (PIL): Python Imaging Library
matplotlib: Plotting and visualization
Flask: Web framework, used for the server application
python-socketio: For real-time communication between client and server
eventlet: Concurrent networking library, used with Flask and Socket.IO

#Installation

#Clone the repository to your local machine:
git clone https://github.com/github.com/Rohitchandran242/autonomous-car-project.git

#Navigate to the project directory:
cd autonomous-car-project

#Create a virtual environment in the project directory:
python -m venv env

#Activate the virtual environment. On Windows, use:
.\env\Scripts\activate

#On Unix or MacOS, use:
source env/bin/activate

#Install the required Python libraries and dependencies:
pip install pandas numpy scikit-learn keras opencv-python-headless Pillow matplotlib Flask python-socketio eventlet

