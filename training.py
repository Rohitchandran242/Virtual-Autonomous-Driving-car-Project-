print("Setting up")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import *
from sklearn.model_selection import train_test_split



#1
path = 'myData'
data = importDataInfo(path)

#2
data  = balanceData(data,display=False)

#3
imagesPath,steerings = loadData(path,data)
# print(imagesPath[0],steering[0])

#4
xTrain,xTest,yTrain,yTest = train_test_split(imagesPath,steerings,test_size=0.2,random_state=5)

# print("total training states",len(xTrain))
# print("total test states",len(xTest))

#5

#6

#7

#8
model = createModel()
model.summary()


#9
history = model.fit(batchGen(xTrain,yTrain,100,1),steps_per_epoch = 300,epochs=20,
	validation_data= batchGen(xTest,yTest,100,0),validation_steps=200)

#10

model.save('model.h5')
print("Model Saved")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

