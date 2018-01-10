from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import numpy as np

model = VGG16(weights='imagenet')  # 224×224
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))  # 224×224
x = image.img_to_array(img)     # (3, 224, 224)
x = np.expand_dims(x, axis=0)  # (1, 3, 224, 224)
x = preprocess_input(x)  
y_pred = model.predict(x) 
print(decode_predictions(y_pred))  


# according to:
#http://blog.csdn.net/xuhaijiao99/article/details/55803060
#http://keras-cn.readthedocs.io/en/latest/other/application/
