from keras.models import load_model
import sys
import numpy as np
from keras.preprocessing import image

model_path = sys.argv[1]

image_path = sys.argv[2]

labels = {0: 'ANGER', 1: 'DISGUST', 2: 'FEAR', 3: 'HAPPINESS', 
			4: 'NEUTRAL', 5: 'SADNESS', 6: 'SURPRISE'}

test_image = image.load_img(image_path, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

model = load_model(model_path)

preds = model.predict_classes(test_image)[0]

print(labels[preds])