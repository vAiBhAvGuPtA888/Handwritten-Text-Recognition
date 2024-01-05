#!/usr/bin/env python
# coding: utf-8

# In[12]:


import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text


# In[23]:


if __name__ == "__main__":
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("Models/03_handwriting_recognition/202301111911/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    # Provide the path to a single PNG image
    image_path = "cong.jpg"

    image = cv2.imread(image_path)

    prediction_text = model.predict(image)

    cer = get_cer(prediction_text, label)
    print(f"Image: {image_path}, Prediction: {prediction_text}, CER: {cer}")

    # Show the image and predicted word
    image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
    cv2.imshow("Image", image)
    print(f"Predicted Word: {prediction_text}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

