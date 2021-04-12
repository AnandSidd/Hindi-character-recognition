import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from image_processing import final_processing

model1 = load_model("model1.h5")

mapping={
  1:"क", 2:"ख", 3:"ग", 4:"घ", 5:"ड",
  6:"च", 7:"छ", 8:"ज", 9:"झ", 10:"ञ",
  11:"ट", 12:"ठ", 13:"ड", 14:"ढ", 15:"ण",
  16:"त", 17:"थ", 18:"द", 19:"ध", 20:"न",
  21:"प", 22:"फ", 23:"ब", 24:"भ", 25:"म",
  26:"य", 27:"र", 28:"ल", 29:"व", 30:"श", 31:"ष",
  32:"स", 33:"ह",34:"क्ष", 35:"त्र", 36:"ज्ञ",
  37:"अ", 38:"आ", 39:"इ", 40:"ई", 41:"उ", 42:"ऊ", 43:"ऋ", 44:"ए", 45:"ऐ", 46:"ओ", 47:"औ",
  48:"अं " , 49:"अ:"}


def predict(img):

    characters = final_processing(img)
    answer=[]
    for c in characters:
      # cv2.rectangle(c, (5,0), (c.shape[0]-5, 8), (0,0,0), -1)
      c = cv2.copyMakeBorder(c, 0, 0, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
      c=cv2.bitwise_not(c)
      rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
      c= cv2.dilate(c, rect_kernel,iterations=1)

      c = cv2.resize(c,(32,32),interpolation=cv2.INTER_AREA)
      c = c.reshape((-1,32,32,1))
      predicted = model1.predict(c)
      predicted = np.argmax(predicted, axis = 1)
      answer.append(mapping[predicted[0]])


    return answer


def test():
    image_paths = ['./Images/image1.jpeg','./Images/image2.jpeg','./Images/image3.jpeg']
    # correct_answers = [list1,list2,list3]
    score = 0
    multiplication_factor=2 #depends on character set size

    for i,image_path in enumerate(image_paths):
        image = cv2.imread(image_path) # This input format wont change
        answer = predict(image) # a list is expected
        print(''.join(answer))# will be the output string

    #     if correct_answers[i] == answer:
    #         score += len(answer)*multiplication_factor
    
    # print('The final score of the participant is',score)


if __name__ == "__main__":
    test()
