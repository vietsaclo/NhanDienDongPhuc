import time
import cv2
import argparse
import numpy as np
import os

weightName = './Files/yolov3.weights'
configName = './Files/yolov3.cfg'
className = './Files/yolov3.txt'

# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True,
#                 help='path to input image')
# ap.add_argument('-c', '--config', required=True,
#                 help='path to yolo config file')
# ap.add_argument('-w', '--weights', required=True,
#                 help='path to yolo pre-trained weights')
# ap.add_argument('-cl', '--classes', required=True,
#                 help='path to text file containing class names')
# args = ap.parse_args()

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

classes = None

with open(className, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(weightName, configName)

'''
  Hàm nhận diện đối tượng
  @param: sourceImage là nguồn hình ảnh, có thể là một đường dẫn hình hoặc một hình
  đã được đọc lên bằng OpenCV
  @param: classesName là phân lớp cần lưu trữ, classesName= 0 tức là người.
  chi tiết từng classes xem tại File yolov3.txt

  @return: một danh sách các hình ảnh con, là các hình ảnh người đã được nhận dạng.
'''
def fun_DetectObject(sourceImage, classesName= 0, isShowDetectionFull: bool= False):
  image = None
  width = None
  height = None
  scale = 0.00392
  if type(sourceImage) is str:
    try:
      image = cv2.imread(sourceImage)
    except:
      print('Path sourceImage non valid!')
      return
  else:
    image = sourceImage
  
  try:
    width = image.shape[1]
    height = image.shape[0]
  except:
    print('sourceIamge non valid!')
    return
  
  blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

  net.setInput(blob)

  outs = net.forward(get_output_layers(net))

  class_ids = []
  confidences = []
  boxes = []
  conf_threshold = 0.5
  nms_threshold = 0.4

  for out in outs:
      for detection in out:
          scores = detection[5:]
          class_id = np.argmax(scores)
          confidence = scores[class_id]
          if confidence > 0.5:
              center_x = int(detection[0] * width)
              center_y = int(detection[1] * height)
              w = int(detection[2] * width)
              h = int(detection[3] * height)
              x = center_x - w / 2
              y = center_y - h / 2
              class_ids.append(class_id)
              confidences.append(float(confidence))
              boxes.append([x, y, w, h])

  indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

  index = 0
  imgOriganal = image.copy()
  imgsGet = []
  for i in indices:
      i = i[0]
      box = boxes[i]
      x = box[0]
      y = box[1]
      w = box[2]
      h = box[3]
      draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
      if class_ids[i] == classesName:
          y = int(y)
          yh = int(y + h)
          x = int(x)
          xw = int(x + w)
          img = imgOriganal[y:yh, x:xw]
          imgsGet.append([img, [y, yh, x, xw]])
      index+=1

  if isShowDetectionFull:
    cv2.imshow('ff' ,image)
    cv2.waitKey()
  print('Len '+ classes[classesName] + ' Detection: ' + str(len(imgsGet)))
  return imgsGet

if __name__ == '__main__':
    count_id = 109

    index = 0
    DIR_ = 'D:/imgs/OutCongNhan'
    fileNames = os.listdir(DIR_)
    max_ = len(fileNames)
    for i in range(max_):
      fileName = DIR_ + '/' + fileNames[i]
      imageGet = fun_DetectObject(sourceImage= fileName)
      for I in range(len(imageGet)):
          imgResize = cv2.resize(imageGet[I][0], (200, 500))
          # cv2.imwrite(DIR_ + '/' + 'out_th/th_' + str(count_id) + '.jpg', imgResize)
          # count_id += 1
          cv2.imshow('f', imgResize)
          cv2.waitKey()
      print('done: {0}/{1}'.format(i+1, max_))
