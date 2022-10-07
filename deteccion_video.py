# import dependencies
from IPython.display import display, Image
# from google.colab.output import eval_js
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
# import html
import time
import matplotlib.pyplot as plt


# import darknet functions to perform object detections
from darknet.darknet import *

# load in our YOLOv4 architecture network
cfg_path = "./yolov4-tiny-custom.cfg"
obj_path = "./darknet/data/obj.data"
weights_path = "./training/yolov4-tiny-custom_best.weights"
network, class_names, class_colors = load_network(cfg_path, obj_path, weights_path)
width = network_width(network)
height = network_height(network)


# # darknet helper function to run detection on image
def darknet_helper(img, width, height):
  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

#   # get image ratios to convert bounding boxes to proper size
  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  # run model on darknet style image to get detections
  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  return detections, width_ratio, height_ratio

# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
def bbox_to_bytes(bbox_array):
  """
  Params:
          bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
  Returns:
        bytes: Base64 image byte string
  """
  # convert array into PIL image
  bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
  iobuf = io.BytesIO()
  # format bbox into png for return
  bbox_PIL.save(iobuf, format='png')
  # format return string
  bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

  return bbox_bytes  



RUTA_VIDEO = "video_bolas_prueba_clahe.avi"
RUTA_RESULTADOS = "./resultados_deteccion/"

vidcap = cv2.VideoCapture(RUTA_VIDEO)
count = 0
print("Iniciando la predicción en video")
tiempos = []
t_inicio_video = time.time()
t1 = time.time()
while True:
    # cv2.imwrite("frame%d.jpg" % count, image_)     # save frame as JPEG file      
    success_, image_ = vidcap.read()

    if not success_:
        break

    data_image = image_
    predict_image = darknet_helper(data_image, 416, 416)

    # loop through detections and draw them on transparent overlay image
    height_ratio = predict_image[2]
    width_ratio = predict_image[1]

    # create tra3nsparent overlay for bounding box
    # bbox_array = np.zeros([416,416,4], dtype=np.uint8)
    for label, confidence, bbox in predict_image[0]:
        left, top, right, bottom = bbox2points(bbox)
        left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
        bbox_array = cv2.rectangle(data_image, (left, top), (right, bottom), class_colors[label], 2)
        bbox_array = cv2.putText(data_image, "{} [{:.2f}]".format(label, float(confidence)),
                            (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            class_colors[label], 2)

    tiempos.append((time.time()-t1))

    cv2.imwrite(f"{RUTA_RESULTADOS}/predict_ball_{count}.jpg", data_image)
    # cv2_imshow(data_image)
    # plt.show()
    count += 1
    t1 = time.time()

print("Fin de la predicción en video, duración:", (time.time()-t_inicio_video))
print(f"Procesadas {count} imagenes")
print(f"Tiempos: {tiempos} \nPromedio (se omite el primer dato): {np.mean(tiempos[1:])}")
