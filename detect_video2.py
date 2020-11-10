import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

from gtts import gTTS 
import os
import simpleaudio as sa
from playsound import playsound
from imutils import paths


flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/paris.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def waitf(duration):
    start = time.time()
    while True:
        if time.time()-start>duration:
            return 1
        if cv2.waitKey(1) == ord('q'):
            return 0

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def check_blur(image,threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    if fm<threshold:
        return 0
    return 1
                    



def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    fps = 0.0
    count = 0
    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        n=1

        if cv2.waitKey(1) == ord('a'):
            while True:
                _, img = vid.read()

                if (check_blur(img,threshold=200)==0):                    #ADJUST THRESHOLD HERE
                    cv2.imshow('output', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    print("blurred")
                    continue

                img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                img_in = tf.expand_dims(img_in, 0)
                img_in = transform_images(img_in, FLAGS.size)

                t1 = time.time()
                boxes, scores, classes, nums = yolo.predict(img_in)
                fps  = ( fps + (1./(time.time()-t1)) ) / 2

                img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                cv2.imshow('output', img)

                text=""
                for i in range(80):
                    if scores[0][i]==0:
                        break
                    Class=int(classes[0][i])
                    place=((boxes[0][i][2]-boxes[0][i][0])/2)+boxes[0][i][0]
                    print(place,class_names[Class])
                    if place<.33:
                        side='left'
                    elif place<.66:
                        side='center'
                    else:
                        side='right'
                    if side=='center':
                        text=text+" There is a "+class_names[Class]+'in the '+side+'.'
                    else:
                        text=text+" There is a "+class_names[Class]+'on the '+side+'.'

                
                try:
                    # text = "This is a test."
                    speech = gTTS(text = text, slow = False)
                    speech.save(r'C:\\Users\\HARINI\\Object-Detection-API\\audio\\text'+str(n)+'.wav')  #CHANGE THESE 2 PATHS TO YOUR OWN PATH
                    os.system(r'C:\\Users\\HARINI\\Object-Detection-API\\audio\\text'+str(n)+'.wav')
                    n=n+1
                except:
                    continue
                # wave_obj = sa.WaveObject.from_wave_file("text.wav")
                # play_obj = wave_obj.play()
                # play_obj.wait_done()
                # playsound('text.wav')
                
                # time.sleep(6)

                if not(waitf(9)):
                    break

                # if cv2.waitKey(1) == ord('q'):
                #     break

        # if cv2.waitKey(1) == ord('z'):
        #     t1 = time.time()
        #     img=cv2.imread(r'C:\\Users\\HARINI\\Object-Detection-API\\image.jpg')
        #     img = tf.expand_dims(img, 0)
        #     img = transform_images(img, FLAGS.size)
        #     boxes, scores, classes, nums = yolo.predict(img)
        #     fps  = ( fps + (1./(time.time()-t1)) ) / 2

        #     img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
            
        #     text=""
        #     for i in range(80):
        #         if scores[0][i]==0:
        #             break
        #         Class=int(classes[0][i])
        #         place=((boxes[0][i][2]-boxes[0][i][0])/2)+boxes[0][i][0]
        #         print(place,class_names[Class])
        #         if place<.33:
        #             side='left'
        #         elif place<.66:
        #             side='center'
        #         else:
        #             side='right'
        #         if side=='center':
        #             text=text+" There is a "+class_names[Class]+'in the '+side+'.'
        #         else:
        #             text=text+" There is a "+class_names[Class]+'on the '+side+'.'

        #     # text = "This is a test."
        #     speech = gTTS(text = text, slow = False) 
        #     speech.save("text.wav")
        #     os.system("text.wav")
        #     # wave_obj = sa.WaveObject.from_wave_file("text.wav")
        #     # play_obj = wave_obj.play()
        #     # play_obj.wait_done()
        #     # playsound('text.wav')
        #     while True:
        #         cv2.imshow('output', img)
        #         if cv2.waitKey(1) == ord('q'):
        #             break


        if FLAGS.output:
            out.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
