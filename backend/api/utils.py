import datetime
import os

import cv2
import numpy as np
import tensorflow.compat.v1 as tf


# Model File Path #

'''
refer from https://github.com/songhengyang/face_landmark_factory.git

Solved problems
1. tensorflow v1 compatibility problem
2. cv2, keras installation
3. invalid asset path(relative path)
'''


class MarkDetector:
    """Facial landmark detector by Convolutional Neural Network"""

    def __init__(self, mark_model, cnn_input_size):
        """Initialization"""
        # A face detector is required for mark detection.
        self.face_detector = FaceDetector()
        self.marks = None

        if mark_model.split(".")[-1] == "pb":
            # Get a TensorFlow session ready to do landmark detection
            # Load a (frozen) Tensorflow model into memory.
            self.cnn_input_size = cnn_input_size
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()

                with tf.gfile.GFile(mark_model, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            self.graph = detection_graph
            self.sess = tf.Session(graph=detection_graph)


        else:
            self.cnn_input_size = cnn_input_size
            # with CustomObjectScope({'tf': tf}):
            with tf.keras.utils.custom_object_scope({'smoothL1': smoothL1,
                                                     'relu6': relu6,
                                                     'DepthwiseConv2D': DepthwiseConv2D,
                                                     'mask_weights': mask_weights,
                                                     'tf': tf}):
                self.sess = tf.keras.models.load_model(mark_model)

    def detect_marks_tensor(self, image_np, input_name, output_name):
        # Actual detection.
        predictions = self.sess.run(
            output_name,
            feed_dict={input_name: image_np})

        # Convert predictions to landmarks
        marks = np.array(predictions).flatten()
        marks = np.reshape(marks, (-1, 2))

        return marks

    def detect_marks_keras(self, image_np):
        """Detect marks from image"""
        predictions = self.sess.predict_on_batch(image_np)

        # Convert predictions to landmarks.
        marks = np.array(predictions).flatten()
        marks = np.reshape(marks, (-1, 2))

        return marks

    @staticmethod
    def draw_box(image, boxes, box_color=(255, 255, 255)):
        """Draw square boxes on image"""
        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]), box_color)

    @staticmethod
    def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def get_square_box(box):
        """Get a square box out of the given box, by expanding it."""
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:  # Already a square.
            return box
        elif diff > 0:  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:  # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):
        """Check if the box is in image"""
        rows = image.shape[0]
        cols = image.shape[1]
        return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

    def extract_cnn_facebox(self, image):
        """Extract face area from image."""
        _, raw_boxes = self.face_detector.get_faceboxes(image=image, threshold=0.5)
        faceboxes = []
        for box in raw_boxes:
            # Move box down.
            diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
            offset_y = int(abs(diff_height_width / 2))
            box_moved = self.move_box(box, [0, offset_y])
            # Make box square.
            facebox = self.get_square_box(box_moved)

            if self.box_in_image(facebox, image):
                faceboxes.append(facebox)
        return faceboxes

    @staticmethod
    def draw_marks(image, marks, color=(255, 255, 255), thick=1):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), thick, color, -1, cv2.LINE_AA)


class FaceDetector:
    """Detect human face from image"""

    def __init__(self,
                 dnn_proto_text='./api/assets/deploy.prototxt',
                 dnn_model='./api/assets/res10_300x300_ssd_iter_140000.caffemodel'):
        """Initialization"""
        self.face_net = cv2.dnn.readNetFromCaffe(dnn_proto_text, dnn_model)
        self.detection_result = None

    def get_faceboxes(self, image, threshold=0.5):
        """
        Get the bounding box of faces in image using dnn.
        """
        rows, cols, _ = image.shape

        confidences = []
        faceboxes = []
        # cv2.dnn.blobFromImage() ret 4-dim array
        self.face_net.setInput(cv2.dnn.blobFromImage(
            image, 1.0, (rows, cols), (104.0, 177.0, 123.0), False, False))
        detections = self.face_net.forward()

        for result in detections[0, 0, :, :]:
            confidence = result[2]
            if confidence > threshold:
                x_left_bottom = int(result[3] * cols)
                y_left_bottom = int(result[4] * rows)
                x_right_top = int(result[5] * cols)
                y_right_top = int(result[6] * rows)
                confidences.append(confidence)
                faceboxes.append([x_left_bottom, y_left_bottom, x_right_top, y_right_top])

        self.detection_result = [faceboxes, confidences]

        return confidences, faceboxes

    def draw_all_result(self, image):
        """Draw the detection result on image"""
        for facebox, conf in self.detection_result:
            cv2.rectangle(image, (facebox[0], facebox[1]),
                          (facebox[2], facebox[3]), (0, 255, 0))
            label = "face: %.4f" % conf
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(image, (facebox[0], facebox[1] - label_size[1]),
                          (facebox[0] + label_size[0],
                           facebox[1] + base_line),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(image, label, (facebox[0], facebox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))



class FaceMarker:
    CURRENT_MODEL = './api/facial_models/facial_landmark_SqueezeNet.pb'
    CNN_INPUT_SIZE = 64

    def __init__(self, base_dir='/tmp/landmark'):
        self.mark_detector = MarkDetector(self.CURRENT_MODEL, self.CNN_INPUT_SIZE)
        self.base_dir = base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def save_temp(self, file):
        tokens = file.name.split('.')
        if len(tokens) < 2:
            return None
        ext = tokens[len(tokens) - 1]
        filename = f'{self.base_dir}/{datetime.datetime.now().microsecond}.{ext}'
        with open(filename, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
            return filename

        return None

    def save_image_from_cv2(self, frame, filename):
        dir = os.path.dirname(filename)
        if not os.path.exists(dir):
            os.makedirs(dir)

        return cv2.imwrite(filename, frame)

    def detect_face(self, frame):
        faceboxes = self.mark_detector.extract_cnn_facebox(frame)
        for facebox in faceboxes:
            face_img = frame[facebox[1]: facebox[3], facebox[0]: facebox[2]] # ltrb
            cv2.rectangle(frame, (facebox[0], facebox[1]), (facebox[2], facebox[3]), (0, 255, 0), 2)
            face_img = cv2.resize(face_img, (self.CNN_INPUT_SIZE, self.CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img0 = face_img.reshape(1, self.CNN_INPUT_SIZE, self.CNN_INPUT_SIZE, 1)

            marks = self.mark_detector.detect_marks_tensor(face_img0, 'input_2:0', 'output/BiasAdd:0')
            marks *= facebox[2] - facebox[0]
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]
            self.mark_detector.draw_marks(frame, marks, color=(255, 255, 255), thick=2)