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



## must need if model is not for tensorflow

from keras import backend as K
from keras.layers import Conv2D
from keras import initializers, regularizers, constraints
from keras.utils import conv_utils
from keras.engine import InputSpec

# import numpy as np

# Our Own Loss Function
HUBER_DELTA = 0.5


def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return K.sum(x)


weights = np.empty((136,))  # Outer: Brows: Nose: Eyes: Mouth (0.5 : 1 : 2 : 3 : 1)

weights[0:33] = 3
weights[33:53] = 1
weights[53:71] = 2
weights[71:95] = 2
weights[95:] = 1
def mask_weights(y_true, y_pred):
    x = K.abs(y_true - y_pred) * weights
    return K.sum(x)


def relu6(x):
    return K.relu(x, max_value=6)


class DepthwiseConv2D(Conv2D):
    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0], self.kernel_size[1],
                                  input_dim, self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(input_dim * self.depth_multiplier,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.bias:
            outputs = K.bias_add(
                outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding, self.strides[1])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(
            self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(
            self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(
            self.depthwise_constraint)
        return config
