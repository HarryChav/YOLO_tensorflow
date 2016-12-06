#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import cv2
import time
import sys


class YOLO_TF:  # noqa
    fromfile = None
    tofile_img = 'test/output.jpg'
    tofile_txt = 'test/output.txt'
    imshow = True
    filewrite_img = False
    filewrite_txt = False
    disp_console = True
    weights_file = 'weights/YOLO_tiny.ckpt'
    alpha = 0.1
    threshold = 0.2
    iou_threshold = 0.5
    num_class = 20
    num_box = 2
    grid_size = 7
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    w_img = 640
    h_img = 480

    def __init__(self, argvs=[]):
        self.argv_parser(argvs)
        self.build_networks()
        if self.fromfile is not None:
            self.detect_from_file(self.fromfile)

    def argv_parser(self, argvs):
        for i in range(1, len(argvs), 2):
            if argvs[i] == '-fromfile':
                self.fromfile = argvs[i + 1]
            if argvs[i] == '-tofile_img':
                self.tofile_img = argvs[i + 1]
                self.filewrite_img = True
            if argvs[i] == '-tofile_txt':
                self.tofile_txt = argvs[i + 1]
                self.filewrite_txt = True
            if argvs[i] == '-imshow':
                if argvs[i + 1] == '1':
                    self.imshow = True
                else:
                    self.imshow = False
            if argvs[i] == '-disp_console':
                if argvs[i + 1] == '1':
                    self.disp_console = True
                else:
                    self.disp_console = False

    def build_networks(self):
        self.x = tf.placeholder('float32', [None, None, 3], name="input")
        self.input = tf.image.resize_bilinear(tf.expand_dims(self.x, 0), [448, 448]) / (255. / 2.) - 1.

        model_output = self._build_networks(self.input)[0, :]
        self._build_interpret(model_output)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)
        if self.disp_console:
            print "Loading complete!" + '\n'

    def _build_networks(self, input):
        if self.disp_console:
            print "Building YOLO_tiny graph..."
        # self.x = tf.placeholder('float32', [None, 448, 448, 3])
        self.conv_1 = self.conv_layer(1, input, 16, 3, 1)
        self.pool_2 = self.pooling_layer(2, self.conv_1, 2, 2)
        self.conv_3 = self.conv_layer(3, self.pool_2, 32, 3, 1)
        self.pool_4 = self.pooling_layer(4, self.conv_3, 2, 2)
        self.conv_5 = self.conv_layer(5, self.pool_4, 64, 3, 1)
        self.pool_6 = self.pooling_layer(6, self.conv_5, 2, 2)
        self.conv_7 = self.conv_layer(7, self.pool_6, 128, 3, 1)
        self.pool_8 = self.pooling_layer(8, self.conv_7, 2, 2)
        self.conv_9 = self.conv_layer(9, self.pool_8, 256, 3, 1)
        self.pool_10 = self.pooling_layer(10, self.conv_9, 2, 2)
        self.conv_11 = self.conv_layer(11, self.pool_10, 512, 3, 1)
        self.pool_12 = self.pooling_layer(12, self.conv_11, 2, 2)
        self.conv_13 = self.conv_layer(13, self.pool_12, 1024, 3, 1)
        self.conv_14 = self.conv_layer(14, self.conv_13, 1024, 3, 1)
        self.conv_15 = self.conv_layer(15, self.conv_14, 1024, 3, 1)
        self.fc_16 = self.fc_layer(16, self.conv_15, 256, flat=True, linear=False)
        self.fc_17 = self.fc_layer(17, self.fc_16, 4096, flat=False, linear=False)
        # skip dropout_18
        self.fc_19 = self.fc_layer(19, self.fc_17, 1470, flat=False, linear=True)
        return self.fc_19

    def _build_interpret(self, output):
        probs = tf.zeros((7, 7, 2, 20))
        class_probs = tf.reshape(output[0:980], (7, 7, 20))
        scales = tf.reshape(output[980:1078], (7, 7, 2))
        boxes = tf.reshape(output[1078:], (7, 7, 2, 4))
        offset = tf.constant(np.transpose(np.reshape(np.array([np.arange(7)] * 14), (2, 7, 7)), (1, 2, 0)), dtype=tf.float32)

        boxes = tf.transpose(tf.pack([
            (boxes[:, :, :, 0] + offset) / 7.,
            (boxes[:, :, :, 1] + tf.transpose(offset, (1, 0, 2))) / 7.,
            tf.mul(boxes[:, :, :, 2], boxes[:, :, :, 2]),
            tf.mul(boxes[:, :, :, 3], boxes[:, :, :, 3])
        ]), (1, 2, 3, 0))

        # boxes[:, :, :, 0] += offset
        # boxes[:, :, :, 1] += tf.transpose(offset, (1, 0, 2))
        # boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / 7.0
        # boxes[:, :, :, 2] = tf.mul(boxes[:, :, :, 2], boxes[:, :, :, 2])
        # boxes[:, :, :, 3] = tf.mul(boxes[:, :, :, 3], boxes[:, :, :, 3])

        # boxes[:, :, :, 0] *= self.w_img
        # boxes[:, :, :, 1] *= self.h_img
        # boxes[:, :, :, 2] *= self.w_img
        # boxes[:, :, :, 3] *= self.h_img
        probs_list = []
        for i in range(2):
            probs_list_i = []
            scale = scales[:, :, i]
            for j in range(20):
                probs_list_i.append(tf.mul(class_probs[:, :, j], scale))  # 20, 7, 7
            probs_list.append(tf.pack(probs_list_i))
        probs = tf.pack(probs_list)  # 2,20,7,7
        probs = tf.transpose(probs, [2, 3, 0, 1], name="probs")  # 7,7,2,20
        # return probs, boxes
        self.classes_prob = tf.reshape(tf.reduce_max(probs, 3), [-1], name="classes_prob")
        self.classes_arg = tf.reshape(tf.argmax(probs, 3), [-1], name="classes_arg")
        self.boxes = tf.reshape(boxes, [-1, 4], name="boxes")
        mask = self.classes_prob > self.threshold
        self.class_prob_filtered = tf.boolean_mask(self.classes_prob, mask, name="probs_filtered") + 0.
        self.classes_arg_filtered = tf.boolean_mask(self.classes_arg, mask, name="classes_num_filtered") + 0
        self.boxes_filtered = tf.boolean_mask(self.boxes, mask, name="boxes_filtered") + 0.

    def _interpret_output(self, probs_filtered, classes_num_filtered, boxes_filtered):
        boxes_filtered[:, 0] *= self.w_img
        boxes_filtered[:, 1] *= self.h_img
        boxes_filtered[:, 2] *= self.w_img
        boxes_filtered[:, 3] *= self.h_img
        # 过滤出置信度大于0.2的
        # filter_mat_probs = probs >= self.threshold
        # filter_mat_boxes = np.nonzero(filter_mat_probs)
        # boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        # probs_filtered = probs[filter_mat_probs]
        # classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        # 按置信度排序, 可以不要
        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]
        # 排除包含关系的
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]
        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])
        return result

    def conv_layer(self, idx, inputs, filters, size, stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal([size, size, int(channels), filters], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))

        pad_size = size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        inputs_pad = tf.pad(inputs, pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID', name=str(idx) + '_conv')
        conv_biased = tf.add(conv, biases, name=str(idx) + '_conv_biased')
        if self.disp_console:
            print '    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx, size, size, stride, filters, int(channels))
        return tf.maximum(self.alpha * conv_biased, conv_biased, name=str(idx) + '_leaky_relu')

    def pooling_layer(self, idx, inputs, size, stride):
        if self.disp_console:
            print '    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx, size, size, stride)
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME', name=str(idx) + '_pool')

    def fc_layer(self, idx, inputs, hiddens, flat=False, linear=False):
        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
            inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs
        weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))
        if self.disp_console:
            print '    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (idx, hiddens, int(dim), int(flat), 1 - int(linear))
        if linear:
            return tf.add(tf.matmul(inputs_processed, weight), biases, name=str(idx) + '_fc')
        ip = tf.add(tf.matmul(inputs_processed, weight), biases)
        return tf.maximum(self.alpha * ip, ip, name=str(idx) + '_fc')

    def detect_from_cvmat(self, img):
        s = time.time()
        self.h_img, self.w_img, _ = img.shape
        # img_resized = cv2.resize(img, (448, 448))
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # noqa
        # img_resized_np = np.asarray(img_RGB)
        # inputs = np.zeros((1, 448, 448, 3), dtype='float32')
        # inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
        in_dict = {self.x: img_RGB}
        # net_output = self.sess.run([self.probs, self.boxes], feed_dict=in_dict)
        # self.result = self.interpret_output(net_output[0])
        class_prob_filtered, classes_arg_filtered, boxes_filtered = self.sess.run([self.class_prob_filtered, self.classes_arg_filtered, self.boxes_filtered], feed_dict=in_dict)
        # import ipdb; ipdb.set_trace()
        self.result = self._interpret_output(class_prob_filtered, classes_arg_filtered, boxes_filtered)
        self.show_results(img, self.result)
        strtime = str(time.time() - s)
        if self.disp_console:
            print 'Elapsed time : ' + strtime + ' secs' + '\n'

    def detect_from_file(self, filename):
        if self.disp_console:
            print 'Detect from ' + filename
        img = cv2.imread(filename)
        # img = misc.imread(filename)
        self.detect_from_cvmat(img)

    def detect_from_crop_sample(self):
        self.w_img = 640
        self.h_img = 420
        f = np.array(open('person_crop.txt', 'r').readlines(), dtype='float32')
        inputs = np.zeros((1, 448, 448, 3), dtype='float32')
        for c in range(3):
            for y in range(448):
                for x in range(448):
                    inputs[0, y, x, c] = f[c * 448 * 448 + y * 448 + x]

        in_dict = {self.x: inputs}
        net_output = self.sess.run(self.fc_19, feed_dict=in_dict)
        self.boxes, self.probs = self.interpret_output(net_output[0])
        img = cv2.imread('person.jpg')
        self.show_results(self.boxes, img)

    def interpret_output(self, output):
        probs = np.zeros((7, 7, 2, 20))
        class_probs = np.reshape(output[0:980], (7, 7, 20))
        scales = np.reshape(output[980:1078], (7, 7, 2))
        boxes = np.reshape(output[1078:], (7, 7, 2, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)] * 14), (2, 7, 7)), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / 7.0
        boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
        boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])

        boxes[:, :, :, 0] *= self.w_img
        boxes[:, :, :, 1] *= self.h_img
        boxes[:, :, :, 2] *= self.w_img
        boxes[:, :, :, 3] *= self.h_img

        for i in range(2):
            for j in range(20):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        return result

    def show_results(self, img, results):
        img_cp = img.copy()
        if self.filewrite_txt:
            ftxt = open(self.tofile_txt, 'w')
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3]) // 2
            h = int(results[i][4]) // 2
            if self.disp_console:
                print '    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + str(int(results[i][3])) + ',' + str(int(results[i][4])) + '], Confidence = ' + str(results[i][5])
            if self.filewrite_img or self.imshow:
                cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
                cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if self.filewrite_txt:
                ftxt.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h) + ',' + str(results[i][5]) + '\n')
        if self.filewrite_img:
            if self.disp_console:
                print '    image file writed : ' + self.tofile_img
            cv2.imwrite(self.tofile_img, img_cp)
        if self.imshow:
            cv2.imshow('YOLO_tiny detection', img_cp)
            cv2.waitKey(1)
        if self.filewrite_txt:
            if self.disp_console:
                print '    txt file writed : ' + self.tofile_txt
            ftxt.close()

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def training(self):  # TODO add training function!
        return None


def main(argvs):
    YOLO_TF(argvs)
    cv2.waitKey(1000)


if __name__ == '__main__':
    main(sys.argv)
