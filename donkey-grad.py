from keras.preprocessing import image
from keras.models import load_model
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import os

modelpath = 'alanwells_jun17.hdf5'

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def norm_img(x):
    img = np.float32(x)
    return (img - img.mean() / np.std(img))/255.0

def load_image(img_path):
    img = image.load_img(img_path, target_size=(120, 160))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='conv2d_5'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = load_model(modelpath)
    return new_model

def grad_cam(input_model, image, category_index, layer_name):
    model = input_model
    nb_classes = 15

    loss = model.output[0][:, category_index]
    conv_output = model.get_layer(layer_name).output
    #grads = normalize(K.gradients(loss, conv_output)[0])
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.input, K.learning_phase()], [conv_output, grads])

    output, grads_val = gradient_function([image, 1])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (160, 120))
    cam = np.maximum(cam, 1)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    #image += 0.5
    #image *= 255
    #image -= np.min(image)
    #image = np.minimum(image, 255)

    cam = mix_heatmap(image, heatmap)
    return cam, heatmap


def mix_heatmap(img, heatmap):
    if len(img.shape) > 3:
        img = img[0]
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(img)
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)
    return cam


model = load_model(modelpath)

if len(sys.argv) < 2:
    print('FATAL: Must specify the input directory')
    sys.exit()

IMGDIR = sys.argv[1]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('output.mp4', fourcc, 20., (160, 120))

nfiles = len(os.listdir(IMGDIR))
for ifile, fname in enumerate(os.listdir(IMGDIR)):
    path = os.path.join(IMGDIR, fname)
    if '.jpg' not in path: continue
    original_input = load_image(path)
    preprocessed_input = norm_img(original_input)
    predictions = model.predict(preprocessed_input)
    predicted_class = np.argmax(predictions[0][0])
    cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, 'conv2d_5')
    if (ifile % 100 == 0): print('%.2f' % (ifile*100./nfiles))
    img = np.uint8( original_input[0] )
    img = mix_heatmap(original_input, heatmap)
    #cv2.imwrite("gradcam.jpg", img)
    #video.write(original_input[0])
    video.write(img)
    if (ifile > 300): break

video.release()
