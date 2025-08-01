import os
import re
import pickle
import numpy as np
from tqdm import tqdm

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model


BASE_DIR = os.path.join(".", "kaggle", "input", "flickr8k")
WORKING_DIR = os.path.join(".", "kaggle", "working")

# Load VGG16 model and remove the final classification layer
# model = VGG16()
# model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# print(model.summary())

# Extract features from images
features = {}
# directory = os.path.join(BASE_DIR, 'Images')

# for img_name in tqdm(os.listdir(directory)):
#     img_path = os.path.join(directory, img_name)

#     try:
#         # Load the image from file
#         image = load_img(img_path, target_size=(224, 224))
#         # Convert PIL image to numpy array
#         image = img_to_array(image)
#         # Reshape data for model
#         image = image.reshape(
#             (1, image.shape[0], image.shape[1], image.shape[2]))
#         # Preprocess image for VGG
#         image = preprocess_input(image)
#         # Extract features
#         feature = model.predict(image, verbose=0)
#         # Get image ID
#         image_id = img_name.split('.')[0]
#         # Store feature
#         features[image_id] = feature

#     except Exception as e:
#         print(f"Error processing {img_name}: {str(e)}")
#         continue

# # Store features in pickle
# features_path = os.path.join(WORKING_DIR, 'features.pkl')
# pickle.dump(features, open(features_path, 'wb'))
# print(f"Features saved to: {features_path}")

# print(features)

# load features from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)


with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()

mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)


def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            # Use regex to remove non-alphabetic characters
            caption = re.sub(r'[^A-Za-z\s]', '', caption)
            # Use regex to replace multiple spaces with single space
            caption = re.sub(r'\s+', ' ', caption)
            caption = 'startseq ' + \
                ' '.join([word for word in caption.split()
                         if len(word) > 1]) + ' endseq'
            captions[i] = caption


# print(mapping['1000268201_693b08cb0e'])
clean(mapping)
# print(mapping['1000268201_693b08cb0e'])

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

# print(len(all_captions))
# print(all_captions[:10])

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

# print(vocab_size)

# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
# print(max_length)

# image_ids = list(mapping.keys())
# split = int(len(image_ids) * 0.90)
# train = image_ids[:split]
# test = image_ids[split:]

# create data generator to get data in batch (avoids session crash)


# def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
#     # loop over images
#     X1, X2, y = list(), list(), list()
#     n = 0
#     while 1:
#         for key in data_keys:
#             n += 1
#             captions = mapping[key]
#             # process each caption
#             for caption in captions:
#                 # encode the sequence
#                 seq = tokenizer.texts_to_sequences([caption])[0]
#                 # split the sequence into X, y pairs
#                 for i in range(1, len(seq)):
#                     # split into input and output pairs
#                     in_seq, out_seq = seq[:i], seq[i]
#                     # pad input sequence
#                     in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
#                     # encode output sequence
#                     out_seq = to_categorical(
#                         [out_seq], num_classes=vocab_size)[0]
#                     # store the sequences
#                     X1.append(features[key][0])
#                     X2.append(in_seq)
#                     y.append(out_seq)
#             if n == batch_size:
#                 X1, X2, y = np.array(X1), np.array(X2), np.array(y)
#                 yield (X1, X2), y
#                 X1, X2, y = list(), list(), list()
#                 n = 0


# # encoder model
# # image feature layers
# inputs1 = Input(shape=(4096,))
# fe1 = Dropout(0.4)(inputs1)
# fe2 = Dense(256, activation='relu')(fe1)
# # sequence feature layers
# inputs2 = Input(shape=(max_length,))
# se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
# se2 = Dropout(0.4)(se1)
# se3 = LSTM(256)(se2)

# # decoder model
# decoder1 = add([fe2, se3])
# decoder2 = Dense(256, activation='relu')(decoder1)
# outputs = Dense(vocab_size, activation='softmax')(decoder2)

# model = Model(inputs=[inputs1, inputs2], outputs=outputs)
# model.compile(loss='categorical_crossentropy', optimizer='adam')

# # plot the model
# # plot_model(model, show_shapes=True)


# # train the model
# epochs = 15
# batch_size = 64
# steps = len(train) // batch_size

# for i in range(epochs):
#     # create data generator
#     generator = data_generator(
#         train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
#     # fit for one epoch
#     model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

# # save the model
# model.save(os.path.join(WORKING_DIR, 'best_model.h5'))


# Recreate the model architecture
def create_model(vocab_size, max_length):
    # encoder model
    # image feature layers
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence feature layers
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)

    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# Try to load weights only
try:
    model = create_model(vocab_size, max_length)
    model.load_weights(os.path.join(WORKING_DIR, 'best_model.h5'))
    print("Model weights loaded successfully!")
except Exception as e:
    print(f"Error loading weights: {e}")


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate caption for an image


def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += ' ' + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text


# validate with test data
# actual, predicted = list(), list()

# for key in tqdm(test):
#     # get actual caption
#     captions = mapping[key]
#     # predict the caption for image
#     y_pred = predict_caption(model, features[key], tokenizer, max_length)
#     # split into words
#     actual_captions = [caption.split() for caption in captions]
#     y_pred = y_pred.split()
#     # append to the list
#     actual.append(actual_captions)
#     predicted.append(y_pred)

# # calculate BLEU score
# print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
# print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

# Second code snippet - Image Caption Generation


def generate_caption(image_name):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('---------------------Predicted---------------------')
    print(y_pred)
    # plt.imshow(image)
    # plt.axis('off')  # Optional: remove axes for cleaner display
    # plt.title(f"Image: {image_name}")  # Optional: add title
    # plt.show()


generate_caption('667626_18933d713e.jpg')
generate_caption('1000268201_693b08cb0e.jpg')
generate_caption('1002674143_1b742ab4b8.jpg')
generate_caption('1007320043_627395c3d8.jpg')
generate_caption('1057210460_09c6f4c6c1.jpg')


print('success')
