import tensorflow as tf
import os
import json
import pandas as pd
import re
import numpy as np
import random
import collections
import pickle
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Constants
BASE_PATH = './dataset'
MAX_LENGTH = 40
VOCABULARY_SIZE = 15000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIM = 512
UNITS = 512
EPOCHS = 10

# Load and preprocess data
def load_data():
    with open(f'{BASE_PATH}/annotations/captions_train2017.json', 'r') as f:
        data = json.load(f)['annotations']

    img_cap_pairs = []
    for sample in data:
        img_name = '%012d.jpg' % sample['image_id']
        img_cap_pairs.append([img_name, sample['caption']])

    captions = pd.DataFrame(img_cap_pairs, columns=['image', 'caption'])
    captions['image'] = captions['image'].apply(lambda x: f'{BASE_PATH}/train2017/{x}')
    captions = captions.sample(70000).reset_index(drop=True)
    captions['caption'] = captions['caption'].apply(preprocess)
    return captions

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return f'[start] {text} [end]'

# Tokenization
def tokenize(captions):
    tokenizer = tf.keras.layers.TextVectorization(max_tokens=VOCABULARY_SIZE, standardize=None, output_sequence_length=MAX_LENGTH)
    tokenizer.adapt(captions['caption'])
    pickle.dump(tokenizer.get_vocabulary(), open('vocab_coco.file', 'wb'))
    return tokenizer

# Prepare datasets
def prepare_datasets(captions, tokenizer):
    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(captions['image'], captions['caption']):
        img_to_cap_vector[img].append(cap)

    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys) * 0.8)
    img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

    train_imgs, train_captions = [], []
    for imgt in img_name_train_keys:
        train_imgs.extend([imgt] * len(img_to_cap_vector[imgt]))
        train_captions.extend(img_to_cap_vector[imgt])

    val_imgs, val_captions = [], []
    for imgv in img_name_val_keys:
        val_imgs.extend([imgv] * len(img_to_cap_vector[imgv]))
        val_captions.extend(img_to_cap_vector[imgv])

    return train_imgs, train_captions, val_imgs, val_captions

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [299, 299])
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

def load_image_and_caption(img_path, caption):
    img = load_image(img_path)
    caption = tokenizer(caption)
    return img, caption

def create_dataset(imgs, captions):
    dataset = tf.data.Dataset.from_tensor_slices((imgs, captions))
    dataset = dataset.map(lambda img, cap: load_image_and_caption(img, cap), num_parallel_calls=tf.data.AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

# CNN Encoder
class CNN_Encoder(tf.keras.Model):
    def __init__(self):
        super(CNN_Encoder, self).__init__()
        self.inception_v3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        self.inception_v3.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(EMBEDDING_DIM, activation='relu')

    def call(self, x):
        x = self.inception_v3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

 # RNN Decoder
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        x = self.fc1(output)
        x = self.fc2(x)
        return x, state

# BERT Encoder
class BERT_Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(BERT_Encoder, self).__init__()
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.position_embedding = tf.keras.layers.Embedding(MAX_LENGTH, embedding_dim)

    def call(self, x):
        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(tf.range(start=0, limit=MAX_LENGTH, delta=1))
        return token_embedding + position_embedding

# Image Captioning Model
class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, cnn_encoder, rnn_decoder, bert_encoder):
        super(ImageCaptioningModel, self).__init__()
        self.cnn_encoder = cnn_encoder
        self.rnn_decoder = rnn_decoder
        self.bert_encoder = bert_encoder

    def call(self, img, caption):
        img = self.cnn_encoder(img)
        caption = self.bert_encoder(caption)
        hidden = tf.zeros((caption.shape[0], self.rnn_decoder.units))
        outputs = []
        for t in range(1, caption.shape[1]):
            output, hidden = self.rnn_decoder(caption[:, t-1], hidden)
            outputs.append(output)
        outputs = tf.stack(outputs, axis=1)
        return outputs

# Train the model
def train_model(model, train_dataset, val_dataset):
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(EPOCHS):
        train_loss = tf.keras.metrics.Mean(name='train_loss')  # Create a new Mean instance for each epoch
        val_loss = tf.keras.metrics.Mean(name='val_loss')      # Create a new Mean instance for each epoch

        # Training loop
        for img, cap in train_dataset:
            with tf.GradientTape() as tape:
                output = model(img, cap)
                loss = loss_object(cap[:, 1:], output)  # Exclude the start token from the labels
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)

        # Validation loop
        for img, cap in val_dataset:
            output = model(img, cap)
            loss = loss_object(cap[:, 1:], output)  # Exclude the start token from the labels
            val_loss(loss)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss.result()}, Val Loss: {val_loss.result()}')

if __name__ == '__main__':
    captions = load_data()
    tokenizer = tokenize(captions)
    train_imgs, train_captions, val_imgs, val_captions = prepare_datasets(captions, tokenizer)
    train_dataset = create_dataset(train_imgs, train_captions)
    val_dataset = create_dataset(val_imgs, val_captions)

    cnn_encoder = CNN_Encoder()
    rnn_decoder = RNN_Decoder(EMBEDDING_DIM, UNITS, VOCABULARY_SIZE)
    bert_encoder = BERT_Encoder(VOCABULARY_SIZE, EMBEDDING_DIM)
    model = ImageCaptioningModel(cnn_encoder, rnn_decoder, bert_encoder)

    train_model(model, train_dataset, val_dataset)