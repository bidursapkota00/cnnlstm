import os
import re
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

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


features = {}

with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)


with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()

vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

mapping = {}
for line in tqdm(captions_doc.split('\n')):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)


def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = re.sub(r'[^A-Za-z\s]', '', caption)
            caption = re.sub(r'\s+', ' ', caption)
            caption = 'startseq ' + \
                ' '.join([word for word in caption.split()
                         if len(word) > 1]) + ' endseq'
            captions[i] = caption


clean(mapping)

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(caption.split()) for caption in all_captions)


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


def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break

    return in_text


def generate_caption(image_name):
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('---------------------Predicted---------------------')
    print(y_pred)


class ImageFamiliarityDetector:
    def __init__(self, training_features, similarity_threshold=0.3, confidence_threshold=0.5):
        """
        Initialize the familiarity detector

        Args:
            training_features: Dictionary of features from training images
            similarity_threshold: Minimum cosine similarity to training data
            confidence_threshold: Minimum prediction confidence required
        """
        self.training_features = training_features
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold

        # Convert training features to array for similarity computation
        self.training_feature_matrix = np.vstack(
            list(training_features.values()))

        # Optional: Create clusters of training features for more sophisticated detection
        self.setup_clusters()

    def setup_clusters(self, n_clusters=50):
        """Create clusters of training features to better understand data distribution"""
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = self.kmeans.fit_predict(
            self.training_feature_matrix)

        # Calculate cluster statistics
        self.cluster_stats = {}
        for i in range(n_clusters):
            cluster_features = self.training_feature_matrix[self.cluster_labels == i]
            self.cluster_stats[i] = {
                'center': self.kmeans.cluster_centers_[i],
                'std': np.std(cluster_features, axis=0),
                'size': len(cluster_features)
            }

    def extract_features(self, image_path):
        """Extract VGG16 features from a new image"""
        try:
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape(
                (1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            features = vgg_model.predict(image, verbose=0)
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def calculate_similarity_score(self, image_features):
        """Calculate similarity between image features and training data"""
        # Reshape if needed
        if len(image_features.shape) > 1:
            image_features = image_features.flatten().reshape(1, -1)

        # Calculate cosine similarity with all training features
        similarities = cosine_similarity(
            image_features, self.training_feature_matrix)

        # Get statistics
        max_similarity = np.max(similarities)
        mean_similarity = np.mean(similarities)

        # Find nearest cluster
        cluster_distances = np.linalg.norm(
            self.kmeans.cluster_centers_ - image_features, axis=1
        )
        nearest_cluster = np.argmin(cluster_distances)
        cluster_distance = cluster_distances[nearest_cluster]

        return {
            'max_similarity': max_similarity,
            'mean_similarity': mean_similarity,
            'nearest_cluster': nearest_cluster,
            'cluster_distance': cluster_distance,
            'top_5_similarities': np.sort(similarities.flatten())[-5:]
        }

    def calculate_prediction_confidence(self, model_predictions):
        """Calculate confidence based on model output distribution"""
        # Get the probability distribution of predictions
        max_prob = np.max(model_predictions)
        entropy = -np.sum(model_predictions * np.log(model_predictions + 1e-8))

        # Normalized entropy (lower is more confident)
        normalized_entropy = entropy / np.log(len(model_predictions))
        confidence = max_prob * (1 - normalized_entropy)

        return confidence, max_prob, normalized_entropy

    def is_familiar(self, image_path, model, tokenizer, max_length, verbose=True):
        """
        Determine if an image is familiar to the model

        Returns:
            dict: Contains familiarity decision and detailed metrics
        """
        # Extract features from the new image
        image_features = self.extract_features(image_path)
        if image_features is None:
            return {'is_familiar': False, 'reason': 'Could not extract features'}

        # Calculate similarity metrics
        similarity_metrics = self.calculate_similarity_score(image_features)

        # Generate caption and get prediction confidence
        caption_with_probs = self.predict_caption_with_confidence(
            model, image_features, tokenizer, max_length
        )

        # Decision logic
        is_similar_enough = similarity_metrics['max_similarity'] > self.similarity_threshold
        is_confident_enough = caption_with_probs['avg_confidence'] > self.confidence_threshold

        # Additional checks
        cluster_check = similarity_metrics['cluster_distance'] < np.mean([
            stats['std'].mean() * 2 for stats in self.cluster_stats.values()
        ])

        # is_familiar = is_similar_enough and is_confident_enough and cluster_check
        is_familiar = is_similar_enough and is_confident_enough

        result = {
            'is_familiar': is_familiar,
            'similarity_score': similarity_metrics['max_similarity'],
            'prediction_confidence': caption_with_probs['avg_confidence'],
            'predicted_caption': caption_with_probs['caption'],
            'detailed_metrics': {
                'similarity_metrics': similarity_metrics,
                'caption_metrics': caption_with_probs,
                'cluster_check': cluster_check
            }
        }

        if verbose:
            print(
                f"Similarity Score: {similarity_metrics['max_similarity']:.3f}")
            print(
                f"Prediction Confidence: {caption_with_probs['avg_confidence']:.3f}")
            print(f"Cluster Distance Check: {cluster_check}")
            print(f"Is Familiar: {is_familiar}")

        return result

    def predict_caption_with_confidence(self, model, image_features, tokenizer, max_length):
        """Generate caption while tracking prediction confidence"""
        in_text = 'startseq'
        confidences = []

        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], max_length)

            # Get full probability distribution
            predictions = model.predict(
                [image_features, sequence], verbose=0)[0]

            # Calculate confidence for this prediction
            confidence, max_prob, entropy = self.calculate_prediction_confidence(
                predictions)
            confidences.append(confidence)

            # Get the predicted word
            yhat = np.argmax(predictions)
            word = self.idx_to_word(yhat, tokenizer)

            if word is None:
                break

            in_text += ' ' + word

            if word == 'endseq':
                break

        return {
            'caption': in_text,
            'confidences': confidences,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'min_confidence': np.min(confidences) if confidences else 0
        }

    def idx_to_word(self, integer, tokenizer):
        """Convert integer to word using tokenizer"""
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None


def generate_caption_with_familiarity_check(image_name, detector, model, tokenizer, max_length, mapping):
    """Generate caption with familiarity detection"""

    # Handle both full paths and just filenames
    if os.path.exists(image_name):
        img_path = image_name
        image_id = os.path.basename(image_name).split('.')[0]
    else:
        image_id = image_name.split('.')[0]
        img_path = os.path.join(BASE_DIR, "Images", image_name)

    print(f"\n{'='*50}")
    print(f"Processing: {os.path.basename(img_path)}")
    print(f"{'='*50}")

    # Check if image exists
    if not os.path.exists(img_path):
        print("Image file not found!")
        return

    # Load and display image
    image = Image.open(img_path)

    # Show actual captions if available in training data
    if image_id in mapping:
        print('\n---------------------Actual Captions---------------------')
        for caption in mapping[image_id]:
            print(caption)
    else:
        print(
            '\n---------------------New Image (not in training data)---------------------')

    # Check familiarity
    familiarity_result = detector.is_familiar(
        img_path, model, tokenizer, max_length)

    print('\n---------------------Prediction---------------------')
    if familiarity_result['is_familiar']:
        print("Image appears familiar to the model")
        print(f"Predicted Caption: {familiarity_result['predicted_caption']}")
    else:
        print("This model is not familiar with this type of image")
        print("The image appears to be outside the model's training domain.")
        print(
            f"Attempted caption (low confidence): {familiarity_result['predicted_caption']}")

        # Provide specific reasons
        if familiarity_result['similarity_score'] < detector.similarity_threshold:
            print(
                f"   - Low visual similarity to training data ({familiarity_result['similarity_score']:.3f})")
        if familiarity_result['prediction_confidence'] < detector.confidence_threshold:
            print(
                f"   - Low prediction confidence ({familiarity_result['prediction_confidence']:.3f})")


# Initialize familiarity detector
detector = ImageFamiliarityDetector(
    training_features=features,
    similarity_threshold=0.7,  # Adjust based on your needs
    confidence_threshold=0.2    # Adjust based on your needs
)

# Test with your existing images
test_images = [
    '667626_18933d713e.jpg',
    '1000268201_693b08cb0e.jpg',
    '1002674143_1b742ab4b8.jpg',
    # Add any new/unfamiliar images here
    'train.png'
]

for image_name in test_images:
    generate_caption_with_familiarity_check(
        image_name, detector, model, tokenizer, max_length, mapping
    )
