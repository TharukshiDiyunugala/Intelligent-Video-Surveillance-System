import os
import numpy as np
import cv2
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import glob

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')
    print("Created 'models' directory")

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2

def build_autoencoder(input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)):
    """
    Build a convolutional autoencoder for anomaly detection
    """
    # Encoder
    input_img = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    
    decoded = Conv2D(CHANNELS, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return autoencoder

def load_training_data(data_dir='data/train'):
    """
    Load training images from directory
    If no data directory exists, generate synthetic data for demonstration
    """
    if not os.path.exists(data_dir):
        print(f"Warning: Training data directory '{data_dir}' not found.")
        print("Generating synthetic training data for demonstration...")
        return generate_synthetic_data()
    
    image_paths = glob.glob(os.path.join(data_dir, '**/*.jpg'), recursive=True)
    image_paths.extend(glob.glob(os.path.join(data_dir, '**/*.png'), recursive=True))
    
    if len(image_paths) == 0:
        print(f"Warning: No images found in '{data_dir}'.")
        print("Generating synthetic training data for demonstration...")
        return generate_synthetic_data()
    
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    
    images = np.array(images, dtype='float32') / 255.0
    print(f"Loaded {len(images)} training images from {data_dir}")
    return images

def generate_synthetic_data(num_samples=1000):
    """
    Generate synthetic training data for demonstration purposes
    """
    print(f"Generating {num_samples} synthetic images...")
    images = []
    
    for i in range(num_samples):
        # Create random patterns
        img = np.random.rand(IMG_HEIGHT, IMG_WIDTH, CHANNELS) * 0.3
        
        # Add some structure (rectangles, circles)
        if i % 2 == 0:
            cv2.rectangle(img, (50, 50), (150, 150), (0.7, 0.7, 0.7), -1)
        else:
            cv2.circle(img, (IMG_WIDTH//2, IMG_HEIGHT//2), 50, (0.7, 0.7, 0.7), -1)
        
        images.append(img)
    
    return np.array(images, dtype='float32')

def main():
    print("=" * 60)
    print("Training Autoencoder Model for Video Surveillance")
    print("=" * 60)
    
    # Load training data
    print("\n[1/4] Loading training data...")
    X_train = load_training_data()
    
    print(f"Training data shape: {X_train.shape}")
    
    # Build model
    print("\n[2/4] Building autoencoder model...")
    autoencoder = build_autoencoder()
    autoencoder.summary()
    
    # Setup callbacks
    checkpoint = ModelCheckpoint(
        'models/autoencoder.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=VALIDATION_SPLIT
    )
    
    # Train model
    print(f"\n[3/4] Training model for {EPOCHS} epochs...")
    history = autoencoder.fit(
        datagen.flow(X_train, X_train, batch_size=BATCH_SIZE, subset='training'),
        validation_data=datagen.flow(X_train, X_train, batch_size=BATCH_SIZE, subset='validation'),
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Save final model
    print("\n[4/4] Saving final model...")
    autoencoder.save('models/autoencoder.h5')
    print(f"Model saved to 'models/autoencoder.h5'")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print("\nYou can now run: python app.py")

if __name__ == "__main__":
    main()
