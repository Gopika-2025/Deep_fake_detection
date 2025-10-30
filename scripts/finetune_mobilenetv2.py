import os
import random
import shutil
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def flatten_folder(base_folder):
    base_path = Path(base_folder)
    for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            for img_file in subfolder.glob("*.*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dest = base_path / img_file.name
                    if not dest.exists():
                        shutil.move(str(img_file), str(dest))
            # Remove empty subfolder
            if not any(subfolder.iterdir()):
                subfolder.rmdir()
    print(f"✅ Flattened nested folders inside '{base_folder}'")

def split_dataset(src_base, train_dir, val_dir, categories, train_split=0.8):
    random.seed(42)
    for category in categories:
        src_folder = src_base / category
        train_cat_dir = train_dir / category
        val_cat_dir = val_dir / category

        os.makedirs(train_cat_dir, exist_ok=True)
        os.makedirs(val_cat_dir, exist_ok=True)

        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(list(src_folder.glob(ext)))

        print(f"Found {len(image_paths)} images in {src_folder}")

        if len(image_paths) == 0:
            print(f"⚠️ WARNING: No images found in {src_folder}")
            continue

        random.shuffle(image_paths)
        split_idx = int(len(image_paths) * train_split)
        train_imgs = image_paths[:split_idx]
        val_imgs = image_paths[split_idx:]

        for img in train_imgs:
            shutil.copy(img, train_cat_dir / img.name)
        for img in val_imgs:
            shutil.copy(img, val_cat_dir / img.name)

        print(f"✅ Copied {len(train_imgs)} train and {len(val_imgs)} val images for '{category}'")

def check_folder_nonempty(folder):
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        if len(list(folder.glob(ext))) > 0:
            return True
    return False

def main():
    base_dir = Path("D:/Deep_fake_morphing/data/cropped_faces")
    categories = ['real', 'fake']
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"

    # Step 1: Flatten any nested folders
    flatten_folder(base_dir / "real")
    flatten_folder(base_dir / "fake")

    # Step 2: Split dataset if needed
    split_needed = False
    for category in categories:
        if not check_folder_nonempty(train_dir / category) or not check_folder_nonempty(val_dir / category):
            split_needed = True
            break

    if split_needed:
        print("Splitting dataset into train and val folders...")
        split_dataset(base_dir, train_dir, val_dir, categories)
    else:
        print("Train/Val folders already exist and are not empty. Skipping split.")

    # Step 3: Verify data exists
    for split_folder in [train_dir, val_dir]:
        for category in categories:
            folder = split_folder / category
            if not check_folder_nonempty(folder):
                raise FileNotFoundError(f"No images found in {folder}. Please check your dataset!")

    # Step 4: Prepare data generators
    img_size = 224
    batch_size = 32

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        zoom_range=0.2,
        rotation_range=15
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary'
    )

    print("Loading validation data...")
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary'
    )

    # Step 5: Build and compile model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    base_model.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # Step 6: Train model
    epochs = 10
    print("Starting training...")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    # Step 7: Save model
    model.save("mobilenetv2_deepfake_model.keras")
    print("✅ Model saved as mobilenetv2_deepfake_model.keras")

if __name__ == "__main__":
    main()
