import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from tensorflow.keras.applications import (
    MobileNetV2,
    DenseNet121,
    ResNet50
)
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ====================
# CONFIGURATION
# ====================
class Config:
    DATASET_DIR = "/home/smurfy/Desktop/Plant-Doctor/DATASET/Grape"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 8
    VAL_SPLIT = 0.2
    SEED = 42
    PLOT_DIR = "./GRAPH"
    MODEL_DIR = "./MODELS"

# ====================
# GPU MEMORY LIMITATION HANDLING
# ====================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ====================
# DATA PIPELINE
# ====================
def load_data(config):
    train_ds = image_dataset_from_directory(
        config.DATASET_DIR,
        validation_split=config.VAL_SPLIT,
        subset="training",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int"
    )

    val_ds = image_dataset_from_directory(
        config.DATASET_DIR,
        validation_split=config.VAL_SPLIT,
        subset="validation",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int"
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))
    val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))

    return train_ds.concatenate(val_ds), class_names

# ====================
# CLASSIFIER HEAD
# ====================
def build_custom_cnn_head(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name="cnn_classifier")

# ====================
# PLOTTING UTILITIES
# ====================
def save_confusion_matrix(y_true, y_pred, class_names, title, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{title}_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_training_curves(history, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Accuracy plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_path = os.path.join(save_dir, f"{model_name}_accuracy.png")
    plt.savefig(acc_path)
    plt.close()
    print(f"Saved accuracy plot to {acc_path}")

    # Loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(save_dir, f"{model_name}_loss.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved loss plot to {loss_path}")

def plot_model_comparison(metrics_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]['accuracy'] for m in models]
    losses = [metrics_dict[m]['val_loss'] for m in models]

    # Accuracy bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(models, accuracies, color='skyblue')
    plt.title("Model Comparison - Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    acc_path = os.path.join(save_dir, "model_accuracy_comparison.png")
    plt.savefig(acc_path)
    plt.close()
    print(f"Saved model accuracy comparison plot to {acc_path}")

    # Loss bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(models, losses, color='salmon')
    plt.title("Model Comparison - Validation Loss")
    plt.ylabel("Validation Loss")
    plt.grid(axis='y')
    plt.tight_layout()
    loss_path = os.path.join(save_dir, "model_loss_comparison.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved model loss comparison plot to {loss_path}")

# ====================
# MAIN EXECUTION
# ====================
def main():
    config = Config()
    dataset, class_names = load_data(config)
    num_classes = len(class_names)

    base_models = {
        "MobileNetV2": MobileNetV2,
        "DenseNet121": DenseNet121,
        "ResNet50": ResNet50
    }

    model_metrics = {}  # For storing accuracy and loss per model

    for model_name, base_model_fn in base_models.items():
        print(f"\n--- Using {model_name} for feature extraction ---")

        base_model = base_model_fn(
            input_shape=config.IMG_SIZE + (3,),
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
        base_model.trainable = False

        features, labels = [], []
        unbatched_ds = dataset.unbatch().batch(4)

        for batch_images, batch_labels in tqdm(unbatched_ds, desc=f"Extracting with {model_name}", colour="cyan"):
            if model_name == "DenseNet121":
                preprocessed = tf.keras.applications.densenet.preprocess_input(batch_images)
            elif model_name == "ResNet50":
                preprocessed = tf.keras.applications.resnet50.preprocess_input(batch_images)
            else:
                preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(batch_images)

            batch_features = base_model(preprocessed, training=False).numpy()
            features.append(batch_features)
            labels.append(batch_labels.numpy())

        features = np.concatenate(features)
        labels = np.argmax(np.concatenate(labels), axis=1)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=config.SEED)

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, tf.keras.utils.to_categorical(y_train, num_classes))).batch(32)
        val_ds = tf.data.Dataset.from_tensor_slices((X_test, tf.keras.utils.to_categorical(y_test, num_classes))).batch(32)

        cnn_classifier = build_custom_cnn_head(input_shape=features.shape[1:], num_classes=num_classes)
        cnn_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(f"Training CNN classifier on top of {model_name} features")
        history = cnn_classifier.fit(train_ds, validation_data=val_ds, epochs=20)

        # Plot accuracy and loss curves
        curve_dir = os.path.join(config.PLOT_DIR, model_name, "CNN")
        plot_training_curves(history, model_name, curve_dir)

        y_pred_probs = cnn_classifier.predict(val_ds)
        y_pred = np.argmax(y_pred_probs, axis=1)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy with {model_name}: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # Save confusion matrix
        sub_dir = os.path.join(config.PLOT_DIR, model_name, "CNN")
        save_confusion_matrix(y_test, y_pred, class_names, f"{model_name}_CNN", sub_dir)

        # Save model
        save_path = os.path.join(config.MODEL_DIR, model_name, "CNN", f"{model_name}_CNN.h5")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cnn_classifier.save(save_path)
        print(f"Saved CNN classifier model to {save_path}")

        final_val_loss = history.history['val_loss'][-1]
        model_metrics[model_name] = {
            "accuracy": acc,
            "val_loss": final_val_loss
        }

    # Plot comparison between models
    plot_model_comparison(model_metrics, config.PLOT_DIR)

if __name__ == "__main__":
    main()
