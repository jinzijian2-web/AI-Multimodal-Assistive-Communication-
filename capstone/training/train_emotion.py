"""
Emotion Recognition Model Training Script
Using FER2013 dataset in folder format
Includes confusion matrix visualization
"""

import tensorflow as tf
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== Configuration ====================
EMOTIONS = {
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
    'neutral': 4, 'sad': 5, 'surprise': 6
}
EMOTION_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Training parameters
EPOCHS = 20
BATCH_SIZE = 64

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'fer2013')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'training_results')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set matplotlib to use English (avoid font issues)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== Data Loading ====================
def load_images_from_folder(folder_path, img_size=(48, 48)):
    """Load images from folder structure"""
    X = []
    y = []
    
    for emotion_name, emotion_id in EMOTIONS.items():
        emotion_folder = os.path.join(folder_path, emotion_name)
        
        if not os.path.exists(emotion_folder):
            print(f"Warning: Folder not found {emotion_folder}")
            continue
        
        image_files = [f for f in os.listdir(emotion_folder) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"Loading {emotion_name}: {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(emotion_folder, img_file)
            try:
                img = tf.keras.utils.load_img(img_path, color_mode='grayscale', target_size=img_size)
                img_array = tf.keras.utils.img_to_array(img) / 255.0
                X.append(img_array)
                y.append(emotion_id)
            except Exception as e:
                print(f"Failed to load: {img_file}")
    
    return np.array(X), np.array(y)

# ==================== Model Creation ====================
def create_emotion_model():
    """Create emotion recognition CNN model"""
    model = tf.keras.Sequential([
        # First convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                               input_shape=(48, 48, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Fully connected layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==================== Visualization Functions ====================
def plot_training_history(history, save_path):
    """Plot training history curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy curve
    axes[0].plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Mark best validation accuracy
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    axes[0].scatter(best_epoch, best_val_acc, color='green', s=100, zorder=5)
    axes[0].annotate(f'Best: {best_val_acc:.2%}', 
                    xy=(best_epoch, best_val_acc),
                    xytext=(best_epoch + 1, best_val_acc - 0.05),
                    fontsize=10)
    
    # Loss curve
    axes[1].plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Training history saved: {save_path}")
    return best_val_acc, best_epoch

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    overall_acc = np.sum(cm.diagonal()) / np.sum(cm)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 11})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix (Overall Accuracy: {overall_acc:.2%})', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    print(f"Confusion matrix saved: {save_path}")
    
    print("\nPer-class accuracy:")
    for name, acc in zip(class_names, class_acc):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {name:8}: {acc:.2%} {bar}")
    
    return cm, class_acc, overall_acc

def plot_normalized_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot normalized confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 11})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Normalized Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    print(f"Normalized confusion matrix saved: {save_path}")

# ==================== Report Generation ====================
def save_training_report(history, test_acc, class_acc, cm, class_names, save_path):
    """Save training report"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Emotion Recognition Model Training Report\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("[Training Parameters]\n")
        f.write(f"  - Optimizer: Adam (learning_rate=0.001)\n")
        f.write(f"  - Loss Function: Categorical Crossentropy\n")
        f.write(f"  - Batch Size: {BATCH_SIZE}\n")
        f.write(f"  - Epochs: {len(history.history['loss'])} (target: 20)\n\n")
        
        f.write("[Training Results]\n")
        f.write(f"  - Final Training Accuracy: {history.history['accuracy'][-1]:.4f} ({history.history['accuracy'][-1]*100:.2f}%)\n")
        f.write(f"  - Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f} ({history.history['val_accuracy'][-1]*100:.2f}%)\n")
        f.write(f"  - Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}\n")
        f.write(f"  - Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n\n")
        
        f.write("[Per-class Accuracy]\n")
        for name, acc in zip(class_names, class_acc):
            f.write(f"  {name}: {acc:.4f} ({acc*100:.2f}%)\n")
        
        f.write("\n[Confusion Matrix]\n")
        f.write("True\\Pred")
        for name in class_names:
            f.write(f"{name:>8}")
        f.write("\n")
        for i, name in enumerate(class_names):
            f.write(f"{name:>8}")
            for j in range(7):
                f.write(f"{cm[i][j]:8d}")
            f.write("\n")
        
        f.write("\n[Training Details]\n")
        f.write("Epoch\tTrain Acc\tVal Acc\t\tTrain Loss\tVal Loss\n")
        for i in range(len(history.history['loss'])):
            f.write(f"{i+1}\t{history.history['accuracy'][i]:.4f}\t\t"
                   f"{history.history['val_accuracy'][i]:.4f}\t\t"
                   f"{history.history['loss'][i]:.6f}\t{history.history['val_loss'][i]:.6f}\n")
    
    print(f"Training report saved: {save_path}")

# ==================== Main Training Function ====================
def train():
    """Main training function"""
    print("=" * 60)
    print("Emotion Recognition Model Training")
    print(f"Epochs: {EPOCHS}")
    print("=" * 60)
    
    # Create timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f'emotion_run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be saved to: {run_dir}")
    
    # Data paths
    train_path = os.path.join(DATA_DIR, 'train')
    test_path = os.path.join(DATA_DIR, 'test')
    
    if not os.path.exists(train_path):
        print(f"Error: Training folder not found: {train_path}")
        return
    
    # Load data
    print("\n[1/5] Loading training data...")
    X_train_full, y_train_full = load_images_from_folder(train_path)
    print(f"Training data loaded: {X_train_full.shape[0]} images")
    
    # Print class distribution
    print("\nTraining set class distribution:")
    for i, name in enumerate(EMOTION_NAMES):
        count = np.sum(y_train_full == i)
        print(f"  {name}: {count} ({count/len(y_train_full)*100:.1f}%)")
    
    # Split training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )
    
    # Load test data
    if os.path.exists(test_path):
        print("\nLoading test data...")
        X_test, y_test = load_images_from_folder(test_path)
        print(f"Test data loaded: {X_test.shape[0]} images")
    else:
        print("\nTest folder not found, splitting from training data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
    
    # Convert to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 7)
    y_val = tf.keras.utils.to_categorical(y_val, 7)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, 7)
    
    print(f"\nData split:")
    print(f"  Training: {X_train.shape[0]} images")
    print(f"  Validation: {X_val.shape[0]} images")
    print(f"  Test: {X_test.shape[0]} images")
    
    # Create model
    print("\n[2/5] Creating model...")
    model = create_emotion_model()
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(run_dir, 'emotion_model_best.h5'),
            monitor='val_accuracy', save_best_only=True, verbose=1
        )
    ]
    
    # Train
    print(f"\n[3/5] Starting training ({EPOCHS} epochs)...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Get predictions for confusion matrix
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Save final model
    final_model_path = os.path.join(run_dir, 'emotion_model_final.h5')
    model.save(final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Plot training history
    print("\n[4/5] Generating training curves...")
    plot_path = os.path.join(run_dir, 'training_history.png')
    best_val_acc, best_epoch = plot_training_history(history, plot_path)
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    
    # Plot confusion matrix
    print("\n[5/5] Generating confusion matrices...")
    cm_path = os.path.join(run_dir, 'confusion_matrix.png')
    cm, class_acc, overall_acc = plot_confusion_matrix(y_test, y_pred, EMOTION_NAMES, cm_path)
    
    cm_norm_path = os.path.join(run_dir, 'confusion_matrix_normalized.png')
    plot_normalized_confusion_matrix(y_test, y_pred, EMOTION_NAMES, cm_norm_path)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=EMOTION_NAMES))
    
    # Save training report
    print("\nSaving training report...")
    report_path = os.path.join(run_dir, 'training_report.txt')
    save_training_report(history, test_acc, class_acc, cm, EMOTION_NAMES, report_path)
    
    # Save to default location
    model.save(os.path.join(MODELS_DIR, 'emotion_model.h5'))
    print(f"\nModel also saved to: {os.path.join(MODELS_DIR, 'emotion_model.h5')}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"📁 Results saved to: {run_dir}")
    print(f"📊 Test Accuracy: {test_acc:.2%}")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2%}")
    print("=" * 60)

if __name__ == '__main__':
    train()

