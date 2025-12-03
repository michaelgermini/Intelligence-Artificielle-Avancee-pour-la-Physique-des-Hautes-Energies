# 22.4 TensorFlow/Keras - Fondamentaux

---

## Introduction

**TensorFlow** est un framework de deep learning développé par Google, particulièrement adapté pour la production et le déploiement. **Keras** fournit une API haut niveau qui simplifie la construction et l'entraînement de modèles. Cette section présente les fondamentaux de TensorFlow et Keras.

---

## Installation et Configuration

### Setup

```python
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Vérifier GPU
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
    gpu = tf.config.list_physical_devices('GPU')[0]
    print(f"GPU: {gpu}")
```

---

## Tenseurs TensorFlow

### Création et Opérations

```python
# Création de tenseurs
t1 = tf.constant([[1, 2], [3, 4]])
t2 = tf.constant([[5, 6], [7, 8]])

# Opérations
sum_t = tf.add(t1, t2)  # ou t1 + t2
prod_t = tf.multiply(t1, t2)  # ou t1 * t2
matmul = tf.matmul(t1, t2)  # ou t1 @ t2

# Variables (modifiables)
var = tf.Variable(initial_value=[[1.0, 2.0], [3.0, 4.0]])
var.assign([[5.0, 6.0], [7.0, 8.0]])  # Modifier valeur

# Tensors avec gradients
x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y = x ** 2
grad = tape.gradient(y, x)
print(f"dy/dx = {grad.numpy()}")  # 4.0
```

---

## Keras: API Simplifiée

### Modèle Séquentiel

```python
from tensorflow import keras
from tensorflow.keras import layers

# Modèle séquentiel (le plus simple)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Compiler modèle
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Afficher architecture
model.summary()
```

---

## API Fonctionnelle

### Modèles Plus Complexes

```python
# API fonctionnelle pour modèles non-séquentiels
inputs = keras.Input(shape=(10,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(3, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Entraînement

### Fit et Callbacks

```python
# Données simulées
import numpy as np

X_train = np.random.randn(1000, 10)
y_train = np.random.randint(0, 3, (1000,))
X_val = np.random.randn(200, 10)
y_val = np.random.randint(0, 3, (200,))

# Entraînement
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(patience=2)
    ]
)

# Évaluation
test_loss, test_acc = model.evaluate(X_val, y_val)
print(f"Test accuracy: {test_acc:.4f}")
```

---

## Callbacks

### Utilitaires d'Entraînement

```python
callbacks = [
    # Early stopping
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    
    # Save best model
    keras.callbacks.ModelCheckpoint(
        'checkpoints/best_model.h5',
        monitor='val_loss',
        save_best_only=True
    ),
    
    # Learning rate reduction
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    ),
    
    # TensorBoard
    keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]
```

---

## Exemple Complet

### Pipeline Entraînement

```python
# 1. Préparer données
X_train = np.random.randn(1000, 10).astype(np.float32)
y_train = np.random.randint(0, 3, (1000,)).astype(np.int32)

# 2. Créer modèle
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# 3. Compiler
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Entraîner
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    verbose=1
)

# 5. Prédictions
predictions = model.predict(X_train[:5])
print(f"Predictions shape: {predictions.shape}")
```

---

## Exercices

### Exercice 22.4.1
Créez un modèle Keras pour classification avec 3 couches cachées et entraînez-le.

### Exercice 22.4.2
Implémentez un modèle avec API fonctionnelle Keras qui a deux branches séparées.

### Exercice 22.4.3
Configurez des callbacks (early stopping, model checkpoint) et observez leur effet.

---

## Points Clés à Retenir

> 📌 **TensorFlow offre framework complet pour production**

> 📌 **Keras simplifie création et entraînement modèles**

> 📌 **API fonctionnelle permet modèles complexes (branches, skip connections)**

> 📌 **Callbacks automatisent tâches communes (early stopping, saving)**

> 📌 **TensorFlow/Keras excellent pour déploiement production**

---

*Section précédente : [22.3.3 DataLoaders](./22_03_03_DataLoaders.md) | Section suivante : [22.5 Bonnes Pratiques](./22_05_Bonnes_Pratiques.md)*

