import os
import pickle
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


CIFAR_DIR = os.path.expanduser("~/localdata/cifar-10-batches-py")
RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)

# CIFAR-10 のダウンロード

def load_batch(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="bytes")

    x = data[b"data"]
    y = np.array(data[b"labels"])

    x = x.reshape(-1, 3, 32, 32)
    x = x.transpose(0, 2, 3, 1)

    return x, y

print("Loading CIFAR-10...")

train_images = []
train_labels = []

for i in range(1, 6):
    x, y = load_batch(
        os.path.join(CIFAR_DIR, f"data_batch_{i}")
    )
    train_images.append(x)
    train_labels.append(y)

x_train = np.concatenate(train_images)
y_train = np.concatenate(train_labels)

x_test, y_test = load_batch(
    os.path.join(CIFAR_DIR, "test_batch")
)

print("Train:", x_train.shape)
print("Test :", x_test.shape)

# bird cat dog のみのデータ


target_classes = [2, 3, 5]

train_mask = np.isin(y_train, target_classes)
test_mask = np.isin(y_test, target_classes)

x_train = x_train[train_mask]
y_train = y_train[train_mask]

x_test = x_test[test_mask]
y_test = y_test[test_mask]

mapping = {
    2: 0,  # bird
    3: 1,  # cat
    5: 2   # dog
}

y_train = np.array([mapping[v] for v in y_train])
y_test = np.array([mapping[v] for v in y_test])

y_train_cat = to_categorical(y_train, 3)
y_test_cat = to_categorical(y_test, 3)

print("Filtered train:", x_train.shape)
print("Filtered test :", x_test.shape)

# Resolution degradation

def degrade_images(images, size):
    output = []

    for img in images:
        small = cv2.resize(
            img,
            (size, size),
            interpolation=cv2.INTER_AREA
        )

        restored = cv2.resize(
            small,
            (32, 32),
            interpolation=cv2.INTER_NEAREST
        )

        output.append(restored)

    return np.array(output)

# Sample image


sample = x_test[0]

sizes = [32, 16, 8, 4]

plt.figure(figsize=(12, 3))

for i, size in enumerate(sizes):

    if size == 32:
        img = sample
    else:
        img = degrade_images(
            np.array([sample]),
            size
        )[0]

    plt.subplot(1, 4, i + 1)
    plt.imshow(img)
    plt.title(f"{size}x{size}")
    plt.axis("off")

plt.tight_layout()
plt.savefig(
    os.path.join(
        RESULT_DIR,
        "sample_resolution.png"
    ),
    dpi=300
)
plt.close()

# Model

def build_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(32, 32, 3)
        ),

        tf.keras.layers.Rescaling(
            1.0 / 255
        ),

        tf.keras.layers.Conv2D(
            32,
            3,
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            64,
            3,
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.5
        ),

        tf.keras.layers.Dense(
            3,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


resolutions = [32, 16, 8, 4]

accuracy_results = {}
metrics_table = []

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

for res in resolutions:

    print("=" * 50)
    print("Resolution:", res)
    print("=" * 50)

    if res == 32:

        train_img = x_train
        test_img = x_test

    else:

        train_img = degrade_images(
            x_train,
            res
        )

        test_img = degrade_images(
            x_test,
            res
        )

    model = build_model()

    history = model.fit(
        train_img,
        y_train_cat,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    loss, acc = model.evaluate(
        test_img,
        y_test_cat,
        verbose=0
    )

    accuracy_results[res] = acc

    pred = model.predict(
        test_img,
        verbose=0
    )

    pred_class = np.argmax(
        pred,
        axis=1
    )

    report = classification_report(
        y_test,
        pred_class,
        output_dict=True
    )

    metrics_table.append({

        "Resolution": res,

        "Bird Precision":
            report["0"]["precision"],

        "Bird Recall":
            report["0"]["recall"],

        "Bird F1":
            report["0"]["f1-score"],

        "Cat Precision":
            report["1"]["precision"],

        "Cat Recall":
            report["1"]["recall"],

        "Cat F1":
            report["1"]["f1-score"],

        "Dog Precision":
            report["2"]["precision"],

        "Dog Recall":
            report["2"]["recall"],

        "Dog F1":
            report["2"]["f1-score"]
    })

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="train"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            f"history_{res}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    cm = confusion_matrix(
        y_test,
        pred_class
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Bird",
            "Cat",
            "Dog"
        ]
    )

    disp.plot()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            f"confusion_matrix_{res}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

#  CSVの保存

df = pd.DataFrame(metrics_table)

df.to_csv(
    os.path.join(
        RESULT_DIR,
        "metrics.csv"
    ),
    index=False
)

#正答率のグラフ出力


plt.figure(figsize=(8, 5))

x = np.array(
    list(accuracy_results.keys()),
    dtype=float
)

y = np.array(
    list(accuracy_results.values()),
    dtype=float
)

plt.plot(
    x,
    y,
    marker="o"
)

plt.xlabel("Resolution")
plt.ylabel("Accuracy")
plt.grid(True)

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "accuracy_vs_resolution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# Recall graph
# ==========================================

plt.figure(figsize=(8, 5))

x = df["Resolution"].to_numpy(dtype=float)

plt.plot(
    x,
    df["Bird Recall"].to_numpy(dtype=float),
    marker="o",
    label="Bird"
)

plt.plot(
    x,
    df["Cat Recall"].to_numpy(dtype=float),
    marker="o",
    label="Cat"
)

plt.plot(
    x,
    df["Dog Recall"].to_numpy(dtype=float),
    marker="o",
    label="Dog"
)

plt.xlabel("Resolution")
plt.ylabel("Recall")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "recall_vs_resolution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Finished.")
print("Results saved in ./results")
