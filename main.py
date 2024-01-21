import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import uuid
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import Model, layers, Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def get_dataset(dataset_path, image_size, batch_size, split_size):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        seed=99,
        shuffle=True,
        image_size=(image_size, image_size),
        batch_size=batch_size
    )
    print(ds.class_names)
    ds_size = len(ds)

    train_size = int(split_size * ds_size)
    val_size = int((1 - split_size) * ds_size)
    #     test_size = int(test_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)

    return train_ds, val_ds


def plot_loss_and_acc(history, save_path):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 5}
    plt.rc('font', **font)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.title('Loss Graph')
    plt.savefig(f'{save_path}/loss curve of 6 inch 4H-SiC boule_with pretrain.png')
    plt.show()

    # plot the accuracy curve
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.title('Accuracy Graph')
    plt.savefig(f'{save_path}/accuracy curve of 6 inch 4H-SiC boule_with pretrain.png')
    plt.show()


def plot_feature_maps(layer_idx, based_model, image_path, save_path):
    vis_model = Model(inputs=based_model.inputs, outputs=based_model.layers[layer_idx].output)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    features = vis_model.predict(img)
    fig = plt.figure(figsize=(30, 30))
    for i in range(1, 26):
        plt.subplot(5, 5, i)
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 20}
        plt.rc('font', **font)
        plt.imshow(features[0, :, :, i - 1], cmap='viridis')
        # plt.show()
    plt.savefig(f'{save_path}/6 inch 4H-SiC boule_with pretrain_feature maps_carbon inclusion_2023-07-15_152953.bmp-171layer.jpg')


def funcPredictAndDisplay(model, img_path, result_dir, class_names):
    img = tf.keras.preprocessing.image.load_img(img_path)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.preprocessing.image.smart_resize(img_array, (224, 224))
    img_array = tf.expand_dims(img_array, 0)

    pred = model.predict(img_array)

    pred_class = class_names[np.argmax(pred[0])]
    confidence = round(100 * (np.max(pred[0])), 2)

    # Display predicted class and confidence on the image
    plt.figure()
    plt.imshow(img)
    plt.title(f"Predicted: {pred_class}\nConfidence: {confidence}%")
    plt.axis('off')

    # Save the modified image with a unique filename
    result_filepath = os.path.join(result_dir, f"result_{uuid.uuid1()}.png")
    plt.savefig(result_filepath)
    plt.close()


def plot_confusion_matrix(labels, pred, target, save_path):
    y_pred = np.argmax(pred, axis=1)
    cm = confusion_matrix(target, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    colors = [(1, 1, 1), (91 / 255, 191 / 255, 176 / 255)]  # White to green
    cmap = LinearSegmentedColormap.from_list('white_to_green', colors, N=256)

    # Plot the confusion matrix with adjusted font size
    plt.figure(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cmn, display_labels=labels)
    disp.plot(cmap=cmap)

    # Set the font size for class labels outside the matrix
    font = {'family': 'normal', 'weight': 'bold', 'size': 12}
    plt.rc('font', **font)

    plt.title('Confusion Matrix', fontsize=14)  # Adjust title font size
    plt.xlabel('Predicted Labels', fontsize=2)  # Adjust x-axis label font size
    plt.ylabel('True Labels', fontsize=2)  # Adjust y-axis label font size

    # Save the confusion matrix plot with 300dpi resolution
    plt.savefig(f"./{save_path}/6 inch 4H-SiC boule_with pretrain_confusion_matrix.jpg", dpi=300)

    # Show the confusion matrix plot
    plt.show()

    classification_rep = classification_report(target, y_pred, target_names=labels, digits=3, output_dict=True)
    classification_df = pd.DataFrame(classification_rep).transpose()
    classification_df.to_csv('with pretrain_performance metrics.csv', index=False)
    print(classification_report(target, y_pred, target_names=labels, digits=3))


def train_resnet_model(num_classes, image_size, dropout, lr, epochs, save_path, phase):
    res50_callbacks = EarlyStopping(monitor="loss", patience=5, verbose=1, mode='auto')
    csv_logger = CSVLogger('6 inch 4H-SiC boule_with pretrain_result.csv')
    res50_best_model_file = f"{save_path}/res50_drop_best_weights.h5"
    res50_best_model = ModelCheckpoint(res50_best_model_file, monitor='val_accuracy', save_best_only=True, verbose=1)

    res50_model = ResNet50(weights='imagenet', pooling='avg',
                           include_top=False,
                           input_shape=(image_size, image_size, 3),
                           classes=1000)

    # layers trainable are set false
    for layer in res50_model.layers:
        layer.trainable = False

    last_output = res50_model.layers[-1].output
    x = Dense(128, activation='relu')(last_output)  # First Dense layer with 512 neuron and relu activation function
    x = Dropout(dropout)(x)  # Dropout layer add with 40%
    x = Dense(64, activation='relu')(x)  # Third Dense layer with 256 neuron and relu activation function
    x = Dropout(dropout)(x)  # Dropout layer add with 30%
    x = Dense(num_classes, activation='softmax')(x)  # Fourth Dense layer with 50 neuron and softmax activation function
    model = Model(res50_model.input, x)

    model.summary()

    learn_rate = lr
    adam = Adam(learning_rate=learn_rate)
    model.compile(optimizer=adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    if phase == 'eval':
        return model
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds,
                        callbacks=[res50_best_model, res50_callbacks, csv_logger], verbose=1)
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='G:\\code\\12.29\\6 inch 4H-SiC boule dataset_\\6 inch 4H-SiC boule_test dataset', help="your dataset path")
    parser.add_argument('--phase', type=str, default='eval', choices=['train', 'eval'], help="choose to train or eval")
    parser.add_argument('--image_size', type=int, default=224, help="image width and height in datasets")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--split_size', type=float, default=0.8, help="split size for train and validation dataset")
    parser.add_argument('--epochs', type=int, default=50, help="epochs")
    parser.add_argument('--dropout', type=float, default=0.3, help="dropout ratio")
    parser.add_argument('--num_classes', type=int, default=6, help="num of categories")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--inference_image_path', type=str, default='G:\\code\\12.29\\6 inch 4H-SiC boule dataset_\\test\\2023-07-15_152832.bmp', help="select an image to inference")
    parser.add_argument('--save_path', type=str, default='./result', help="where to save your best model weights")
    args = parser.parse_args()

    if args.phase == 'train':
        train_ds, val_ds = get_dataset(
            dataset_path=args.dataset_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            split_size=args.split_size
        )

        model, history = train_resnet_model(num_classes=args.num_classes,
                                            image_size=args.image_size,
                                            dropout=args.dropout,
                                            lr=args.lr,
                                            epochs=args.epochs,
                                            save_path=args.save_path,
                                            phase=args.phase)

        # plot_model(model, to_file=f"./{args.save_path}/model.png", show_shapes=True)
        plot_loss_and_acc(history, args.save_path)

    elif args.phase == 'eval':
        model = train_resnet_model(num_classes=args.num_classes,
                                   image_size=args.image_size,
                                   dropout=args.dropout,
                                   lr=args.lr,
                                   epochs=args.epochs,
                                   save_path=args.save_path,
                                   phase=args.phase)
        model.load_weights(f'{args.save_path}/res50_drop_best_weights.h5')
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            args.dataset_path,
            seed=99,
            shuffle=False,
            image_size=(args.image_size, args.image_size),
            batch_size=1,
        )
        target = []
        for x, y in test_ds:
            target.append(y.numpy())
        class_names = test_ds.class_names
        model.evaluate(test_ds)
        pred = model.predict(test_ds)
        plot_feature_maps(layer_idx=117,
                          based_model=model,
                          image_path=args.inference_image_path,
                          save_path=args.save_path)
        plot_confusion_matrix(class_names,
                              pred,
                              target,
                              args.save_path)
        funcPredictAndDisplay(model, args.inference_image_path, args.save_path, class_names)


