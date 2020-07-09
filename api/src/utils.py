import matplotlib.pyplot as plt
import numpy as np


class VisualizationUtils:
    def __init__(self, model, data, classes):
        self.data = data
        self.model = model
        self.classes = classes

    # Create plot with 5 x 5 table images
    def create_images_plot(self, dt=1, offset=0, with_predict=True):
        """
        :param dt: (optional, default: 1)
            Data Type: 0 - training images, 1 - test images
        :param offset: (optional, default: 0)
            Offset of data images (start image index to creating plot)
        :param with_predict: (optional, default: True)
            Write labels under image with predictions (if True than <right label>/<predict label>)
        """
        #
        if with_predict:
            predictions = self.model.predict(self.data[2 * dt][offset:offset + 25])
        plt.figure(figsize=(10, 10))

        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.data[2 * dt][offset + i], cmap=plt.cm.binary)
            if with_predict:
                plt.xlabel('%s/%s' % (self.classes[self.data[2 * dt + 1][offset + i]],
                                      self.classes[np.argmax(predictions[i])]))
            else:
                plt.xlabel(self.classes[self.data[2 * dt + 1][offset + i]])

        plt.show()

    # Create graph on model training and validation history
    # To compare graphs
    def plot_history(self, histories, key='acc'):
        plt.figure(figsize=(16, 10))

        for name, history in histories:
            val = plt.plot(history.epoch, history.history['val_'+key],
                           '--', label=name.title()+' Val')
            plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                     label=name.title()+' Train')

        plt.xlabel('Epochs')
        plt.ylabel(key.replace('_', ' ').title())
        plt.legend()

        plt.xlim([0, max(history.epoch)])
        plt.show()

    # Check layers
    def check_layers(self, check_layer):
        dense_layers = [layer for layer in self.model.layers if isinstance(layer, check_layer)]
        print(dense_layers)
        for layer in dense_layers:
            layer_weights = layer.get_weights()
            print("Dense weights: len input %s, sum input %s, len out %s, sum out %s" % (
                len(layer_weights[0]), np.sum(layer_weights[0]), len(layer_weights[1]), np.sum(layer_weights[1]))
            )

    # Show dots with labels and predictions
    def create_dots_plot(self):
        plt.figure(figsize=(8, 6))
        plt.grid(True)
        plt.xlim(-1, len(self.classes) + 1)
        plt.ylim(-1, len(self.classes) + 1)
        plt.scatter(self.data[1], np.argmax(self.model.predict(self.data[0]), axis=1), alpha=0.5, color='red')
        plt.scatter(self.data[3], np.argmax(self.model.predict(self.data[2]), axis=1), alpha=0.5, color='blue')
        plt.show()

    # Generate a table data
    # Columns - Right labels, Rows - Predictions. In cells - number images.
    def get_table_data(self, predict_data, label_data):
        predictions = np.argmax(self.model.predict(predict_data), axis=1)
        tableData = []
        tableColors = []
        default_gradient = [1, 1, 1, 1]
        for i in range(len(self.classes)):
            prediction_info = predictions[np.where(label_data == i)[0]]
            tableData.append(np.zeros(len(self.classes)))
            tableColors.append([default_gradient.copy() for _ in range(len(self.classes))])
            unique, counts = np.unique(prediction_info, return_counts=True)
            for j in range(len(unique)):
                tableData[i][unique[j]] = counts[j]
                tableColors[i][unique[j]][0] -= counts[j] / sum(counts)
                tableColors[i][unique[j]][2] -= counts[j] / sum(counts)

        tableData = np.array(tableData).transpose()
        for i in range(len(self.classes)):
            for j in range(i, len(self.classes)):
                tableColors[i][j], tableColors[j][i] = tableColors[j][i], tableColors[i][j]

        return tableData, tableColors

    def create_table(self):
        # Get table data on train data
        trainTableData = self.get_table_data(self.data[0], self.data[1])
        # Get table data on test data
        testTableData = self.get_table_data(self.data[2], self.data[3])
        # Table data on train + test data
        tableData = (testTableData[0] + trainTableData[0], trainTableData[1])

        # Show tables, with train, test and train + test data
        fig, axs = plt.subplots(3, 1, figsize=(15, 10))

        for i, curTable in enumerate([trainTableData, testTableData, tableData]):
            axs[i].axis('tight')
            axs[i].axis('off')

            axs[i].table(cellText=curTable[0],
                         cellColours=curTable[1],
                         cellLoc='center',
                         rowLabels=["Predict: " + self.classes[0]] + self.classes[1:],
                         colLabels=["Labels: " + self.classes[0]] + self.classes[1:], loc='center')

            for j, image_class in enumerate(self.classes):
                # Print accuracy of every class
                classes_info = "(all: %d): %.3f" % (sum(curTable[0][:, j]), curTable[0][j, j] / sum(curTable[0][:, j]))
                print(image_class, classes_info.replace('.', ','))
            print('\n\n')

        plt.show()
