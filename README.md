# Clustering-and-Classification-with-Deep-Learning-on-the-FashionMNIST-Dataset

**Setup** <br>
Environment: The project was developed using Google Colab with GPU support to accelerate computations.
Libraries: The necessary Python libraries were imported, including TensorFlow and Keras for deep learning tasks.

**Data Import and Preprocessing**<br>
Dataset: The Fashion MNIST dataset was imported and used for this project.
Preprocessing: Data preprocessing involved reshaping the dataset to the required dimensions of 28x28 pixels.

**CNN Model**<br>
Architecture: The project implemented a custom Convolutional Neural Network (CNN) for image classification. While specific architectural details are not provided in this README, please refer to the project's Jupyter notebooks for comprehensive model architecture and hyperparameter information.
Training
Model Training: The custom CNN model was trained on the Fashion MNIST dataset using the training data.

**Feature Extraction** <br>
Encoded Features: One of the fully connected hidden layers from the trained CNN model was extracted using the Keras Model class. This layer represents an encoded feature space.

**Visualization and Clustering** <br>
Principal Component Analysis (PCA): PCA was applied to visualize the encoded features in a reduced dimensionality space. The first two principal components were used for visualization, and data points were color-coded by label values.
Clustering: Clustering techniques, including DBScan and K-means, were evaluated. The resulting clusters were visualized using PCA plots, and noise was removed to improve the clarity of the results.
t-Distributed Stochastic Neighbor Embedding (t-SNE): t-SNE was applied to reduce the dimensionality of the encoded features to two components. The results were visualized alongside the original labels (y_test).
Visual Comparison: t-SNE results were also visualized with DBScan and K-means labels to understand how different clustering methods group data points.

![image](https://github.com/bhupeshdod/Clustering-and-Classification-with-Deep-Learning-on-the-FashionMNIST-Dataset/assets/141383468/a8829e05-8c40-4a1c-88f1-bf40fc6ab973)

| Label | Fashion Item             |
|-------|--------------------------|
| 0     | T-shirt/Top and Shirt    |
| 1     | Sandal                   |
| 2     | Pullover and Trouser     |
| 3     | Sneaker and Bag          |
| 4     | Dress, Ankle and Coat    |

**Guessing the Labels** <br>
K-means Clustering: Based on K-means clustering results, new labels were assigned to clusters formed by the algorithm.
Label Mapping: A mapping was provided to relate these cluster labels to fashion item labels, allowing for the interpretation of which items the model grouped together based on their visual similarities.

| Cluster Label | Actual Lael              | Fashion Item                              |
|---------------|--------------------------|-------------------------------------------|
| Cluster 0     | Label - 2                | Trouser                                   |
| Cluster 1     | Label - 4                | Ankle Boot                                |
| Cluster 2     | Label - 0                | T-Shirt/Top                               |
| Cluster 3     | Label - 0, 2, 4          | Shirt, Dress, Pullover, and Coat          |
| Cluster 4     | Label - 1, 3             | Sandal, Sneaker, and Bag                  |

**Conclusion** <br>
This project utilized deep learning techniques, dimensionality reduction, and clustering to analyze and interpret the Fashion MNIST dataset. The extracted encoded features were visualized and clustered to gain insights into the data's structure. By assigning new labels to clusters, the project provided valuable information about how the model perceived and grouped fashion items.

Feel free to explore the project's Jupyter notebooks for detailed code implementations, model architectures, and visualization results. Continuously experimenting and refining the approach can lead to a deeper understanding of the dataset and further improvements in model performance.

