
# Content-Based Image Retrieval (CBIR) Using Deep Learning

This repository contains code and resources for a Content-Based Image Retrieval (CBIR) system, developed as part of a project using deep learning techniques. The system is designed to retrieve images similar to a query image from a large database based on their visual content. 

## Project Overview

This CBIR system leverages deep convolutional neural networks (DCNNs) to analyze and retrieve images from well-known datasets, including Oxbuild and Paris 6k. Key features are extracted, and dimensionality is reduced using Principal Component Analysis (PCA), resulting in an efficient representation of each image. The trained DCNN model achieved a retrieval accuracy of 92%.

## Key Features

- **Dataset Used**: Oxbuild and Paris 6k image datasets.
- **Feature Extraction**: Used DCNNs to extract meaningful features for similarity-based image retrieval.
- **Dimensionality Reduction**: Implemented PCA to reduce feature dimensionality, enhancing computational efficiency.
- **Model Training and Evaluation**: The DCNN model was trained and evaluated on the dataset, achieving 92% accuracy in retrieving relevant images based on content.

## Project Structure

- `Content_Based_Image_Retrieal_using_Deep_Learning.ipynb`: Main Jupyter Notebook containing all code for data loading, feature extraction, dimensionality reduction, model training, and evaluation.

## Requirements

To run this project, you'll need the following Python libraries:
- `TensorFlow` / `Keras`
- `scikit-learn`
- `numpy`
- `matplotlib`
- `opencv-python`

Install these dependencies using:
```bash
pip install tensorflow scikit-learn numpy matplotlib opencv-python
```

## Usage

1. **Load and Preprocess Dataset**: Load the Oxbuild and Paris 6k datasets into the environment.
2. **Feature Extraction**: Run the code to extract image features using DCNN.
3. **Dimensionality Reduction**: Apply PCA to reduce feature dimensionality, making the retrieval process efficient.
4. **Train the Model**: Train the DCNN model and evaluate its performance, achieving up to 92% retrieval accuracy.
5. **Retrieve Images**: Use the trained model to retrieve images based on the visual similarity of a query image.

## Results

The CBIR model achieved a retrieval accuracy of 92%, showcasing its effectiveness in accurately finding visually similar images in the dataset.

## Future Work

- Explore other dimensionality reduction techniques.
- Experiment with different CNN architectures to further improve accuracy.
- Expand the system to handle additional image datasets.

## Contributing

Feel free to open issues or submit pull requests for improvements and new features.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
