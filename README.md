# **Candy Feature Regression Using CNN**

This project uses a 1D Convolutional Neural Network (CNN) to predict the "winpercent" of candies based on their features. By applying deep learning techniques, the model aims to uncover relationships between candy characteristics and their competitive performance.

## **Features**
- Data preprocessing with Min-Max scaling for normalization.
- Custom train-test split (80%-20%).
- Conv1D architecture for regression:
  - Feature extraction using convolutional layers.
  - Dense layers for final prediction.
- Visualization of training/validation loss and actual vs predicted values.

## **Dataset**
The dataset includes various features of candies, excluding `competitorname`. The target variable is `winpercent`.

## **Model Architecture**
- **Input Layer:** Accepts reshaped features `(num_samples, num_features, 1)`.
- **Conv1D Layer:** 32 filters, kernel size 2, ReLU activation.
- **Flatten Layer:** Converts the 2D output to a 1D array.
- **Dense Layers:**
  - One hidden layer with 64 units and ReLU activation.
  - One output layer with 1 unit for regression.

## **Performance**
- **Test Loss:** 135.22
- Plots for:
  - Training vs validation loss over 50 epochs.
  - Actual vs predicted `winpercent` values.

## **Contributing**
You are welcome to fork this repository and improve the model! Here are a few ideas for enhancements:
- Experiment with different architectures (e.g., more Conv1D layers or dropout layers).
- Use hyperparameter tuning to improve performance.
- Incorporate additional features or datasets for better predictions.
- Visualize results with more sophisticated techniques.

## **Technologies Used**
- **Python**: NumPy, Pandas
- **TensorFlow/Keras**: Model building and training
- **Matplotlib**: Visualization

## **Project Goal**
Demonstrate the use of CNNs for regression tasks and highlight their effectiveness in predicting continuous target values.

## License

This project is open-source and available under the [MIT License](LICENSE).

