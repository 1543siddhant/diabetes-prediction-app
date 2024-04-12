# diabetes-prediction-app

Diabetes Prediction Model

Welcome to the Diabetes Prediction Model repository! This project focuses on leveraging machine learning algorithms to predict the likelihood of diabetes based on various health indicators. Here's a brief overview of what you'll find in this repository:

Overview:

Diabetes is a prevalent health condition affecting millions worldwide. Early detection and intervention are crucial for managing diabetes effectively. This project aims to provide a tool for predicting the risk of diabetes using machine learning techniques.

Data Set :
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

Acknowledgements
Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.

Data Exploration:
Exploratory Data Analysis (EDA) is conducted to understand the underlying patterns and relationships within the dataset. Visualizations such as histograms, box plots, and pair plots are used to explore the distribution of features and identify outliers.

Model Training Process:
Data Loading and Exploration:

The process starts with loading the diabetes dataset using Pandas, a popular data manipulation library in Python. Initial exploratory data analysis (EDA) is performed to understand the structure, distributions, and relationships within the dataset. Descriptive statistics, such as mean, median, min, max, etc., are computed for each feature to gain insights into the data's characteristics. Visualizations, including histograms, box plots, pair plots, and heatmaps, are created using libraries like Matplotlib and Seaborn to visualize the distribution of features and correlations between them.

Data Preprocessing:

The data preprocessing step involves preparing the dataset for training the machine learning model. Standardization using StandardScaler from scikit-learn is applied to scale the numerical features to have a mean of 0 and a standard deviation of 1. This step helps in improving the performance of certain machine learning algorithms. The dataset is split into input features (X) and target variable (y), where X contains the features used for prediction (e.g., glucose levels, BMI), and y contains the binary outcome variable (0 or 1) indicating the presence or absence of diabetes.

Model Selection and Evaluation:

The K-Nearest Neighbors (KNN) classifier is chosen as the machine learning algorithm for this task. KNN is a simple and effective classification algorithm that predicts the class of a data point based on the majority class of its k nearest neighbors. To determine the optimal value of k (number of neighbors), the code iterates through a range of k values and evaluates the classifier's performance using both training and testing datasets. This process helps prevent overfitting and underfitting by identifying the best trade-off between bias and variance. Model performance metrics such as accuracy, precision, recall, and F1-score are computed using the classification_report from scikit-learn. These metrics provide insights into the model's predictive power and its ability to correctly classify instances of diabetes.

Model Persistence:

Once the optimal model is trained and evaluated, it is serialized and saved to disk using the pickle module. Serialization allows the trained model to be stored in a binary format, making it easy to reload and reuse the model for future predictions without having to retrain it from scratch.

How to Use:
Clone the repository to your local machine.
Install the necessary dependencies listed in the requirements.txt file.
Run the Streamlit web application using the command streamlit run app.py.
Enter your health information and click the "Predict" button to receive your diabetes risk prediction.

Contributors:
Siddhant Dinesh Patil
