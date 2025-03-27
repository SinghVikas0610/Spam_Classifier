📩 SMS Spam Classifier
This project is a machine learning-based SMS spam classifier that analyzes text messages to determine whether they are spam or ham (not spam). It leverages data cleaning, NLP vectorization, and multiple ML models, including Naïve Bayes, to compare and achieve the best classification accuracy.

🚀 Features
✅ Ham-Spam Dataset preprocessing and cleaning
✅ NLP-based vectorization (TF-IDF, CountVectorizer) for text representation
✅ Multiple ML Models tested for accuracy comparison
✅ Performance evaluation with accuracy, precision, and recall metrics
✅ Interactive Streamlit Web App for real-time predictions

🛠️ Tech Stack
Python 🐍

scikit-learn (Machine Learning)

pandas & numpy (Data Processing)

NLTK & Text Processing (Natural Language Processing)

📊 Model Training & Evaluation
The dataset undergoes text cleaning (removal of stopwords, punctuation, lemmatization).

Various vectorization techniques (TF-IDF, CountVectorizer) are applied.

Multiple ML models are trained and evaluated, including:

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

Comparison of accuracy across models to find the best one.

🎯 Usage
Enter an SMS message or upload a file.

Click Predict to classify it as Spam or Ham.

View the classification results and model insights!

📜 Dataset
The model is trained on the Ham-Spam Dataset, widely used for spam detection research.

🤝 Contributing
Feel free to fork this repository and submit a pull request with improvements!
