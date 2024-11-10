# Book-Genre-Prediction-using-LSTM
This project leverages the power of LSTM to predict the genre of any book that the user wants , by entering a sentence from that book. 
My project of book genre prediction using deep learning involves training a model on text data (like book descriptions or sample passages) to classify books into genres. Here’s an overview of the main steps:
1.	Data Collection and Preprocessing: Text data from books, often descriptions or samples, is collected and cleaned. This involves tokenization, lowercasing, and removing non-essential words (like stop words).
2.	Model Design: A neural network model, often LSTM (Long Short-Term Memory), is built for text analysis. LSTMs handle sequence data well, capturing context across longer text passages by using past inputs to inform current processing.
3.	Training: The model is trained on labelled data with genre tags. It learns patterns in language and phrases that distinguish genres (e.g., horror, romance, sci-fi).
4.	Evaluation and Tuning: Model performance is evaluated using metrics like accuracy and F1-score. Hyperparameters, such as learning rate and dropout, are tuned to improve generalization and reduce overfitting.
5.	Deployment: Once optimized, the model can be deployed to predict genres for new books, aiding in recommendations or cataloguing systems.

I have hosted this project as a web page using Streamlit and ngrok. Streamlit is a powerful tool for building data-driven web applications quickly, especially for machine learning and data science projects. Ngrok is a tool that helps developers expose a local server to the internet through secure tunnels. This is especially useful for sharing a Streamlit or other local application for remote access without complex network configurations. Ngrok provides a temporary, public URL that can be shared, making it easy for collaborators or clients to interact with the application or test features without a full deployment.
