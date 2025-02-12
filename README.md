# NLP--Sentiment_Analysis_and_Summarization_of_Stock_News
This notebook uses a Natural Language Processing AI-model driven sentiment analysis solution that can process and analyze news articles to gauge market sentiment predicting stock price and volume. It will also summarize lengthy news at a weekly level to enhance the accuracy of their stock price predictions to optimize investment strategies.  GloVe, Word2Vec and Transformer models will be compared for accuracy and will be fine-tuned. 

I experimented with specific configurations to optimize each model's initial tuning parameters after training the models using different classifiers. 
Additionally, I used different available TPU's and GPUs to determine processing tradeoffs with cost and speed: I used the following libraries. 

* 1-To manipulate and analyze data: pandas, numpy. 
* 2-To visualize data: matplotlib.pyplot, seaborn. 
* 3-To parse JSON data: json. 
* 4-To build, tune, and evaluate ML models: 
    sklearn.ensemble: GradientBoostingClassifier, RandomForestClassifier, DecisionTreeClassifier sklearn.model_selection: GridSearchCV, sklearn.metrics: confusion_matrix, accuracy_score, f1_score, precision_score, recall_score. 
* 5-To load/create word embeddings: gensim.models, Word2Vec; KeyedVectors, gensim.scripts.glove2word2vec, glove2 word2vec. 
* 6-To work with transformer models: torch, sentence_transformers 
* 7-To summarize with NLP models: Llama Mistral-7B max_tokens, temperature, top_p, top_k.

![Models TradeOffs](https://github.com/user-attachments/assets/43fd5d1a-1f52-4239-a45b-b24f2611ba03)

Best Model Selection:
Tuning the model Word2Vec with a Decision Tree Classifier gave us comparable performance metrics as using a non-tuned Sentence Transformer Model:
* ->Model: Tuned Word2Vec	->Accuracy: 0.48	->F1-Score: 0.48
* ->Model: Non-Tuned Sentence Transformer	->Accuracy: 0.52	->F1-Score: 0.48

However, the TPU/GPU processing is much higher so for cost considerations, Tuned Word2Vec may be more economical.

![final_model_selection](https://github.com/user-attachments/assets/0a976fb9-7a04-4687-9d24-ab561d5c125e)

With the second part of this project, I used LLama & Mistral-7B. Note: Llama models come in various sizes, including larger ones like Llama 2 70B. Mistral-7B often outperforms larger Llama models in certain tasks despite its smaller size. 
For the news summarizations, I had to monitor the performance and GPU utilization with the following configurations:

![Llama Model Setup](https://github.com/user-attachments/assets/28b07f25-ceab-4400-a643-4120b724ef84)

This is an example of the input to be processed by the model:

![input](https://github.com/user-attachments/assets/e0a80ed1-4a45-4553-b9a8-19929de16a8c)

These are examples of the resulting outputs with a summary extraction, keywords, topics, stock value and price after summarizing the news input above when you enter a specic date (interactive mode):

![output-1](https://github.com/user-attachments/assets/b7e9cc0c-3518-4206-8ad2-de126e4f3084)

And this is an example when you just need a summary for the top 3 positive/negative events per week:

![output-2](https://github.com/user-attachments/assets/e56d419b-28c4-4f91-bba4-1255ee0f8ada)

Summary of my learnings:

Model Development and Hardware Optimization:
- Explored various classifier configurations and tuning parameters
- Evaluated performance trade-offs across different TPUs and GPUs, considering both cost and processing speed

Data Processing and Analysis:
- Pandas and NumPy for data manipulation and numerical operations
- JSON parsing for structured data handling
- Matplotlib and Seaborn for data visualization

Machine Learning Framework (scikit-learn):
- Ensemble Methods:
  - Gradient Boosting Classifier
  - Random Forest Classifier
  - Decision Tree Classifier
- Model Optimization:
  - GridSearchCV for hyperparameter tuning
- Evaluation Metrics:
  - Confusion Matrix
  - Accuracy Score
  - F1 Score
  - Precision Score
  - Recall Score

Natural Language Processing Tools:
- Word Embeddings:
  - Gensim's Word2Vec
  - GloVe (with glove2word2vec conversion)
  - KeyedVectors for embedding management
- Transformer Models:
  - PyTorch
  - Sentence Transformers
- Large Language Models:
  - Llama
  - Mistral-7B with configurable parameters:
    - Maximum tokens
    - Temperature
    - Top-p sampling
    - Top-k sampling







