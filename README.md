# Movie Recommender System using Collaborative Filtering and Content-Based

This project is inspired by the recommendation system that we can see in our daily lives. I decided to try building a recommendation system using machine learning so I looked up resources online and decided to go with collaboraive filtering (CF), and content-based model to design the recommendation system. Each method has its own strengths and weakenesses, for instance, collaborative filtering would not work with new additions as the system can't create an embedding for it and can't query the model with this item.

1. Collaborative-filtering.py - The CF model in this project encompasses model-based CF and memory-based CF. The model-based CF uses Singular Value Decomposition, whereas the memory-based CF is build based on item-item and user-user CF.

2. Content-based.py - Generate recommendations using content-based method. Content-based filtering does not require other users' data during recommendations to one user. This is because it is based on the assumption that if a person likes item A, the person would most probably like items that are similar to item A.

3. app.py - The recommendation file that can simply be deployed using `python` by running the following code on the terminal:
    `python app.py`
   The model deployed in Flask is content-based.

4. Future works
  a. Using neural networks or other models to generate recommendations
  b. How can we provide recommendations that takes the current trend into consideration? Past reviews of users may not be relevant as trends may change over time.
  c. How do we tell when a user preference suddenly changes?
