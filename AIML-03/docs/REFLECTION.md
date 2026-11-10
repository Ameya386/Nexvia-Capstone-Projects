# Individual Reflection - Ameya Kamble (AIML-03)

**What I owned.** I took the project from the brief to a working, evaluated system: I ran the full pipeline on Google Colab, debugged it, and checked the results. I worked with an AI assistant (Claude) to write the code, and I was responsible for running it, fixing environment problems (for example an SSL certificate error when downloading the dataset), and judging whether the results could be trusted.

**What I learned.**
- *How two-stage recommenders work.* A cheap retrieval model narrows millions of items to a few hundred candidates, and a heavier ranker re-orders only those. FAISS makes the first stage fast.
- *Why evaluation design matters more than the model.* My first split gave only 18 evaluation users, so the numbers were meaningless. Splitting each user's history chronologically fixed that. I also learned that splitting by time, not randomly, is what makes a recommender's evaluation honest.
- *Why a model can look fine in training and be worse in practice.* The first rankers were worse than retrieval-only because they trained on easy random negatives and memorised user/item IDs. Training on the retriever's own candidates (hard negatives) and using features instead of IDs made them better than retrieval-only.
- *Reporting honestly.* DCN scored higher than Wide&Deep in the A/B test, but the difference was not statistically significant (p = 0.136), so I report it as directional only.

**What I'd improve with more time.** Run the 25M-rating dataset with Spark, run Triton and Airflow for real instead of only exporting and writing them, replace the offline A/B replay with live traffic, and add richer ranker features (item content, recency).
