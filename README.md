# MovieLens Matrix-Factorization Recommender

A PyTorch matrix-factorization recommender for the MovieLens 1M dataset. The code implements the four Required Tasks in the CS550 project description (data split, rating prediction, top-K recommendation, interactive demo) and adds a long-tail fairness analysis on top: a post-hoc coverage rerank, a Pareto sweep of the fairness-accuracy trade-off, and a breakdown of how the rerank affects cold, warm, and heavy users differently.

## Quick start

```bash
python -m venv ~/.virtualenvs/movieLens
~/.virtualenvs/movieLens/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
~/.virtualenvs/movieLens/bin/pip install -r requirements.txt

cd movieLens
~/.virtualenvs/movieLens/bin/python -m code.main
~/.virtualenvs/movieLens/bin/streamlit run code/demo/app.py
```

`python -m code.main` reproduces every number and figure we cite, writing `results.json`, `pareto.png`, `user_groups.png`, and `mf_model.pt` to `../reports/` (outside the repo).

## The problem

MovieLens 1M is roughly one million 1-to-5 star ratings given by about 6,000 users to about 3,700 movies. The rating matrix is very sparse; most users have only rated a few dozen movies. A recommender has to fill in the blanks: predict how a user would rate the movies they haven't seen, and from that produce a ranked list of suggestions. We measure this two ways. Rating prediction asks how close the predicted star score is to the true held-out rating. Top-K recommendation asks whether the ten movies we suggest line up with the ones the user actually liked in the held-out set, with the hard rule that items the user already rated during training must be excluded from the list.

## How it's put together

Each module maps to a stage of the pipeline. `code/data.py` downloads MovieLens 1M and splits the ratings per user into 80% training, 10% validation, and 10% test. Splitting per user guarantees that every user appears in both the training and test sets, which is necessary for the per-user evaluation metrics to mean anything.

`code/model.py` defines the matrix-factorization model as a PyTorch `nn.Module`. Each user u gets a latent vector pᵤ of 50 numbers and a scalar bias bᵤ; each movie i gets a latent vector qᵢ and a bias bᵢ. A predicted rating is μ + bᵤ + bᵢ + ⟨pᵤ, qᵢ⟩, where μ is the global training mean. The biases capture "tough grader" and "generally well-liked" effects; the inner product captures the personalized match between a user's taste and a movie's features. Training (`code/train.py`) feeds batches of 4,096 (user, item, rating) triples through Adam, minimizing mean-squared error, and watches the validation loss to decide when to stop. Training converges in a handful of epochs on a single GPU and takes about a minute.

Three non-MF baselines sit alongside the main model in `code/baselines.py`. GlobalMean always predicts the training mean and serves as a floor. MostPopular ranks items by how often they were rated in training. UserCF (user-based collaborative filtering) predicts a user's rating for an item as a similarity-weighted average of the ratings given by the fifty most-similar users, using cosine similarity on mean-centered ratings.

Evaluation lives in `code/evaluation.py`. For rating prediction it reports MAE and RMSE. For top-K it generates the per-user top-10 on the GPU: it computes the full user-by-item score matrix in batches, sets the scores of items the user already rated in training to negative infinity so they cannot be selected, then takes the top-10. The ranking metrics are Precision@10, Recall@10, F1@10, and NDCG@10, treating a test rating of 4 or 5 as "relevant".

`code/fairness.py` implements the long-tail analysis. Items in the bottom 20% by training-rating count are tagged as tail. For a given coverage target t, the rerank routine walks each user's top-10 and, if fewer than t·10 tail items are present, swaps the lowest-scored non-tail items out for the highest-scored unseen tail items. The same module sweeps the rerank across six coverage targets to produce the Pareto curve, and partitions users by training activity (cold: fewer than 20 ratings, warm: 20 to 100, heavy: more than 100) to show how the rerank's accuracy cost falls across groups.

`code/main.py` is the orchestrator that runs every stage end to end. `code/demo/app.py` is the Streamlit demo: pick a user, choose K, and toggle the fairness rerank on and off to see how the top-K changes.

## Results

The matrix-factorization model wins on rating prediction with a test MAE around 0.68 and RMSE around 0.87, comfortably beating UserCF (0.74 / 0.95) and GlobalMean (0.93 / 1.12). It predicts star ratings within about two-thirds of a star on average, which is standard for MovieLens 1M.

The top-10 ranking story is more nuanced. Aggregated across all 6,000 users, MostPopular gets NDCG@10 of roughly 0.155, the matrix factorization gets 0.116, and UserCF gets about 0.004. MostPopular looking strong is not a bug; popular movies are, on average, the ones held out in the test set too, so a purely popularity-based recommender does well in aggregate once the exclude-rated rule is enforced. UserCF's low number reflects its known failure mode: items rated by only one or two of a user's fifty neighbors get extreme weighted-average scores and dominate the top of the list. When we break NDCG down by user activity level the picture flips for the users we can actually personalize for. Heavy users reach NDCG@10 of about 0.197 under matrix factorization, well above MostPopular's overall 0.155. Warm users sit around 0.07 and cold users around 0.05, which is where MostPopular has the edge because there is little signal to personalize on.

The fairness rerank does what it's supposed to. Asking for 25% long-tail coverage produces about 30% actual coverage (we overshoot slightly because swaps happen in integer chunks) and costs roughly 15% of NDCG. The Pareto plot in `reports/pareto.png` traces this trade-off smoothly across six coverage levels from 0 to 0.40, and the user-group plot in `reports/user_groups.png` shows that the NDCG cost is absorbed roughly proportionally by each group rather than concentrated on one.

The latent-factor ablation confirms 50 factors is a reasonable default: N=20 is a shade under-capacity, N=100 starts to overfit, and N=50 sits at the bottom of the validation-loss curve.

## Reproducibility

The seed (42) is set across Python's `random`, NumPy, and PyTorch including CUDA. The per-user split uses a seeded `np.random.default_rng`. CUDA is not put into fully deterministic mode, so individual ranking metrics can vary at the fourth decimal place from run to run; the picture above is stable.
