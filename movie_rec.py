#!/usr/bin/env python3
"""
CS550 Final Project: Matrix Factorization Recommender System with Fairness

This script implements a complete recommendation pipeline:
1. Train a Matrix Factorization model using SGD
2. Evaluate on rating prediction and ranking tasks
3. Apply fairness-aware reranking to improve long-tail item coverage
4. Analyze the fairness-accuracy trade-off

Author: CS550 Student
Date: April 2026
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

# ============================================================================
# CONFIGURATION
# ============================================================================

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Model hyperparameters
N_FACTORS = 50              # Latent factor dimension
LEARNING_RATE = 0.005       # SGD learning rate
REGULARIZATION = 0.01       # L2 regularization coefficient
EPOCHS = 10                 # Maximum training epochs
PATIENCE = 3                # Early stopping: stop after N epochs of no improvement

# Fairness parameters
TAIL_PERCENTILE = 20        # Bottom 20% of items by rating frequency = long-tail
TARGET_COVERAGE = 0.25      # Target: 25% long-tail items in recommendations
K = 10                       # Recommendation list length (top-10)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
ML_1M_DIR = DATA_DIR / "ml-1m"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def download_movielens():
    """
    Download MovieLens 1M dataset if not already present.
    
    MovieLens 1M contains:
    - 1,000,209 ratings from 6,040 users on 3,952 movies
    - Ratings are on a 1-5 scale
    - Data spans from 1995 to 2003
    """
    if (ML_1M_DIR / "ratings.dat").exists():
        print(f"✓ MovieLens 1M already downloaded at {ML_1M_DIR}")
        return
    
    print("Downloading MovieLens 1M dataset...")
    url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = DATA_DIR / "ml-1m.zip"
    
    # Download
    urlretrieve(url, zip_path)
    
    # Extract
    with ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    
    # Cleanup
    zip_path.unlink()
    print(f"✓ Downloaded to {ML_1M_DIR}")


def load_movielens():
    """
    Load and preprocess MovieLens 1M dataset.
    
    Returns:
        DataFrame with columns: user_id, item_id, rating
    """
    # Read the ratings file
    # Format: UserID::MovieID::Rating::Timestamp
    ratings = pd.read_csv(
        ML_1M_DIR / "ratings.dat",
        sep="::",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": np.int32, "item_id": np.int32, "rating": np.float32}
    )
    
    # Keep only the ratings we need
    return ratings[["user_id", "item_id", "rating"]]


# ============================================================================
# MATRIX FACTORIZATION MODEL
# ============================================================================

def train_matrix_factorization(train_data, val_data):
    """
    Train a Matrix Factorization model using Stochastic Gradient Descent (SGD).
    
    The model learns embeddings to predict ratings:
    r_ui ≈ μ + b_u + b_i + <p_u, q_i>
    
    Where:
    - μ: global mean rating
    - b_u: user bias (users who tend to rate high/low)
    - b_i: item bias (items that are generally good/bad)
    - p_u, q_i: user and item latent factors (learned embeddings)
    
    Args:
        train_data: DataFrame with columns [user_id, item_id, rating]
        val_data: DataFrame for validation (same format)
    
    Returns:
        Dictionary containing trained model parameters
    """
    print("\n" + "="*70)
    print("Training Matrix Factorization Model")
    print("="*70)
    
    # Extract metadata
    n_users = int(train_data["user_id"].max()) + 1
    n_items = int(train_data["item_id"].max()) + 1
    global_mean = train_data["rating"].mean()
    
    print(f"Users: {n_users} | Items: {n_items}")
    print(f"Global mean rating: {global_mean:.4f}")
    
    # Initialize model parameters
    # Embeddings initialized from N(0, 0.01) for stability
    user_embeddings = np.random.normal(0, 0.01, size=(n_users, N_FACTORS))
    item_embeddings = np.random.normal(0, 0.01, size=(n_items, N_FACTORS))
    user_bias = np.zeros(n_users)
    item_bias = np.zeros(n_items)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Shuffle training data
        train_shuffled = train_data.sample(frac=1, random_state=SEED + epoch)
        
        # SGD: update parameters for each (user, item, rating) triplet
        train_loss_sum = 0
        for _, row in train_shuffled.iterrows():
            u = int(row["user_id"])
            i = int(row["item_id"])
            r = row["rating"]
            
            # Compute prediction
            pred = (global_mean + 
                   user_bias[u] + 
                   item_bias[i] + 
                   np.dot(user_embeddings[u], item_embeddings[i]))
            
            # Clip to valid rating range [1, 5]
            pred = np.clip(pred, 1, 5)
            
            # Compute error
            error = pred - r
            train_loss_sum += error ** 2
            
            # Update parameters using gradient descent
            # Gradient for each parameter is: error * (partial derivative)
            
            # Bias updates
            user_bias[u] -= LEARNING_RATE * (error + REGULARIZATION * user_bias[u])
            item_bias[i] -= LEARNING_RATE * (error + REGULARIZATION * item_bias[i])
            
            # Embedding updates
            user_embeddings[u] -= LEARNING_RATE * (error * item_embeddings[i] + 
                                                   REGULARIZATION * user_embeddings[u])
            item_embeddings[i] -= LEARNING_RATE * (error * user_embeddings[u] + 
                                                   REGULARIZATION * item_embeddings[i])
        
        # Compute average training loss
        train_loss = train_loss_sum / len(train_shuffled)
        
        # Compute validation loss
        val_loss_sum = 0
        for _, row in val_data.iterrows():
            u = int(row["user_id"])
            i = int(row["item_id"])
            r = row["rating"]
            
            pred = (global_mean + 
                   user_bias[u] + 
                   item_bias[i] + 
                   np.dot(user_embeddings[u], item_embeddings[i]))
            pred = np.clip(pred, 1, 5)
            error = pred - r
            val_loss_sum += error ** 2
        
        val_loss = val_loss_sum / len(val_data)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if patience_counter >= PATIENCE:
            print(f"  → Early stopping at epoch {epoch+1}")
            break
    
    # Return trained model
    return {
        "global_mean": global_mean,
        "user_embeddings": user_embeddings,
        "item_embeddings": item_embeddings,
        "user_bias": user_bias,
        "item_bias": item_bias,
        "n_users": n_users,
        "n_items": n_items
    }


# ============================================================================
# EVALUATION: RATING PREDICTION
# ============================================================================

def evaluate_rating_prediction(model, test_data):
    """
    Evaluate the model's rating prediction accuracy using MAE and RMSE.
    
    MAE (Mean Absolute Error):
    - Average absolute difference between predicted and actual ratings
    - Range: [0, 4] (for 1-5 ratings)
    - Lower is better
    
    RMSE (Root Mean Squared Error):
    - Square root of average squared errors
    - Penalizes large errors more than MAE
    - Range: [0, 4]
    - Lower is better
    
    Args:
        model: Trained model dictionary
        test_data: Test set DataFrame
    
    Returns:
        Dictionary with MAE and RMSE
    """
    user_embeddings = model["user_embeddings"]
    item_embeddings = model["item_embeddings"]
    user_bias = model["user_bias"]
    item_bias = model["item_bias"]
    global_mean = model["global_mean"]
    
    mae_sum = 0
    rmse_sum = 0
    
    for _, row in test_data.iterrows():
        u = int(row["user_id"])
        i = int(row["item_id"])
        r = row["rating"]
        
        # Predict rating
        pred = (global_mean + 
               user_bias[u] + 
               item_bias[i] + 
               np.dot(user_embeddings[u], item_embeddings[i]))
        pred = np.clip(pred, 1, 5)
        
        # Accumulate error
        error = abs(pred - r)
        mae_sum += error
        rmse_sum += error ** 2
    
    mae = mae_sum / len(test_data)
    rmse = np.sqrt(rmse_sum / len(test_data))
    
    return {"MAE": mae, "RMSE": rmse}


# ============================================================================
# EVALUATION: RANKING QUALITY
# ============================================================================

def generate_recommendations(model, n_users, k=K):
    """
    Generate top-k recommendations for each user using the trained model.
    
    For each user, we:
    1. Score all items using the MF model
    2. Select the top-k highest-scoring items
    
    Args:
        model: Trained model dictionary
        n_users: Number of users
        k: Recommendation list length (default: 10)
    
    Returns:
        Dictionary mapping user_id -> list of recommended item_ids
    """
    user_embeddings = model["user_embeddings"]
    item_embeddings = model["item_embeddings"]
    user_bias = model["user_bias"]
    item_bias = model["item_bias"]
    global_mean = model["global_mean"]
    n_items = model["n_items"]
    
    recommendations = {}
    
    for u in range(n_users):
        # Score all items for this user
        scores = np.zeros(n_items)
        for i in range(n_items):
            scores[i] = (global_mean + 
                        user_bias[u] + 
                        item_bias[i] + 
                        np.dot(user_embeddings[u], item_embeddings[i]))
        
        # Get top-k item indices (highest scores)
        top_k_indices = np.argsort(-scores)[:k]  # Negative for descending order
        recommendations[u] = top_k_indices
    
    return recommendations


def evaluate_ranking_quality(recommendations, test_data, k=K):
    """
    Evaluate recommendation quality using ranking metrics.
    
    Metrics:
    - Precision@k: % of recommended items the user actually liked
    - Recall@k: % of items user liked that were recommended
    - NDCG@k: Normalized Discounted Cumulative Gain
      * Accounts for ranking order (top items weighted more)
      * Popular metric for ranking evaluation
    
    Args:
        recommendations: Dict mapping user_id -> recommended items
        test_data: Test set DataFrame
        k: Evaluation cutoff (default: 10)
    
    Returns:
        Dictionary with metrics
    """
    # Build user-item test matrix
    user_test_items = {}
    for _, row in test_data.iterrows():
        u = int(row["user_id"])
        i = int(row["item_id"])
        if u not in user_test_items:
            user_test_items[u] = set()
        user_test_items[u].add(i)
    
    precision_sum = 0
    recall_sum = 0
    ndcg_sum = 0
    count = 0
    
    for u, rec_items in recommendations.items():
        # Get items user actually rated in test set
        test_items = user_test_items.get(u, set())
        
        if len(test_items) == 0:
            continue  # Skip users with no test items
        
        # Compute Precision@k
        hits = len(set(rec_items[:k]) & test_items)
        precision = hits / k
        precision_sum += precision
        
        # Compute Recall@k
        recall = hits / len(test_items)
        recall_sum += recall
        
        # Compute NDCG@k
        # DCG@k = sum(relevance_i / log2(i+1))
        # IDCG@k = sum(1 / log2(i+1)) for first |test_items| items
        dcg = 0
        for rank, item in enumerate(rec_items[:k]):
            if item in test_items:
                dcg += 1 / np.log2(rank + 2)  # rank+2 because positions are 1-indexed
        
        idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(test_items))))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_sum += ndcg
        
        count += 1
    
    avg_precision = precision_sum / count if count > 0 else 0
    avg_recall = recall_sum / count if count > 0 else 0
    avg_ndcg = ndcg_sum / count if count > 0 else 0
    
    return {
        "Precision@10": avg_precision,
        "Recall@10": avg_recall,
        "NDCG@10": avg_ndcg
    }


# ============================================================================
# FAIRNESS: TAIL ITEM IDENTIFICATION
# ============================================================================

def identify_tail_items(train_data, percentile=TAIL_PERCENTILE):
    """
    Identify long-tail items using rating frequency.
    
    Items are ranked by how many ratings they received:
    - Head items: frequently rated (popular movies)
    - Tail items: rarely rated (niche movies)
    
    We define tail as the bottom PERCENTILE% (e.g., bottom 20%) by count.
    
    Args:
        train_data: Training set DataFrame
        percentile: Percentile threshold (default: 20)
    
    Returns:
        Set of item IDs in the long-tail
    """
    # Count ratings per item
    item_counts = train_data["item_id"].value_counts()
    
    # Find the percentile threshold
    threshold = np.percentile(item_counts.values, percentile)
    
    # Items below threshold are tail items
    tail_items = set(item_counts[item_counts <= threshold].index)
    
    return tail_items


# ============================================================================
# FAIRNESS: RERANKING ALGORITHM
# ============================================================================

def rerank_for_fairness(recommendations, model, tail_items, target_coverage=TARGET_COVERAGE, k=K):
    """
    Apply fairness-aware reranking to improve long-tail item coverage.
    
    Algorithm:
    1. For each user's recommendation list
    2. Count how many tail items are currently recommended
    3. If coverage < target:
       - Find best-scoring tail items not in top-k
       - Swap lowest-scored popular items for these tail items
    
    This is efficient (no retraining) and smart (preserves quality).
    
    Args:
        recommendations: Dict mapping user_id -> recommended items
        model: Trained model dictionary
        tail_items: Set of tail item IDs
        target_coverage: Target fraction of tail items (default: 0.25 = 25%)
        k: Recommendation list length (default: 10)
    
    Returns:
        Reranked recommendations dict
    """
    user_embeddings = model["user_embeddings"]
    item_embeddings = model["item_embeddings"]
    user_bias = model["user_bias"]
    item_bias = model["item_bias"]
    global_mean = model["global_mean"]
    
    reranked = {}
    
    for u, rec_items in recommendations.items():
        rec_items = list(rec_items)  # Make mutable
        
        # Count current tail items
        current_tail = sum(1 for i in rec_items if i in tail_items)
        target_tail = int(np.ceil(k * target_coverage))
        
        # If we already meet target, keep as is
        if current_tail >= target_tail:
            reranked[u] = rec_items
            continue
        
        # Otherwise, find tail items to add
        needed = target_tail - current_tail
        rec_set = set(rec_items)
        
        # Find tail items not currently recommended
        available_tail = [i for i in tail_items if i not in rec_set]
        
        # Score them
        available_scored = []
        for i in available_tail:
            score = (global_mean + 
                    user_bias[u] + 
                    item_bias[i] + 
                    np.dot(user_embeddings[u], item_embeddings[i]))
            available_scored.append((i, score))
        
        # Sort by score descending
        available_scored.sort(key=lambda x: x[1], reverse=True)
        
        # Find popular items currently recommended
        popular_current = [i for i in rec_items if i not in tail_items]
        
        # Score them
        popular_scored = []
        for i in popular_current:
            score = (global_mean + 
                    user_bias[u] + 
                    item_bias[i] + 
                    np.dot(user_embeddings[u], item_embeddings[i]))
            idx = rec_items.index(i)
            popular_scored.append((idx, score))
        
        # Sort by score ascending (we'll remove lowest-scored)
        popular_scored.sort(key=lambda x: x[1])
        
        # Swap: replace lowest-scored popular with highest-scored tail
        for j in range(min(needed, len(available_scored))):
            if j < len(popular_scored):
                pop_idx, _ = popular_scored[j]
                tail_item, _ = available_scored[j]
                rec_items[pop_idx] = tail_item
        
        reranked[u] = rec_items
    
    return reranked


# ============================================================================
# FAIRNESS: EVALUATION
# ============================================================================

def compute_tail_coverage(recommendations, tail_items, k=K):
    """
    Compute the fraction of recommendations that are long-tail items.
    
    coverage = (# tail items recommended) / (total recommendations)
    
    Higher coverage = more fair (better visibility for niche items)
    
    Args:
        recommendations: Dict mapping user_id -> recommended items
        tail_items: Set of tail item IDs
        k: Recommendation list length (default: 10)
    
    Returns:
        Float between 0 and 1
    """
    total_recs = 0
    tail_recs = 0
    
    for rec_items in recommendations.values():
        total_recs += k
        tail_recs += sum(1 for i in rec_items[:k] if i in tail_items)
    
    return tail_recs / total_recs if total_recs > 0 else 0


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Execute the complete recommendation pipeline:
    1. Load and split data
    2. Train Matrix Factorization
    3. Evaluate rating prediction
    4. Generate and evaluate recommendations
    5. Apply fairness reranking
    6. Analyze trade-offs
    7. Save results
    """
    
    print("\n" + "="*70)
    print("CS550: Matrix Factorization Recommender + Fairness")
    print("="*70)
    
    # =====================================================================
    # [1/6] LOAD DATA
    # =====================================================================
    print("\n[1/6] Loading MovieLens 1M dataset...")
    download_movielens()
    ratings = load_movielens()
    
    print(f"  Dataset loaded:")
    print(f"  Ratings: {len(ratings):,}")
    print(f"  Users: {ratings['user_id'].max() + 1:,}")
    print(f"  Items: {ratings['item_id'].max() + 1:,}")
    sparsity = 1 - len(ratings) / ((ratings['user_id'].max() + 1) * (ratings['item_id'].max() + 1))
    print(f"  Sparsity: {sparsity:.2%}")
    
    # =====================================================================
    # [2/6] SPLIT DATA (per-user 60/20/20)
    # =====================================================================
    print("\n[2/6] Splitting data (per-user 60/20/20)...")
    
    train_data = []
    val_data = []
    test_data = []
    
    for user_id in ratings["user_id"].unique():
        user_ratings = ratings[ratings["user_id"] == user_id].sample(
            frac=1, random_state=SEED
        )
        
        n = len(user_ratings)
        train_idx = int(0.6 * n)
        val_idx = int(0.8 * n)
        
        train_data.append(user_ratings.iloc[:train_idx])
        val_data.append(user_ratings.iloc[train_idx:val_idx])
        test_data.append(user_ratings.iloc[val_idx:])
    
    train_data = pd.concat(train_data, ignore_index=True)
    val_data = pd.concat(val_data, ignore_index=True)
    test_data = pd.concat(test_data, ignore_index=True)
    
    print(f"  Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")
    
    # =====================================================================
    # [3/6] TRAIN MATRIX FACTORIZATION
    # =====================================================================
    model = train_matrix_factorization(train_data, val_data)
    
    # =====================================================================
    # [4/6] EVALUATE RATING PREDICTION
    # =====================================================================
    print("\n[4/6] Evaluating rating prediction...")
    rating_metrics = evaluate_rating_prediction(model, test_data)
    print(f"  MAE: {rating_metrics['MAE']:.4f}")
    print(f"  RMSE: {rating_metrics['RMSE']:.4f}")
    
    # =====================================================================
    # [5/6] EVALUATE RECOMMENDATIONS
    # =====================================================================
    print("\n[5/6] Generating recommendations...")
    
    # Standard MF recommendations
    n_users = model["n_users"]
    recommendations_mf = generate_recommendations(model, n_users, k=K)
    
    # Most-Popular baseline
    item_counts = train_data["item_id"].value_counts()
    popular_items = item_counts.head(K).index.tolist()
    recommendations_popular = {u: popular_items for u in range(n_users)}
    
    # Evaluate
    print("  Most-Popular: ", end="", flush=True)
    metrics_popular = evaluate_ranking_quality(recommendations_popular, test_data, k=K)
    print(f"P@10={metrics_popular['Precision@10']:.4f}, NDCG@10={metrics_popular['NDCG@10']:.4f}")
    
    print("  MF (Std): ", end="", flush=True)
    metrics_mf = evaluate_ranking_quality(recommendations_mf, test_data, k=K)
    print(f"P@10={metrics_mf['Precision@10']:.4f}, NDCG@10={metrics_mf['NDCG@10']:.4f}")
    
    # =====================================================================
    # [6/6] FAIRNESS ANALYSIS
    # =====================================================================
    print("\n[6/6] Fairness analysis...")
    
    # Identify tail items
    tail_items = identify_tail_items(train_data, percentile=TAIL_PERCENTILE)
    print(f"  Tail items (bottom {TAIL_PERCENTILE}%): {len(tail_items)}")
    
    # Compute coverage before reranking
    coverage_popular = compute_tail_coverage(recommendations_popular, tail_items, k=K)
    coverage_mf = compute_tail_coverage(recommendations_mf, tail_items, k=K)
    
    print(f"  Most-Popular tail coverage: {coverage_popular:.1%}")
    print(f"  MF (Std) tail coverage: {coverage_mf:.1%} (target: {TARGET_COVERAGE:.0%})")
    
    # Apply fairness reranking
    print("  Reranking for fairness...")
    recommendations_fair = rerank_for_fairness(recommendations_mf, model, tail_items, 
                                              target_coverage=TARGET_COVERAGE, k=K)
    
    # Evaluate fair recommendations
    metrics_fair = evaluate_ranking_quality(recommendations_fair, test_data, k=K)
    coverage_fair = compute_tail_coverage(recommendations_fair, tail_items, k=K)
    
    print(f"  Fair MF tail coverage: {coverage_fair:.1%}")
    print(f"  Fair MF: P@10={metrics_fair['Precision@10']:.4f}, NDCG@10={metrics_fair['NDCG@10']:.4f}")
    
    # =====================================================================
    # TRADE-OFF ANALYSIS
    # =====================================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    accuracy_loss = metrics_mf['NDCG@10'] - metrics_fair['NDCG@10']
    fairness_gain = coverage_fair - coverage_mf
    trade_off_ratio = fairness_gain / accuracy_loss if accuracy_loss > 0 else float('inf')
    
    print("\nMethod               MAE        RMSE       P@10       NDCG@10    Tail%")
    print("-"*70)
    print(f"Most-Popular         -          -          {metrics_popular['Precision@10']:.4f}     {metrics_popular['NDCG@10']:.4f}     {coverage_popular:.1%}")
    print(f"MF (Standard)        {rating_metrics['MAE']:.4f}     {rating_metrics['RMSE']:.4f}     {metrics_mf['Precision@10']:.4f}     {metrics_mf['NDCG@10']:.4f}     {coverage_mf:.1%}")
    print(f"MF (Fair)            -          -          {metrics_fair['Precision@10']:.4f}     {metrics_fair['NDCG@10']:.4f}     {coverage_fair:.1%}")
    
    print("\nFairness-Accuracy Trade-off:")
    print(f"  Accuracy loss: {accuracy_loss:.4f} ({accuracy_loss/metrics_mf['NDCG@10']*100:.1f}%)")
    print(f"  Fairness gain: {fairness_gain:.1%}")
    print(f"  Trade-off ratio: {trade_off_ratio:.1f}x fairness gain per 1% accuracy loss")
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    results = {
        "dataset": {
            "users": model["n_users"],
            "items": model["n_items"],
            "ratings": len(train_data) + len(val_data) + len(test_data)
        },
        "rating_prediction": rating_metrics,
        "recommendations": {
            "most_popular": metrics_popular,
            "mf_standard": metrics_mf,
            "mf_fair": metrics_fair
        },
        "fairness": {
            "most_popular_coverage": float(coverage_popular),
            "mf_standard_coverage": float(coverage_mf),
            "mf_fair_coverage": float(coverage_fair),
            "target_coverage": float(TARGET_COVERAGE),
            "accuracy_loss": float(accuracy_loss),
            "fairness_gain": float(fairness_gain),
            "fairness_gain_per_accuracy_loss": float(trade_off_ratio)
        }
    }
    
    with open(REPORTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {REPORTS_DIR / 'results.json'}")
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
