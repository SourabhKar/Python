# Advanced Topics Deep Dive 🔬
## Comprehensive Guide to Specialized Data Science Domains

---

## 📑 Table of Contents
1. [Low-Level System Design](#low-level-system-design)
2. [Hyperparameter Tuning Advanced Techniques](#hyperparameter-tuning)
3. [Semi-Supervised Learning](#semi-supervised-learning)
4. [Scaling & Optimization Strategies](#scaling-optimization)
5. [LLM Evaluation & Monitoring](#llm-evaluation)
6. [Agentic AI Architecture](#agentic-ai)
7. [Production MLOps Best Practices](#production-mlops)
8. [Error Handling & Robustness](#error-handling)

---

## 1. Low-Level System Design

### Design Principles for ML Systems

#### SOLID Principles Applied to ML
**Single Responsibility Principle**
```python
# Bad: Class doing too much
class ModelPipeline:
    def load_data(self): pass
    def preprocess(self): pass
    def train(self): pass
    def evaluate(self): pass
    def deploy(self): pass

# Good: Separate responsibilities
class DataLoader:
    def load(self, source): pass

class Preprocessor:
    def transform(self, data): pass

class ModelTrainer:
    def fit(self, data): pass

class ModelEvaluator:
    def evaluate(self, model, data): pass

class ModelDeployer:
    def deploy(self, model): pass
```

**Open/Closed Principle**
```python
from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, data):
        pass

class TextFeatureExtractor(FeatureExtractor):
    def extract(self, data):
        # TF-IDF, embeddings, etc.
        return features

class ImageFeatureExtractor(FeatureExtractor):
    def extract(self, data):
        # CNN features, SIFT, etc.
        return features

# Can add new extractors without modifying existing code
```

**Liskov Substitution Principle**
```python
class Model(ABC):
    @abstractmethod
    def predict(self, X):
        pass

class LinearRegression(Model):
    def predict(self, X):
        return self.coef_ @ X + self.intercept_

class RandomForest(Model):
    def predict(self, X):
        return self.ensemble.predict(X)

# Any model can be substituted without breaking code
def batch_predict(model: Model, batches):
    return [model.predict(batch) for batch in batches]
```

**Interface Segregation Principle**
```python
# Bad: Fat interface
class MLModel:
    def train(self): pass
    def predict(self): pass
    def explain(self): pass
    def deploy(self): pass

# Good: Segregated interfaces
class Trainable(ABC):
    @abstractmethod
    def train(self, X, y): pass

class Predictable(ABC):
    @abstractmethod
    def predict(self, X): pass

class Explainable(ABC):
    @abstractmethod
    def explain(self, X): pass

# Models implement only what they need
class SimpleModel(Trainable, Predictable):
    def train(self, X, y): pass
    def predict(self, X): pass

class AdvancedModel(Trainable, Predictable, Explainable):
    def train(self, X, y): pass
    def predict(self, X): pass
    def explain(self, X): pass
```

**Dependency Inversion Principle**
```python
# Bad: High-level depends on low-level
class ModelTrainer:
    def __init__(self):
        self.data_loader = CSVDataLoader()  # Concrete dependency

# Good: Depend on abstractions
class DataLoader(ABC):
    @abstractmethod
    def load(self): pass

class ModelTrainer:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader  # Abstract dependency

# Can inject any data loader
trainer = ModelTrainer(CSVDataLoader())
trainer = ModelTrainer(DatabaseLoader())
trainer = ModelTrainer(APILoader())
```

### Essential Design Patterns for ML

#### 1. Strategy Pattern (Algorithm Selection)
```python
class FeatureScaler(ABC):
    @abstractmethod
    def scale(self, data): pass

class StandardScaler(FeatureScaler):
    def scale(self, data):
        return (data - data.mean()) / data.std()

class MinMaxScaler(FeatureScaler):
    def scale(self, data):
        return (data - data.min()) / (data.max() - data.min())

class FeatureProcessor:
    def __init__(self, scaler: FeatureScaler):
        self.scaler = scaler
    
    def process(self, data):
        return self.scaler.scale(data)

# Use different strategies at runtime
processor = FeatureProcessor(StandardScaler())
processor = FeatureProcessor(MinMaxScaler())
```

#### 2. Factory Pattern (Model Creation)
```python
class ModelFactory:
    @staticmethod
    def create_model(model_type: str, **kwargs):
        if model_type == "linear":
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**kwargs)
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**kwargs)
        elif model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Easy model creation
model = ModelFactory.create_model("xgboost", n_estimators=100)
```

#### 3. Observer Pattern (Training Monitoring)
```python
class TrainingObserver(ABC):
    @abstractmethod
    def update(self, epoch, metrics): pass

class LoggingObserver(TrainingObserver):
    def update(self, epoch, metrics):
        print(f"Epoch {epoch}: {metrics}")

class MLflowObserver(TrainingObserver):
    def update(self, epoch, metrics):
        mlflow.log_metrics(metrics, step=epoch)

class Trainer:
    def __init__(self):
        self.observers = []
    
    def attach(self, observer: TrainingObserver):
        self.observers.append(observer)
    
    def train(self, epochs):
        for epoch in range(epochs):
            metrics = self._train_epoch()
            self._notify(epoch, metrics)
    
    def _notify(self, epoch, metrics):
        for observer in self.observers:
            observer.update(epoch, metrics)
```

#### 4. Singleton Pattern (Model Registry)
```python
class ModelRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.models = {}
        return cls._instance
    
    def register(self, name, model):
        self.models[name] = model
    
    def get(self, name):
        return self.models.get(name)

# Single global registry
registry1 = ModelRegistry()
registry2 = ModelRegistry()
assert registry1 is registry2  # Same instance
```

#### 5. Pipeline Pattern (Data Processing)
```python
from typing import List, Callable

class Pipeline:
    def __init__(self, steps: List[Callable]):
        self.steps = steps
    
    def execute(self, data):
        result = data
        for step in self.steps:
            result = step(result)
        return result

# Create processing pipeline
pipeline = Pipeline([
    lambda x: x.dropna(),
    lambda x: x.fillna(0),
    lambda x: (x - x.mean()) / x.std(),
    lambda x: x.values
])

processed_data = pipeline.execute(raw_data)
```

### Caching Strategies

#### 1. LRU Cache for Expensive Operations
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def compute_expensive_features(data_hash):
    # Expensive feature computation
    return features

# Automatic caching of results
```

#### 2. Redis for Distributed Caching
```python
import redis
import pickle

class ModelCache:
    def __init__(self, host='localhost', port=6379):
        self.redis = redis.Redis(host=host, port=port)
    
    def cache_prediction(self, key, value, ttl=3600):
        self.redis.setex(key, ttl, pickle.dumps(value))
    
    def get_prediction(self, key):
        cached = self.redis.get(key)
        return pickle.loads(cached) if cached else None

cache = ModelCache()

def predict_with_cache(model, features):
    cache_key = hash(features.tobytes())
    
    # Check cache first
    cached_pred = cache.get_prediction(cache_key)
    if cached_pred is not None:
        return cached_pred
    
    # Compute and cache
    prediction = model.predict(features)
    cache.cache_prediction(cache_key, prediction)
    return prediction
```

#### 3. Feature Store Caching
```python
class FeatureStore:
    def __init__(self):
        self.online_store = {}  # Fast lookup
        self.offline_store = {}  # Batch processing
    
    def get_features(self, entity_id, feature_names):
        # Try online store first
        if entity_id in self.online_store:
            return self.online_store[entity_id]
        
        # Fallback to offline store
        features = self._compute_features(entity_id, feature_names)
        self.online_store[entity_id] = features
        return features
```

---

## 2. Hyperparameter Tuning Advanced Techniques

### Bayesian Optimization with Optuna
```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
    }
    
    model = xgb.XGBRegressor(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, 
                           scoring='neg_mean_squared_error').mean()
    return -score

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, n_jobs=-1)

print(f"Best params: {study.best_params}")
print(f"Best score: {study.best_value}")
```

### Multi-Objective Optimization
```python
def multi_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10)
    }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Optimize for both accuracy and inference time
    accuracy = model.score(X_val, y_val)
    inference_time = measure_inference_time(model, X_val)
    
    return accuracy, -inference_time  # Maximize accuracy, minimize time

study = optuna.create_study(directions=['maximize', 'minimize'])
study.optimize(multi_objective, n_trials=100)

# Get Pareto front
pareto_front = study.best_trials
```

### Hyperband & ASHA (Asynchronous Successive Halving)
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_model(config):
    model = xgb.XGBRegressor(**config)
    
    for i in range(10):  # Iterative training
        model.set_params(n_estimators=model.n_estimators + 10)
        model.fit(X_train, y_train)
        
        score = model.score(X_val, y_val)
        tune.report(score=score)

config = {
    "max_depth": tune.randint(3, 15),
    "learning_rate": tune.loguniform(0.01, 0.3),
    "subsample": tune.uniform(0.5, 1.0)
}

scheduler = ASHAScheduler(
    metric="score",
    mode="max",
    max_t=100,
    grace_period=10,
    reduction_factor=3
)

analysis = tune.run(
    train_model,
    config=config,
    num_samples=50,
    scheduler=scheduler
)
```

### Population-Based Training
```python
from ray.tune.schedulers import PopulationBasedTraining

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="score",
    mode="max",
    perturbation_interval=5,
    hyperparam_mutations={
        "learning_rate": lambda: np.random.uniform(0.001, 0.1),
        "momentum": lambda: np.random.uniform(0.8, 0.99)
    }
)

tune.run(train_model, scheduler=pbt, num_samples=8)
```

---

## 3. Semi-Supervised Learning

### Self-Training
```python
class SelfTrainingClassifier:
    def __init__(self, base_model, threshold=0.9):
        self.model = base_model
        self.threshold = threshold
    
    def fit(self, X_labeled, y_labeled, X_unlabeled, iterations=10):
        self.model.fit(X_labeled, y_labeled)
        
        for _ in range(iterations):
            # Predict on unlabeled data
            probs = self.model.predict_proba(X_unlabeled)
            max_probs = probs.max(axis=1)
            predictions = probs.argmax(axis=1)
            
            # Select confident predictions
            confident_mask = max_probs >= self.threshold
            
            if confident_mask.sum() == 0:
                break
            
            # Add confident predictions to training set
            X_labeled = np.vstack([X_labeled, X_unlabeled[confident_mask]])
            y_labeled = np.hstack([y_labeled, predictions[confident_mask]])
            X_unlabeled = X_unlabeled[~confident_mask]
            
            # Retrain
            self.model.fit(X_labeled, y_labeled)
        
        return self
```

### Co-Training
```python
class CoTrainingClassifier:
    def __init__(self, model1, model2, k=5, iterations=10):
        self.model1 = model1
        self.model2 = model2
        self.k = k  # Number of examples to add per iteration
        self.iterations = iterations
    
    def fit(self, X1_labeled, X2_labeled, y_labeled, 
            X1_unlabeled, X2_unlabeled):
        # Train initial models on different feature sets
        self.model1.fit(X1_labeled, y_labeled)
        self.model2.fit(X2_labeled, y_labeled)
        
        for _ in range(self.iterations):
            # Model 1 predicts on unlabeled (view 2)
            probs1 = self.model1.predict_proba(X1_unlabeled)
            
            # Model 2 predicts on unlabeled (view 1)
            probs2 = self.model2.predict_proba(X2_unlabeled)
            
            # Select most confident predictions
            confident1_idx = probs1.max(axis=1).argsort()[-self.k:]
            confident2_idx = probs2.max(axis=1).argsort()[-self.k:]
            
            # Add to training sets
            X1_labeled = np.vstack([X1_labeled, X1_unlabeled[confident2_idx]])
            X2_labeled = np.vstack([X2_labeled, X2_unlabeled[confident1_idx]])
            y_labeled = np.hstack([
                y_labeled,
                probs2[confident2_idx].argmax(axis=1),
                probs1[confident1_idx].argmax(axis=1)
            ])
            
            # Retrain
            self.model1.fit(X1_labeled, y_labeled)
            self.model2.fit(X2_labeled, y_labeled)
        
        return self
```

### Pseudo-Labeling with Consistency Regularization
```python
import torch
import torch.nn as nn

class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred1, pred2):
        return self.mse(pred1, pred2)

def train_with_consistency(model, labeled_loader, unlabeled_loader, 
                           epochs=100, lambda_u=1.0):
    optimizer = torch.optim.Adam(model.parameters())
    ce_loss = nn.CrossEntropyLoss()
    consistency_loss = ConsistencyLoss()
    
    for epoch in range(epochs):
        for (x_l, y_l), (x_u, _) in zip(labeled_loader, unlabeled_loader):
            # Supervised loss
            pred_l = model(x_l)
            loss_supervised = ce_loss(pred_l, y_l)
            
            # Unsupervised consistency loss
            pred_u1 = model(x_u)
            pred_u2 = model(x_u + torch.randn_like(x_u) * 0.1)  # Augmented
            loss_consistency = consistency_loss(pred_u1, pred_u2)
            
            # Total loss
            loss = loss_supervised + lambda_u * loss_consistency
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### MixMatch
```python
def mixmatch(model, x_labeled, y_labeled, x_unlabeled, alpha=0.75, T=0.5, K=2):
    """MixMatch: A Holistic Approach to Semi-Supervised Learning"""
    batch_size = x_labeled.shape[0]
    
    # 1. Data augmentation for unlabeled
    augmented_unlabeled = [augment(x_unlabeled) for _ in range(K)]
    
    # 2. Sharpening (temperature scaling)
    predictions = [model(aug) for aug in augmented_unlabeled]
    avg_pred = torch.stack(predictions).mean(dim=0)
    sharpened = torch.pow(avg_pred, 1/T)
    sharpened = sharpened / sharpened.sum(dim=1, keepdim=True)
    
    # 3. MixUp
    all_x = torch.cat([x_labeled] + augmented_unlabeled)
    all_y = torch.cat([y_labeled] + [sharpened] * K)
    
    # Shuffle
    idx = torch.randperm(all_x.shape[0])
    
    # MixUp
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * all_x + (1 - lam) * all_x[idx]
    mixed_y = lam * all_y + (1 - lam) * all_y[idx]
    
    return mixed_x, mixed_y
```

---

## 4. Scaling & Optimization Strategies

### Time Optimization

#### 1. Vectorization
```python
# Slow: Loop
def slow_computation(arr):
    result = []
    for i in range(len(arr)):
        result.append(arr[i] ** 2 + 2 * arr[i] + 1)
    return np.array(result)

# Fast: Vectorized
def fast_computation(arr):
    return arr ** 2 + 2 * arr + 1

# Benchmark
arr = np.random.rand(1000000)
%timeit slow_computation(arr)  # ~200ms
%timeit fast_computation(arr)  # ~5ms (40x faster)
```

#### 2. JIT Compilation with Numba
```python
from numba import jit, prange

@jit(nopython=True, parallel=True)
def fast_distance_matrix(X):
    n = X.shape[0]
    distances = np.zeros((n, n))
    
    for i in prange(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(X.shape[1]):
                dist += (X[i, k] - X[j, k]) ** 2
            dist = np.sqrt(dist)
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances

# Can be 100x faster than pure Python
```

#### 3. Parallel Processing
```python
from joblib import Parallel, delayed
from multiprocessing import cpu_count

def process_batch(batch):
    # Expensive computation
    return result

# Sequential
results = [process_batch(batch) for batch in batches]

# Parallel
results = Parallel(n_jobs=cpu_count())(
    delayed(process_batch)(batch) for batch in batches
)
```

#### 4. GPU Acceleration
```python
import cupy as cp  # NumPy-compatible GPU arrays

# CPU
x_cpu = np.random.rand(10000, 10000)
y_cpu = np.matmul(x_cpu, x_cpu.T)  # Slow

# GPU
x_gpu = cp.random.rand(10000, 10000)
y_gpu = cp.matmul(x_gpu, x_gpu.T)  # Much faster
```

### Space Optimization

#### 1. Data Type Optimization
```python
import pandas as pd

def optimize_dtypes(df):
    """Reduce memory usage by optimizing data types"""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype('category')
    
    return df

# Can reduce memory by 80%+
```

#### 2. Sparse Matrices
```python
from scipy.sparse import csr_matrix

# Dense matrix: 10000x10000 = 400MB
dense = np.random.rand(10000, 10000)

# Sparse matrix (1% non-zero): ~4MB
sparse = csr_matrix(dense * (np.random.rand(10000, 10000) < 0.01))

# 100x memory reduction
```

#### 3. Chunking for Large Datasets
```python
def process_large_file(filename, chunksize=100000):
    """Process large files in chunks"""
    results = []
    
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        processed = process_chunk(chunk)
        results.append(processed)
    
    return pd.concat(results)
```

#### 4. Memory-Mapped Files
```python
# Create memory-mapped array
mmap_array = np.memmap('large_array.dat', dtype='float32', 
                       mode='w+', shape=(1000000, 1000))

# Use like regular array, but stays on disk
mmap_array[0] = np.random.rand(1000)
```

### Cost Optimization

#### 1. Spot Instances Strategy
```python
# AWS Spot Instance configuration
spot_config = {
    'instance_type': 'p3.2xlarge',
    'max_price': '1.00',  # Much cheaper than on-demand
    'spot_strategy': 'diversified'
}

# Implement checkpointing for fault tolerance
def train_with_checkpointing(model, data, checkpoint_freq=100):
    for epoch in range(num_epochs):
        train_epoch(model, data)
        
        if epoch % checkpoint_freq == 0:
            save_checkpoint(model, epoch)
```

#### 2. Model Compression
```python
import torch.quantization as quantization

# Post-training static quantization
model_fp32 = load_model()
model_int8 = quantization.quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)

# 4x smaller model, 2-3x faster inference
```

#### 3. Batch Processing
```python
def cost_effective_inference(data, model, batch_size=1000):
    """Batch processing reduces API calls and compute costs"""
    predictions = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_predictions = model.predict(batch)
        predictions.extend(batch_predictions)
    
    return predictions
```

---

## 5. LLM Evaluation & Monitoring

### LLM-as-a-Judge Framework (2026)
```python
from typing import List, Dict

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic library not installed. Install with: pip install anthropic")

class LLMJudge:
    def __init__(self, judge_model="claude-3-opus-20240229"):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic library is required. Install with: pip install anthropic")
        self.client = anthropic.Anthropic()
        self.judge_model = judge_model
    
    def evaluate_response(self, prompt: str, response: str, 
                         criteria: List[str]) -> Dict[str, float]:
        """Evaluate response quality using LLM as judge"""
        
        evaluation_prompt = f"""
        Evaluate the following response based on these criteria: {', '.join(criteria)}
        
        Original Prompt: {prompt}
        
        Response to Evaluate: {response}
        
        For each criterion, provide a score from 1-10 and brief justification.
        Format: {{"criterion": {{"score": X, "reasoning": "..."}}, ...}}
        """
        
        message = self.client.messages.create(
            model=self.judge_model,
            max_tokens=2000,
            messages=[{"role": "user", "content": evaluation_prompt}]
        )
        
        scores = self._parse_scores(message.content)
        return scores
    
    def pairwise_comparison(self, prompt: str, response_a: str, 
                           response_b: str) -> str:
        """Compare two responses and determine winner"""
        
        comparison_prompt = f"""
        Which response is better for the following prompt?
        
        Prompt: {prompt}
        
        Response A: {response_a}
        
        Response B: {response_b}
        
        Respond with either "A", "B", or "Tie", followed by reasoning.
        """
        
        message = self.client.messages.create(
            model=self.judge_model,
            max_tokens=1000,
            messages=[{"role": "user", "content": comparison_prompt}]
        )
        
        return self._parse_winner(message.content)

# Usage
judge = LLMJudge()

scores = judge.evaluate_response(
    prompt="Explain quantum computing",
    response=model_output,
    criteria=["accuracy", "clarity", "completeness", "coherence"]
)

winner = judge.pairwise_comparison(
    prompt=user_prompt,
    response_a=model_a_output,
    response_b=model_b_output
)
```

### ELO Rating System for LLMs
```python
class ELORatingSystem:
    def __init__(self, k_factor=32, initial_rating=1500):
        self.k_factor = k_factor
        self.ratings = {}
        self.initial_rating = initial_rating
    
    def expected_score(self, rating_a, rating_b):
        """Calculate expected win probability"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, model_a, model_b, result):
        """
        Update ELO ratings based on match result
        result: 1 if A wins, 0 if B wins, 0.5 for tie
        """
        rating_a = self.ratings.get(model_a, self.initial_rating)
        rating_b = self.ratings.get(model_b, self.initial_rating)
        
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a
        
        new_rating_a = rating_a + self.k_factor * (result - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1 - result) - expected_b)
        
        self.ratings[model_a] = new_rating_a
        self.ratings[model_b] = new_rating_b
    
    def get_leaderboard(self):
        """Get ranked models"""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

# Usage
elo_system = ELORatingSystem()

# Run tournaments
for prompt in test_prompts:
    response_a = model_a.generate(prompt)
    response_b = model_b.generate(prompt)
    
    winner = judge.pairwise_comparison(prompt, response_a, response_b)
    result = 1 if winner == "A" else (0 if winner == "B" else 0.5)
    
    elo_system.update_ratings("model_a", "model_b", result)

print(elo_system.get_leaderboard())
```

### RAG Evaluation Metrics
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

def evaluate_rag_system(questions, answers, contexts, ground_truths):
    """Comprehensive RAG evaluation"""
    
    dataset = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    }
    
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,  # Answer grounded in context?
            answer_relevancy,  # Answer addresses question?
            context_precision,  # Relevant chunks ranked high?
            context_recall,  # All relevant chunks retrieved?
        ]
    )
    
    return result
```

### Hallucination Detection
```python
class HallucinationDetector:
    def __init__(self, nli_model="microsoft/deberta-large-mnli"):
        from transformers import pipeline
        self.nli = pipeline("text-classification", model=nli_model)
    
    def detect_hallucination(self, source_text: str, generated_text: str) -> float:
        """
        Detect if generated text contradicts or is not supported by source
        Returns: Hallucination score (0-1, higher = more hallucination)
        """
        # Split generated text into claims
        claims = self._extract_claims(generated_text)
        
        hallucination_scores = []
        for claim in claims:
            # Check if claim is entailed by source
            result = self.nli(f"{source_text} [SEP] {claim}")[0]
            
            if result['label'] == 'CONTRADICTION':
                hallucination_scores.append(1.0)
            elif result['label'] == 'NEUTRAL':
                hallucination_scores.append(0.5)
            else:  # ENTAILMENT
                hallucination_scores.append(0.0)
        
        return np.mean(hallucination_scores) if hallucination_scores else 0.0
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract individual claims from text"""
        # Simple sentence splitting (can use more sophisticated methods)
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
```

---

## 6. Agentic AI Architecture

### Multi-Agent System Framework
```python
from typing import List, Dict, Any
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.memory = []
    
    @abstractmethod
    def process(self, task: str, context: Dict[str, Any]) -> str:
        pass
    
    def add_to_memory(self, item: Any):
        self.memory.append(item)

class ResearchAgent(Agent):
    def process(self, task: str, context: Dict[str, Any]) -> str:
        """Research information on given topic"""
        # Use web search, retrieve documents
        research_results = self._search(task)
        self.add_to_memory({"task": task, "results": research_results})
        return research_results

class AnalysisAgent(Agent):
    def process(self, task: str, context: Dict[str, Any]) -> str:
        """Analyze data and extract insights"""
        data = context.get("data")
        insights = self._analyze(data)
        self.add_to_memory({"task": task, "insights": insights})
        return insights

class WritingAgent(Agent):
    def process(self, task: str, context: Dict[str, Any]) -> str:
        """Write content based on research and analysis"""
        research = context.get("research")
        analysis = context.get("analysis")
        content = self._write(task, research, analysis)
        self.add_to_memory({"task": task, "content": content})
        return content

class AgentOrchestrator:
    def __init__(self, agents: List[Agent]):
        self.agents = {agent.name: agent for agent in agents}
        self.workflow = []
    
    def add_step(self, agent_name: str, task: str, 
                 context_dependencies: List[str] = None):
        """Add step to workflow"""
        self.workflow.append({
            "agent": agent_name,
            "task": task,
            "dependencies": context_dependencies or []
        })
    
    def execute(self) -> Dict[str, Any]:
        """Execute multi-agent workflow"""
        results = {}
        
        for step in self.workflow:
            agent = self.agents[step["agent"]]
            
            # Build context from dependencies
            context = {
                dep: results[dep] for dep in step["dependencies"]
                if dep in results
            }
            
            # Execute agent
            result = agent.process(step["task"], context)
            results[step["agent"]] = result
        
        return results

# Usage
orchestrator = AgentOrchestrator([
    ResearchAgent("researcher", "Research"),
    AnalysisAgent("analyzer", "Analysis"),
    WritingAgent("writer", "Writing")
])

orchestrator.add_step("researcher", "Research AI trends 2026")
orchestrator.add_step("analyzer", "Analyze research findings", 
                     ["researcher"])
orchestrator.add_step("writer", "Write comprehensive report",
                     ["researcher", "analyzer"])

results = orchestrator.execute()
```

### ReAct Pattern (Reasoning + Acting)
```python
class ReActAgent:
    def __init__(self, llm, tools: Dict[str, callable]):
        self.llm = llm
        self.tools = tools
    
    def solve(self, task: str, max_iterations: int = 10) -> str:
        """Solve task using ReAct pattern"""
        context = f"Task: {task}\n\n"
        
        for i in range(max_iterations):
            # Thought: Reasoning step
            thought = self._generate_thought(context)
            context += f"Thought {i+1}: {thought}\n"
            
            # Action: Decide what to do
            action = self._generate_action(context)
            context += f"Action {i+1}: {action}\n"
            
            # Check if final answer
            if action.startswith("Final Answer:"):
                return action.replace("Final Answer:", "").strip()
            
            # Observation: Execute action
            observation = self._execute_action(action)
            context += f"Observation {i+1}: {observation}\n\n"
        
        return "Failed to solve task within max iterations"
    
    def _generate_thought(self, context: str) -> str:
        """Generate reasoning thought"""
        prompt = f"{context}\nWhat should I think about next?"
        return self.llm.generate(prompt)
    
    def _generate_action(self, context: str) -> str:
        """Generate action to take"""
        prompt = f"""{context}
        Available actions: {list(self.tools.keys())}
        What action should I take? (or provide Final Answer)
        """
        return self.llm.generate(prompt)
    
    def _execute_action(self, action: str) -> str:
        """Execute tool and return observation"""
        for tool_name, tool_func in self.tools.items():
            if tool_name in action:
                # Extract arguments
                args = self._parse_args(action)
                return tool_func(**args)
        return "Invalid action"

# Tools
def search(query: str) -> str:
    # Implement web search
    return search_results

def calculate(expression: str) -> str:
    return str(eval(expression))

def wikipedia(topic: str) -> str:
    # Fetch Wikipedia article
    return article_text

# Usage
agent = ReActAgent(
    llm=your_llm,
    tools={
        "search": search,
        "calculate": calculate,
        "wikipedia": wikipedia
    }
)

answer = agent.solve("What is the GDP of the country that won the 2022 World Cup?")
```

### Agent Memory Systems
```python
from collections import deque

class AgentMemory:
    def __init__(self, short_term_size=10, embedding_model=None):
        self.short_term = deque(maxlen=short_term_size)
        self.long_term = []
        self.semantic_memory = {}  # Facts and knowledge
        self.embedding_model = embedding_model
    
    def add(self, memory_item: Dict):
        """Add to short-term memory"""
        self.short_term.append(memory_item)
        
        # Consolidate to long-term if important
        if self._is_important(memory_item):
            self.consolidate(memory_item)
    
    def consolidate(self, memory_item: Dict):
        """Move to long-term memory"""
        self.long_term.append(memory_item)
        
        # Create semantic memory
        if "concept" in memory_item:
            self.semantic_memory[memory_item["concept"]] = memory_item["value"]
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant memories"""
        if self.embedding_model:
            # Semantic search
            query_embedding = self.embedding_model.encode(query)
            
            memories = list(self.short_term) + self.long_term
            scores = [
                cosine_similarity(query_embedding, 
                                self.embedding_model.encode(m["content"]))
                for m in memories
            ]
            
            top_k_idx = np.argsort(scores)[-k:]
            return [memories[i] for i in top_k_idx]
        else:
            # Simple recency-based retrieval
            return list(self.short_term)[-k:]
    
    def _is_important(self, memory_item: Dict) -> bool:
        """Determine if memory should be consolidated"""
        # Implement importance scoring
        return memory_item.get("importance", 0) > 0.7
```

---

## 7. Production MLOps Best Practices

### Complete MLOps Pipeline
```python
import mlflow
import dagster
from prefect import flow, task

@task
def load_data():
    """Load and validate data"""
    data = pd.read_csv("data.csv")
    validate_schema(data)
    return data

@task
def preprocess(data):
    """Preprocess data"""
    processed = apply_transformations(data)
    return processed

@task
def train_model(data):
    """Train model with experiment tracking"""
    with mlflow.start_run():
        model = XGBClassifier()
        model.fit(data.drop('target', axis=1), data['target'])
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_val, model.predict(X_val)),
            "auc": roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        })
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return model

@task
def evaluate_model(model, test_data):
    """Evaluate model"""
    predictions = model.predict(test_data.drop('target', axis=1))
    metrics = calculate_metrics(test_data['target'], predictions)
    return metrics

@task
def register_model(model, metrics):
    """Register model if performance is good"""
    if metrics['accuracy'] > 0.9:
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            "production_model"
        )

@flow
def ml_pipeline():
    """Complete ML pipeline"""
    data = load_data()
    processed_data = preprocess(data)
    model = train_model(processed_data)
    metrics = evaluate_model(model, test_data)
    register_model(model, metrics)

# Schedule pipeline
if __name__ == "__main__":
    ml_pipeline.serve(
        name="ml-pipeline",
        cron="0 2 * * *"  # Run daily at 2 AM
    )
```

### Feature Store Implementation
```python
class FeatureStore:
    def __init__(self, online_db, offline_db):
        self.online_db = online_db  # Redis
        self.offline_db = offline_db  # Data warehouse
    
    def register_feature(self, feature_name: str, 
                        transformation: callable,
                        dependencies: List[str] = None):
        """Register feature transformation"""
        self.features[feature_name] = {
            "transformation": transformation,
            "dependencies": dependencies or []
        }
    
    def compute_features(self, entity_ids: List[str],
                        feature_names: List[str],
                        is_training: bool = False) -> pd.DataFrame:
        """Compute features for entities"""
        
        if is_training:
            # Use offline store for training
            return self._compute_from_offline(entity_ids, feature_names)
        else:
            # Use online store for serving
            return self._compute_from_online(entity_ids, feature_names)
    
    def _compute_from_online(self, entity_ids, feature_names):
        """Fast online feature retrieval"""
        features = {}
        
        for feature_name in feature_names:
            # Try cache first
            cached = self.online_db.mget([
                f"{feature_name}:{entity_id}" 
                for entity_id in entity_ids
            ])
            
            # Compute missing features
            missing_idx = [i for i, v in enumerate(cached) if v is None]
            if missing_idx:
                missing_ids = [entity_ids[i] for i in missing_idx]
                computed = self._compute_feature(feature_name, missing_ids)
                
                # Cache results
                for entity_id, value in zip(missing_ids, computed):
                    self.online_db.setex(
                        f"{feature_name}:{entity_id}",
                        3600,  # 1 hour TTL
                        value
                    )
            
            features[feature_name] = cached
        
        return pd.DataFrame(features)
```

### Model Monitoring
```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

class ModelMonitor:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.alerts = []
    
    def check_drift(self, current_data: pd.DataFrame):
        """Check for data drift"""
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset()
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        drift_detected = report.as_dict()['metrics'][0]['result']['drift_detected']
        
        if drift_detected:
            self._trigger_alert("Data drift detected!")
        
        return report
    
    def check_performance_degradation(self, y_true, y_pred):
        """Monitor model performance"""
        current_accuracy = accuracy_score(y_true, y_pred)
        
        # Compare with baseline
        if current_accuracy < self.baseline_accuracy * 0.95:
            self._trigger_alert(
                f"Performance degradation: {current_accuracy:.3f} vs {self.baseline_accuracy:.3f}"
            )
    
    def _trigger_alert(self, message: str):
        """Send alert"""
        self.alerts.append({
            "timestamp": datetime.now(),
            "message": message
        })
        
        # Send to monitoring system
        send_to_slack(message)
        send_to_pagerduty(message)
```

---

## 8. Error Handling & Robustness

### Comprehensive Error Handling
```python
import logging
from functools import wraps
from typing import TypeVar, Callable

T = TypeVar('T')

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Retry decorator with exponential backoff"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    wait_time = delay * (2 ** attempt)
                    logging.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
            
            return None
        return wrapper
    return decorator

class MLPipelineError(Exception):
    """Base exception for ML pipeline"""
    pass

class DataValidationError(MLPipelineError):
    """Data validation failed"""
    pass

class ModelTrainingError(MLPipelineError):
    """Model training failed"""
    pass

class InferenceError(MLPipelineError):
    """Model inference failed"""
    pass

def safe_predict(model, X, fallback_value=None):
    """Safe prediction with error handling"""
    try:
        # Validate input
        if X is None or len(X) == 0:
            raise ValueError("Empty input")
        
        # Validate features
        if X.shape[1] != model.n_features_in_:
            raise ValueError(
                f"Expected {model.n_features_in_} features, "
                f"got {X.shape[1]}"
            )
        
        # Make prediction
        predictions = model.predict(X)
        
        # Validate output
        if np.any(np.isnan(predictions)):
            raise InferenceError("NaN in predictions")
        
        return predictions
        
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        
        if fallback_value is not None:
            return np.full(len(X), fallback_value)
        else:
            raise InferenceError(f"Prediction failed: {e}") from e
```

### Input Validation
```python
from pydantic import BaseModel, validator, Field
from typing import List

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_items=10, max_items=10)
    model_version: str = "v1"
    
    @validator('features')
    def validate_features(cls, features):
        # Check for NaN or inf
        if any(np.isnan(features)) or any(np.isinf(features)):
            raise ValueError("Features contain NaN or inf")
        
        # Check range
        if any(f < -100 or f > 100 for f in features):
            raise ValueError("Features out of valid range")
        
        return features

# FastAPI endpoint with validation
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        model = load_model(request.model_version)
        prediction = model.predict([request.features])
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Graceful Degradation
```python
class RobustMLService:
    def __init__(self):
        self.primary_model = load_model("primary")
        self.fallback_model = load_model("fallback")
        self.cache = {}
    
    def predict(self, features):
        """Predict with graceful degradation"""
        # Try cache first
        cache_key = hash(features.tobytes())
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try primary model
        try:
            prediction = self.primary_model.predict(features)
            self.cache[cache_key] = prediction
            return prediction
        except Exception as e:
            logging.warning(f"Primary model failed: {e}")
        
        # Fallback to simpler model
        try:
            prediction = self.fallback_model.predict(features)
            return prediction
        except Exception as e:
            logging.error(f"Fallback model failed: {e}")
        
        # Last resort: return default
        return self.default_prediction()
```

---

## 🎯 Key Takeaways

### When to Use Each Technique:

**Semi-Supervised Learning**: 
- Use when: Labeled data is expensive, abundant unlabeled data available
- Best for: Text classification, image classification with limited labels
- Limitations: Requires good initial model, can propagate errors

**Hyperparameter Tuning**:
- Grid Search: Small search space, interpretability needed
- Bayesian Optimization: Expensive evaluations, larger search space
- Hyperband: Many hyperparameters, limited budget
- PBT: Online learning, continuous tuning

**Scaling Strategies**:
- Vectorization: First optimization to try, usually 10-100x speedup
- JIT Compilation: CPU-bound operations, 10-1000x speedup
- GPU Acceleration: Large matrix operations, 10-100x speedup
- Distributed Computing: Very large datasets, horizontal scaling

**LLM Evaluation**:
- LLM-as-Judge: Quick evaluation, human-like judgment
- ELO Ratings: Comparing multiple models, leaderboards
- RAG Metrics: Retrieval-augmented systems
- Hallucination Detection: Critical applications, factual accuracy

**Agent Architectures**:
- Single Agent: Simple tasks, limited tool use
- Multi-Agent: Complex workflows, specialized roles
- ReAct: Reasoning-heavy tasks, tool use
- Hierarchical: Complex planning, sub-goals

---

*This document covers advanced topics for top 0.1% mastery. Combine with the main blueprint for comprehensive learning.*

*Last Updated: January 2026*
