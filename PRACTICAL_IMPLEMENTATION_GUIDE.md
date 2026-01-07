# Practical Implementation Guide 🛠️
## Hands-On Projects and Code Examples

---

## 📑 Table of Contents
1. [Week 1 Projects](#week1-projects)
2. [Week 2 Projects](#week2-projects)
3. [Week 3 Projects](#week3-projects)
4. [Week 4 Projects](#week4-projects)
5. [Portfolio Projects](#portfolio-projects)
6. [Code Templates](#code-templates)

---

## Week 1 Projects

### Project 1: Custom Data Structures Library

**Goal**: Implement fundamental data structures from scratch

**File: `custom_ds.py`**
```python
"""
Custom Data Structures Implementation
Educational purpose: Understanding internals
"""

class Node:
    """Node for linked structures"""
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    """Singly Linked List Implementation"""
    def __init__(self):
        self.head = None
        self.size = 0
    
    def append(self, data):
        """Add element to end - O(n)"""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1
    
    def prepend(self, data):
        """Add element to beginning - O(1)"""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def delete(self, data):
        """Delete first occurrence - O(n)"""
        if not self.head:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            self.size -= 1
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                self.size -= 1
                return
            current = current.next
    
    def search(self, data):
        """Search for element - O(n)"""
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return " -> ".join(elements)


class Stack:
    """Stack using list - LIFO"""
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add to top - O(1)"""
        self.items.append(item)
    
    def pop(self):
        """Remove from top - O(1)"""
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Pop from empty stack")
    
    def peek(self):
        """View top element - O(1)"""
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Peek from empty stack")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)


class Queue:
    """Queue using list - FIFO"""
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        """Add to rear - O(1)"""
        self.items.append(item)
    
    def dequeue(self):
        """Remove from front - O(n) due to list.pop(0)"""
        if not self.is_empty():
            return self.items.pop(0)
        raise IndexError("Dequeue from empty queue")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)


class BinarySearchTree:
    """Binary Search Tree Implementation"""
    class TreeNode:
        def __init__(self, value):
            self.value = value
            self.left = None
            self.right = None
    
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        """Insert value - O(log n) average, O(n) worst"""
        if not self.root:
            self.root = self.TreeNode(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = self.TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = self.TreeNode(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        """Search for value - O(log n) average"""
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        if value < node.value:
            return self._search_recursive(node.left, value)
        return self._search_recursive(node.right, value)
    
    def inorder_traversal(self):
        """Inorder traversal - returns sorted list"""
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)


class MinHeap:
    """Min Heap Implementation"""
    def __init__(self):
        self.heap = []
    
    def push(self, item):
        """Add item - O(log n)"""
        self.heap.append(item)
        self._bubble_up(len(self.heap) - 1)
    
    def pop(self):
        """Remove minimum - O(log n)"""
        if not self.heap:
            raise IndexError("Pop from empty heap")
        
        self._swap(0, len(self.heap) - 1)
        min_value = self.heap.pop()
        self._bubble_down(0)
        return min_value
    
    def peek(self):
        """View minimum - O(1)"""
        if self.heap:
            return self.heap[0]
        raise IndexError("Peek from empty heap")
    
    def _bubble_up(self, index):
        parent = (index - 1) // 2
        if index > 0 and self.heap[index] < self.heap[parent]:
            self._swap(index, parent)
            self._bubble_up(parent)
    
    def _bubble_down(self, index):
        left = 2 * index + 1
        right = 2 * index + 2
        smallest = index
        
        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left
        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right
        
        if smallest != index:
            self._swap(index, smallest)
            self._bubble_down(smallest)
    
    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]


# Graph representations
class Graph:
    """Graph using adjacency list"""
    def __init__(self, directed=False):
        self.graph = {}
        self.directed = directed
    
    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []
    
    def add_edge(self, v1, v2, weight=1):
        self.add_vertex(v1)
        self.add_vertex(v2)
        self.graph[v1].append((v2, weight))
        if not self.directed:
            self.graph[v2].append((v1, weight))
    
    def bfs(self, start):
        """Breadth-First Search - O(V + E)"""
        visited = set()
        queue = [start]
        result = []
        
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                for neighbor, _ in self.graph.get(vertex, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
        return result
    
    def dfs(self, start):
        """Depth-First Search - O(V + E)"""
        visited = set()
        result = []
        self._dfs_recursive(start, visited, result)
        return result
    
    def _dfs_recursive(self, vertex, visited, result):
        visited.add(vertex)
        result.append(vertex)
        for neighbor, _ in self.graph.get(vertex, []):
            if neighbor not in visited:
                self._dfs_recursive(neighbor, visited, result)
    
    def dijkstra(self, start):
        """Dijkstra's shortest path - O((V + E) log V)"""
        import heapq
        distances = {vertex: float('infinity') for vertex in self.graph}
        distances[start] = 0
        pq = [(0, start)]
        
        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            
            if current_distance > distances[current_vertex]:
                continue
            
            for neighbor, weight in self.graph[current_vertex]:
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
        
        return distances


# Testing
if __name__ == "__main__":
    # Test LinkedList
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    ll.append(3)
    print(f"LinkedList: {ll}")
    
    # Test BST
    bst = BinarySearchTree()
    for val in [5, 3, 7, 1, 9, 4, 6]:
        bst.insert(val)
    print(f"BST Inorder: {bst.inorder_traversal()}")
    
    # Test Graph
    g = Graph(directed=False)
    g.add_edge('A', 'B')
    g.add_edge('A', 'C')
    g.add_edge('B', 'D')
    g.add_edge('C', 'D')
    print(f"BFS from A: {g.bfs('A')}")
    print(f"DFS from A: {g.dfs('A')}")
```

**Learning Outcomes**:
- Understand time and space complexity
- Implement algorithms from scratch
- Debug pointer/reference issues

---

### Project 2: Statistical Analysis Tool

**Goal**: Build CLI tool for comprehensive statistical analysis

**File: `stats_analyzer.py`**
```python
"""
Statistical Analysis Tool
Comprehensive EDA automation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """Comprehensive statistical analysis tool"""
    
    def __init__(self, data):
        """
        Initialize with pandas DataFrame
        """
        self.data = data
        self.numeric_cols = data.select_dtypes(include=[np.number]).columns
        self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    
    def descriptive_stats(self):
        """Comprehensive descriptive statistics"""
        print("=" * 80)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 80)
        
        # Basic info
        print("\nDataset Shape:", self.data.shape)
        print("\nColumn Types:")
        print(self.data.dtypes)
        
        # Numeric columns
        print("\n" + "-" * 80)
        print("NUMERIC COLUMNS STATISTICS")
        print("-" * 80)
        print(self.data[self.numeric_cols].describe())
        
        # Additional statistics
        print("\nSkewness:")
        print(self.data[self.numeric_cols].skew())
        
        print("\nKurtosis:")
        print(self.data[self.numeric_cols].kurtosis())
        
        # Missing values
        print("\n" + "-" * 80)
        print("MISSING VALUES")
        print("-" * 80)
        missing = self.data.isnull().sum()
        missing_pct = 100 * missing / len(self.data)
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Categorical columns
        if len(self.categorical_cols) > 0:
            print("\n" + "-" * 80)
            print("CATEGORICAL COLUMNS")
            print("-" * 80)
            for col in self.categorical_cols:
                print(f"\n{col}:")
                print(self.data[col].value_counts().head())
    
    def correlation_analysis(self, method='pearson'):
        """
        Correlation analysis with visualization
        Methods: pearson, spearman, kendall
        """
        print("\n" + "=" * 80)
        print(f"CORRELATION ANALYSIS ({method.upper()})")
        print("=" * 80)
        
        # Compute correlation
        corr_matrix = self.data[self.numeric_cols].corr(method=method)
        
        # Display high correlations
        print("\nHighly Correlated Pairs (|r| > 0.7):")
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        for col1, col2, corr_val in high_corr:
            print(f"{col1} <-> {col2}: {corr_val:.3f}")
        
        # Heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, linewidths=1)
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'correlation_{method}.png', dpi=300, bbox_inches='tight')
        print(f"\nCorrelation heatmap saved as 'correlation_{method}.png'")
        plt.close()
    
    def distribution_analysis(self):
        """Analyze distributions of numeric variables"""
        print("\n" + "=" * 80)
        print("DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        n_cols = len(self.numeric_cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(self.numeric_cols):
            ax = axes[idx]
            
            # Histogram with KDE
            self.data[col].hist(bins=30, ax=ax, alpha=0.7, edgecolor='black')
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            
            # Add statistics
            mean_val = self.data[col].mean()
            median_val = self.data[col].median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            ax.legend()
            
            # Normality test
            _, p_value = stats.normaltest(self.data[col].dropna())
            print(f"\n{col}:")
            print(f"  Normality test p-value: {p_value:.4f}")
            print(f"  {'Not normally distributed' if p_value < 0.05 else 'Possibly normally distributed'}")
        
        # Remove empty subplots
        for idx in range(len(self.numeric_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
        print("\nDistribution plots saved as 'distributions.png'")
        plt.close()
    
    def outlier_detection(self, method='iqr'):
        """
        Detect outliers using IQR or Z-score method
        """
        print("\n" + "=" * 80)
        print(f"OUTLIER DETECTION ({method.upper()} METHOD)")
        print("=" * 80)
        
        for col in self.numeric_cols:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.data[(self.data[col] < lower_bound) | 
                                    (self.data[col] > upper_bound)]
            else:  # z-score
                z_scores = np.abs(stats.zscore(self.data[col].dropna()))
                outliers = self.data[z_scores > 3]
            
            print(f"\n{col}:")
            print(f"  Number of outliers: {len(outliers)}")
            print(f"  Percentage: {100 * len(outliers) / len(self.data):.2f}%")
    
    def hypothesis_testing(self, col1, col2=None, test_type='ttest'):
        """
        Perform hypothesis tests
        test_type: ttest, anova, chi2
        """
        print("\n" + "=" * 80)
        print(f"HYPOTHESIS TESTING ({test_type.upper()})")
        print("=" * 80)
        
        if test_type == 'ttest' and col2:
            # Independent t-test
            stat, p_value = stats.ttest_ind(self.data[col1].dropna(), 
                                           self.data[col2].dropna())
            print(f"\nTwo-sample t-test: {col1} vs {col2}")
            print(f"  t-statistic: {stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Result: {'Significant difference' if p_value < 0.05 else 'No significant difference'}")
        
        elif test_type == 'anova':
            # One-way ANOVA
            groups = [group[col1].dropna() for name, group in self.data.groupby(col2)]
            stat, p_value = stats.f_oneway(*groups)
            print(f"\nOne-way ANOVA: {col1} across {col2} groups")
            print(f"  F-statistic: {stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Result: {'Significant difference between groups' if p_value < 0.05 else 'No significant difference'}")
    
    def generate_report(self, filename='analysis_report.txt'):
        """Generate comprehensive text report"""
        import sys
        original_stdout = sys.stdout
        
        with open(filename, 'w') as f:
            sys.stdout = f
            
            self.descriptive_stats()
            self.correlation_analysis()
            self.distribution_analysis()
            self.outlier_detection()
        
        sys.stdout = original_stdout
        print(f"\nFull report saved as '{filename}'")


# Example usage
if __name__ == "__main__":
    # Load sample data (use your own dataset)
    # For demo, create synthetic data
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(18, 70, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'score': np.random.normal(75, 10, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'hours': np.random.exponential(40, 1000)
    })
    
    # Create analyzer
    analyzer = StatisticalAnalyzer(data)
    
    # Run analyses
    analyzer.descriptive_stats()
    analyzer.correlation_analysis()
    analyzer.distribution_analysis()
    analyzer.outlier_detection()
    
    # Generate full report
    analyzer.generate_report()
```

**Learning Outcomes**:
- Master descriptive statistics
- Understand correlation methods
- Practice visualization
- Build reusable tools

---

### Project 3: Data Cleaning Pipeline

**Goal**: Automated data cleaning and preprocessing

**File: `data_cleaner.py`**
```python
"""
Automated Data Cleaning Pipeline
Handle common data quality issues
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer


class DataCleaner:
    """Comprehensive data cleaning pipeline"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_log = []
    
    def log(self, message):
        """Log cleaning operations"""
        self.cleaning_log.append(message)
        print(f"[LOG] {message}")
    
    def handle_missing_values(self, strategy='auto', threshold=0.5):
        """
        Handle missing values
        strategy: 'auto', 'drop', 'mean', 'median', 'mode', 'knn'
        threshold: drop columns with >threshold missing values
        """
        self.log("Handling missing values...")
        
        # Drop columns with too many missing values
        missing_pct = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_pct[missing_pct > threshold].index
        if len(cols_to_drop) > 0:
            self.df = self.df.drop(columns=cols_to_drop)
            self.log(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing")
        
        # Handle remaining missing values
        if strategy == 'auto':
            # Numeric: median, Categorical: mode
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            
            # Numeric imputation
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='median')
                self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
                self.log(f"Imputed {len(numeric_cols)} numeric columns with median")
            
            # Categorical imputation
            if len(categorical_cols) > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                self.df[categorical_cols] = imputer.fit_transform(self.df[categorical_cols])
                self.log(f"Imputed {len(categorical_cols)} categorical columns with mode")
        
        elif strategy == 'knn':
            # KNN imputation (numeric only)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            imputer = KNNImputer(n_neighbors=5)
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
            self.log("Applied KNN imputation")
        
        return self
    
    def remove_duplicates(self, subset=None, keep='first'):
        """Remove duplicate rows"""
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = before - len(self.df)
        if removed > 0:
            self.log(f"Removed {removed} duplicate rows")
        return self
    
    def handle_outliers(self, method='iqr', threshold=1.5):
        """
        Handle outliers
        method: 'iqr' or 'zscore'
        """
        self.log(f"Handling outliers using {method} method...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                
                outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
                if outliers > 0:
                    # Cap outliers
                    self.df[col] = self.df[col].clip(lower, upper)
                    self.log(f"Capped {outliers} outliers in {col}")
            
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers = (z_scores > 3).sum()
                if outliers > 0:
                    # Remove outliers
                    self.df = self.df[z_scores <= 3]
                    self.log(f"Removed {outliers} outliers from {col}")
        
        return self
    
    def encode_categorical(self, method='auto'):
        """
        Encode categorical variables
        method: 'auto', 'label', 'onehot'
        """
        self.log("Encoding categorical variables...")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            n_unique = self.df[col].nunique()
            
            if method == 'auto':
                if n_unique <= 10:
                    # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                    self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                    self.log(f"One-hot encoded {col} ({n_unique} categories)")
                else:
                    # Label encoding for high cardinality
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col])
                    self.log(f"Label encoded {col} ({n_unique} categories)")
            
            elif method == 'label':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
            
            elif method == 'onehot':
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
        
        return self
    
    def scale_features(self, method='standard', exclude=None):
        """
        Scale numeric features
        method: 'standard', 'minmax', 'robust'
        """
        self.log(f"Scaling features using {method} method...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if exclude:
            numeric_cols = [col for col in numeric_cols if col not in exclude]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        self.log(f"Scaled {len(numeric_cols)} numeric columns")
        
        return self
    
    def feature_engineering(self):
        """Create new features"""
        self.log("Performing feature engineering...")
        
        # Example: Create polynomial features for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Add squared features
        for col in numeric_cols[:3]:  # Limit to avoid explosion
            self.df[f'{col}_squared'] = self.df[col] ** 2
        
        self.log(f"Created polynomial features")
        return self
    
    def get_cleaned_data(self):
        """Return cleaned DataFrame"""
        self.log(f"Cleaning complete: {self.original_shape} -> {self.df.shape}")
        return self.df
    
    def save_report(self, filename='cleaning_report.txt'):
        """Save cleaning log"""
        with open(filename, 'w') as f:
            f.write("DATA CLEANING REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Original shape: {self.original_shape}\n")
            f.write(f"Final shape: {self.df.shape}\n\n")
            f.write("Operations performed:\n")
            for log_entry in self.cleaning_log:
                f.write(f"  - {log_entry}\n")
        print(f"Report saved as '{filename}'")


# Example usage
if __name__ == "__main__":
    # Create sample dirty data
    data = pd.DataFrame({
        'age': [25, np.nan, 35, 200, 45, 28, np.nan, 32],
        'income': [50000, 60000, np.nan, 75000, 55000, 62000, 58000, 70000],
        'city': ['NYC', 'LA', 'NYC', 'SF', 'LA', 'NYC', 'SF', 'LA'],
        'score': [85, 90, 78, 92, np.nan, 88, 85, 90]
    })
    
    # Add duplicates
    data = pd.concat([data, data.iloc[[0, 1]]], ignore_index=True)
    
    print("Original Data:")
    print(data)
    print(f"\nShape: {data.shape}")
    
    # Clean data
    cleaner = DataCleaner(data)
    cleaned_data = (cleaner
                   .remove_duplicates()
                   .handle_missing_values(strategy='auto')
                   .handle_outliers(method='iqr')
                   .encode_categorical(method='auto')
                   .scale_features(method='standard')
                   .get_cleaned_data())
    
    print("\nCleaned Data:")
    print(cleaned_data)
    
    # Save report
    cleaner.save_report()
```

**Learning Outcomes**:
- Data quality assessment
- Missing value strategies
- Outlier handling techniques
- Feature scaling methods

---

## Week 2 Projects

### Project 4: ML Model Comparison Framework

**Goal**: Compare multiple ML algorithms systematically

**File: `model_comparer.py`**
```python
"""
ML Model Comparison Framework
Systematic evaluation of multiple algorithms
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import time


class ModelComparer:
    """Compare multiple ML models"""
    
    def __init__(self, task='classification'):
        """
        task: 'classification' or 'regression'
        """
        self.task = task
        self.results = []
        self.models = self._get_default_models()
    
    def _get_default_models(self):
        """Get default model suite"""
        if self.task == 'classification':
            return {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'SVM': SVC(random_state=42),
                'KNN': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(random_state=42)
            }
        else:  # regression
            return {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'XGBoost': xgb.XGBRegressor(random_state=42),
                'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
                'SVR': SVR()
            }
    
    def add_model(self, name, model):
        """Add custom model"""
        self.models[name] = model
    
    def evaluate(self, X, y, cv=5, test_size=0.2):
        """
        Evaluate all models
        """
        print("=" * 80)
        print(f"MODEL COMPARISON - {self.task.upper()}")
        print("=" * 80)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            try:
                # Cross-validation
                start_time = time.time()
                
                if self.task == 'classification':
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
                
                # Train on full training set
                model.fit(X_train, y_train)
                
                # Test set predictions
                y_pred = model.predict(X_test)
                
                training_time = time.time() - start_time
                
                # Metrics
                if self.task == 'classification':
                    test_score = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    self.results.append({
                        'Model': name,
                        'CV Score': cv_scores.mean(),
                        'CV Std': cv_scores.std(),
                        'Test Accuracy': test_score,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1,
                        'Training Time (s)': training_time
                    })
                else:
                    test_score = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    self.results.append({
                        'Model': name,
                        'CV R2': cv_scores.mean(),
                        'CV Std': cv_scores.std(),
                        'Test R2': test_score,
                        'RMSE': rmse,
                        'MAE': mae,
                        'Training Time (s)': training_time
                    })
                
                print(f"  ✓ Complete (CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")
            
            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
        
        return self
    
    def get_results(self):
        """Get results DataFrame"""
        return pd.DataFrame(self.results)
    
    def plot_comparison(self, metric=None, figsize=(12, 6)):
        """Plot model comparison"""
        if not self.results:
            print("No results to plot. Run evaluate() first.")
            return
        
        df = self.get_results()
        
        if metric is None:
            metric = 'Test Accuracy' if self.task == 'classification' else 'Test R2'
        
        # Sort by metric
        df_sorted = df.sort_values(metric, ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Performance comparison
        ax1.barh(df_sorted['Model'], df_sorted[metric])
        ax1.set_xlabel(metric)
        ax1.set_title(f'Model Performance ({metric})')
        ax1.grid(axis='x', alpha=0.3)
        
        # Training time comparison
        ax2.barh(df_sorted['Model'], df_sorted['Training Time (s)'])
        ax2.set_xlabel('Training Time (seconds)')
        ax2.set_title('Training Time Comparison')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nComparison plot saved as 'model_comparison.png'")
        plt.show()
    
    def print_summary(self):
        """Print summary table"""
        if not self.results:
            print("No results. Run evaluate() first.")
            return
        
        df = self.get_results()
        
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
        
        # Best model
        if self.task == 'classification':
            best_model = df.loc[df['Test Accuracy'].idxmax(), 'Model']
            best_score = df.loc[df['Test Accuracy'].idxmax(), 'Test Accuracy']
            print(f"\n🏆 Best Model: {best_model} (Accuracy: {best_score:.4f})")
        else:
            best_model = df.loc[df['Test R2'].idxmax(), 'Model']
            best_score = df.loc[df['Test R2'].idxmax(), 'Test R2']
            print(f"\n🏆 Best Model: {best_model} (R²: {best_score:.4f})")


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, load_diabetes
    
    # Classification example
    print("CLASSIFICATION EXAMPLE")
    X, y = load_breast_cancer(return_X_y=True)
    
    comparer = ModelComparer(task='classification')
    comparer.evaluate(X, y, cv=5)
    comparer.print_summary()
    comparer.plot_comparison()
    
    # Regression example
    print("\n\nREGRESSION EXAMPLE")
    X, y = load_diabetes(return_X_y=True)
    
    comparer_reg = ModelComparer(task='regression')
    comparer_reg.evaluate(X, y, cv=5)
    comparer_reg.print_summary()
    comparer_reg.plot_comparison()
```

**Learning Outcomes**:
- Model selection strategies
- Cross-validation
- Performance metrics comparison
- Training time considerations

---

## Week 3 Projects

### Project 5: Computer Vision Pipeline

**Goal**: End-to-end image classification system

**File: `cv_pipeline.py`**
```python
"""
Computer Vision Pipeline
Image classification with PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class ImageDataset(Dataset):
    """Custom image dataset"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ImageClassifier:
    """Image classification pipeline"""
    
    def __init__(self, num_classes, model_name='resnet18', pretrained=True):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        
        self.model = self.model.to(self.device)
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train(self, train_loader, val_loader, epochs=10, lr=0.001):
        """Train the model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f'  ✓ Saved best model (Val Acc: {val_acc:.2f}%)')
            print()
    
    def evaluate(self, data_loader):
        """Evaluate the model"""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def predict(self, image_path):
        """Predict single image"""
        self.model.eval()
        
        image = Image.open(image_path).convert('RGB')
        image = self.val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
        
        return predicted.item(), probabilities[0].cpu().numpy()
    
    def plot_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Accuracy
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300)
        print("Training history saved as 'training_history.png'")
        plt.show()


# Example usage (pseudo-code - need actual dataset)
if __name__ == "__main__":
    # This is a template - replace with your actual dataset
    print("Computer Vision Pipeline Template")
    print("=" * 80)
    print("To use this pipeline:")
    print("1. Prepare your image dataset")
    print("2. Create DataLoaders with ImageDataset")
    print("3. Initialize ImageClassifier")
    print("4. Train with classifier.train()")
    print("5. Evaluate with classifier.evaluate()")
    print("6. Make predictions with classifier.predict()")
```

**Learning Outcomes**:
- Transfer learning
- Data augmentation
- Training loops in PyTorch
- Model evaluation

---

**Continue in next message for Week 4 projects and portfolio projects...**

---

*This practical guide provides hands-on project templates for each week. Modify and extend these projects based on your specific needs and datasets.*

*Last Updated: January 2026*
