import sqlite3
import math
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# For embedding analysis
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class DatabaseExtractor(object):
    def __init__(self, db_path=None, conn=None):
        self.db_path = db_path
        self.conn = conn
        self._sentence_encoder = None

    def get_table_info(self, table_name: str, sample_size: int = 5) -> Dict[str, Any]:
        """
        Get comprehensive information about a table including schema and sample data
        
        Args:
            table_name (str): Name of the table to analyze
            sample_size (int): Number of sample rows to fetch
            
        Returns:
            Dict[str, Any]: Dictionary with table information
        """
        # Get table schema
        schema = self.get_table_schema(table_name)
        
        # Get sample data
        query = f"""
        SELECT * FROM {table_name}
        LIMIT {sample_size}
        """
        
        close_conn = False
        if self.conn is None:
            if self.db_path is None:
                raise ValueError("Either db_path or conn must be provided")
            self.conn = sqlite3.connect(self.db_path)
            close_conn = True
        
        sample_data = []
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            for row in rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    row_dict[col] = row[i]
                sample_data.append(row_dict)
            cursor.close()
        finally:
            if close_conn:
                self.conn.close()
                self.conn = None
        
        # Compile all information
        return {
            "table_name": table_name,
            "columns": schema,
            "sample_data": sample_data
        }

    def get_table_schema(self, table_name: str) -> list[dict]:
        """
        Query SQLite pragma to get column names and data types for a specific table.
    
        Args:
            table_name (str): Name of the table to query
            
        Returns:
            list: List of dictionaries containing column information (name, data_type)
        """
        # SQL query to get column information from SQLite pragma
        query = f"PRAGMA table_info({table_name});"
        
        # Determine whether to use provided connection or create a new one
        close_conn = False
        if self.conn is None:
            if self.db_path is None:
                raise ValueError("Either db_path or conn must be provided")
            self.conn = sqlite3.connect(self.db_path)
            close_conn = True
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            columns = cursor.fetchall()
            
            result = []
            for col in columns:
                # SQLite pragma table_info returns: cid, name, type, notnull, dflt_value, pk
                column_info = {
                    "column_name": col[1],
                    "data_type": col[2],
                    "max_length": None,  # SQLite doesn't store max length in schema
                    "is_nullable": "NO" if col[3] == 1 else "YES",
                    "default_value": col[4]
                }
                result.append(column_info)
            
            cursor.close()
            return result
        finally:
            if close_conn:
                self.conn.close()
                self.conn = None

    def get_numeric_column_stats(self, table_name: str, column_name: str) -> dict:
        """
        Get statistical information about a numeric column in a table.
        
        Args:
            table_name (str): Name of the table to query
            column_name (str): Name of the numeric column to analyze
            
        Returns:
            dict: Dictionary containing statistical information (min, max, mean, median, stddev, quartiles)
        """
        # SQL query to get statistical information for a numeric column
        # SQLite doesn't have built-in functions for percentiles or stddev
        query = f"""
        SELECT 
            MIN({column_name}) as min,
            MAX({column_name}) as max,
            AVG({column_name}) as mean
        FROM 
            {table_name}
        WHERE
            {column_name} IS NOT NULL
        """
        
        # Determine whether to use provided connection or create a new one
        close_conn = False
        if self.conn is None:
            if self.db_path is None:
                raise ValueError("Either db_path or conn must be provided")
            self.conn = sqlite3.connect(self.db_path)
            close_conn = True
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            
            # For median, quartiles and stddev, we need to fetch all values and calculate in Python
            cursor.execute(f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL ORDER BY {column_name}")
            all_values = [row[0] for row in cursor.fetchall()]
            
            # Calculate median and quartiles
            if all_values:
                all_values.sort()
                n = len(all_values)
                median = all_values[n//2] if n % 2 == 1 else (all_values[n//2-1] + all_values[n//2]) / 2
                q1_idx = n // 4
                q1 = all_values[q1_idx] if n % 4 == 1 else (all_values[q1_idx-1] + all_values[q1_idx]) / 2
                q3_idx = (3 * n) // 4
                q3 = all_values[q3_idx] if (3 * n) % 4 == 1 else (all_values[q3_idx-1] + all_values[q3_idx]) / 2
                
                # Calculate standard deviation
                mean = sum(all_values) / n
                variance = sum((x - mean) ** 2 for x in all_values) / n
                stddev = math.sqrt(variance)
            else:
                median = q1 = q3 = stddev = None
            
            stats = {
                "min": result[0],
                "max": result[1],
                "mean": result[2],
                "median": median,
                "stddev": stddev,
                "q1": q1,
                "q3": q3
            }
            
            cursor.close()
            return stats
        finally:
            if close_conn:
                self.conn.close()
                self.conn = None

    def get_string_column_analysis(self, table_name: str, column_name: str, 
                                  sample_size: int = 1000, 
                                  do_embedding: bool = False) -> dict:
        """
        Comprehensive analysis of a string column including entropy, character frequency,
        length distribution, categorical analysis, and optionally embedding analysis.
        
        Args:
            table_name (str): Name of the table to query
            column_name (str): Name of the string column to analyze
            sample_size (int): Maximum number of rows to sample for analysis
            do_embedding (bool): Whether to perform embedding analysis (requires TensorFlow)
            
        Returns:
            dict: Dictionary containing comprehensive analysis results
        """
        # Get string values for analysis
        values = self._get_string_column_values(table_name, column_name, sample_size)
        
        if not values:
            return {"error": "No data found for analysis"}
        
        # Perform Python-based analyses since SQLite lacks many analytical functions
        result = {
            "length_distribution": self._analyze_length_distribution(values),
            "categorical_analysis": self._analyze_categorical_distribution(values),
            "character_frequency": self._analyze_character_frequency(values),
            "entropy": self._calculate_entropy(values),
        }
        
        # Optionally perform embedding analysis
        if do_embedding and TENSORFLOW_AVAILABLE:
            result["embedding_analysis"] = self._analyze_embeddings(values)
        else:
            result["embedding_analysis"] = {"error": "Embedding analysis not available or not requested"}
            
        return result
    
    def _get_string_column_values(self, table_name: str, column_name: str, 
                                 sample_size: int = 1000) -> List[str]:
        """
        Get values from a string column, with optional sampling.
        
        Args:
            table_name (str): Name of the table to query
            column_name (str): Name of the string column
            sample_size (int): Maximum number of rows to return
            
        Returns:
            List[str]: List of string values from the column
        """
        # SQL query to get values with sampling
        query = f"""
        SELECT {column_name}
        FROM {table_name}
        WHERE {column_name} IS NOT NULL
        LIMIT {sample_size}
        """
        
        # Determine whether to use provided connection or create a new one
        close_conn = False
        if self.conn is None:
            if self.db_path is None:
                raise ValueError("Either db_path or conn must be provided")
            self.conn = sqlite3.connect(self.db_path)
            close_conn = True
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            values = [row[0] for row in cursor.fetchall() if row[0] is not None]
            cursor.close()
            return values
        finally:
            if close_conn:
                self.conn.close()
                self.conn = None
    
    def _calculate_entropy(self, values: List[str]) -> Dict[str, float]:
        """
        Calculate Shannon entropy for string values.
        
        Args:
            values (List[str]): List of string values
            
        Returns:
            Dict[str, float]: Dictionary with entropy values
        """
        if not values:
            return {"error": "No data provided for entropy calculation"}
        
        # Calculate value-level entropy
        value_counts = Counter(values)
        total_values = len(values)
        value_probs = [count / total_values for count in value_counts.values()]
        value_entropy = -sum(p * math.log2(p) for p in value_probs)
        
        # Calculate character-level entropy
        all_chars = ''.join(values)
        char_counts = Counter(all_chars)
        total_chars = len(all_chars)
        char_probs = [count / total_chars for count in char_counts.values()]
        char_entropy = -sum(p * math.log2(p) for p in char_probs)
        
        # Calculate normalized entropies
        unique_values = len(value_counts)
        unique_chars = len(char_counts)
        normalized_value_entropy = value_entropy / math.log2(unique_values) if unique_values > 1 else 0
        normalized_char_entropy = char_entropy / math.log2(unique_chars) if unique_chars > 1 else 0
        
        return {
            "value_entropy": value_entropy,
            "char_entropy": char_entropy,
            "normalized_value_entropy": normalized_value_entropy,
            "normalized_char_entropy": normalized_char_entropy,
            "unique_values": unique_values,
            "total_values": total_values,
            "unique_chars": unique_chars,
            "total_chars": total_chars
        }
    
    def _analyze_length_distribution(self, values: List[str]) -> Dict[str, Any]:
        """
        Analyze the distribution of string lengths.
        
        Args:
            values (List[str]): List of string values
            
        Returns:
            Dict[str, Any]: Dictionary with length distribution statistics
        """
        if not values:
            return {"error": "No data provided for length distribution analysis"}
        
        # Calculate lengths
        lengths = [len(value) for value in values]
        
        # Calculate basic statistics
        min_length = min(lengths)
        max_length = max(lengths)
        mean_length = sum(lengths) / len(lengths)
        
        # Calculate median
        sorted_lengths = sorted(lengths)
        n = len(sorted_lengths)
        median_length = sorted_lengths[n//2] if n % 2 == 1 else (sorted_lengths[n//2-1] + sorted_lengths[n//2]) / 2
        
        # Calculate standard deviation
        variance = sum((x - mean_length) ** 2 for x in lengths) / len(lengths)
        stddev_length = math.sqrt(variance)
        
        # Create histogram
        num_bins = min(20, max_length - min_length + 1)
        if num_bins <= 1:
            num_bins = 2  # Ensure at least 2 bins
        
        bin_width = (max_length - min_length) / num_bins
        bin_edges = [min_length + (i * bin_width) for i in range(num_bins + 1)]
        
        # Count values in each bin
        counts = [0] * num_bins
        for length in lengths:
            bin_idx = min(int((length - min_length) / bin_width), num_bins - 1)
            counts[bin_idx] += 1
        
        return {
            "min_length": min_length,
            "max_length": max_length,
            "mean_length": mean_length,
            "median_length": median_length,
            "stddev_length": stddev_length,
            "histogram": {
                "counts": counts,
                "bin_edges": bin_edges
            }
        }
    
    def _analyze_character_frequency(self, values: List[str]) -> Dict[str, Any]:
        """
        Analyze the frequency of characters in string values.
        
        Args:
            values (List[str]): List of string values
            
        Returns:
            Dict[str, Any]: Dictionary with character frequency analysis
        """
        if not values:
            return {"error": "No data provided for character frequency analysis"}
        
        # Join all values into a single string
        all_chars = ''.join(values)
        total_chars = len(all_chars)
        
        # Count character classes
        lowercase = sum(1 for c in all_chars if c.islower())
        uppercase = sum(1 for c in all_chars if c.isupper())
        digits = sum(1 for c in all_chars if c.isdigit())
        whitespace = sum(1 for c in all_chars if c.isspace())
        punctuation = sum(1 for c in all_chars if c in '.,;:!?-()[]{}"\'"')
        other = total_chars - lowercase - uppercase - digits - whitespace - punctuation
        
        char_classes = {
            "lowercase": lowercase,
            "uppercase": uppercase,
            "digits": digits,
            "whitespace": whitespace,
            "punctuation": punctuation,
            "other": other
        }
        
        # Calculate percentages
        char_class_percentages = {k: (v / total_chars) * 100 if total_chars > 0 else 0 
                                 for k, v in char_classes.items()}
        
        # Get most common characters
        char_counts = Counter(all_chars)
        most_common_chars = [
            {"char": char, "count": count, "percentage": (count / total_chars) * 100}
            for char, count in char_counts.most_common(20)
        ]
        
        # Count unique characters
        unique_chars = len(char_counts)
        
        return {
            "total_chars": total_chars,
            "unique_chars": unique_chars,
            "char_class_counts": char_classes,
            "char_class_percentages": char_class_percentages,
            "most_common_chars": most_common_chars
        }
    
    def _analyze_categorical_distribution(self, values: List[str]) -> Dict[str, Any]:
        """
        Analyze the categorical distribution of string values.
        
        Args:
            values (List[str]): List of string values
            
        Returns:
            Dict[str, Any]: Dictionary with categorical distribution analysis
        """
        if not values:
            return {"error": "No data provided for categorical distribution analysis"}
        
        # Count values
        value_counts = Counter(values)
        total_values = len(values)
        unique_values = len(value_counts)
        
        # Calculate cardinality ratio
        cardinality_ratio = unique_values / total_values
        
        # Determine if it's likely a categorical column
        is_likely_categorical = cardinality_ratio < 0.1 and unique_values < 100
        
        # Get most common values
        most_common_values = [
            {"value": str(value), "count": count, "percentage": (count / total_values) * 100}
            for value, count in value_counts.most_common(20)
        ]
        
        return {
            "total_values": total_values,
            "unique_values": unique_values,
            "cardinality_ratio": cardinality_ratio,
            "is_likely_categorical": is_likely_categorical,
            "most_common_values": most_common_values
        }
    
    def _analyze_embeddings(self, values: List[str], max_clusters: int = 10) -> Dict[str, Any]:
        """
        Analyze string values using TensorFlow Universal Sentence Encoder embeddings.
        
        Args:
            values (List[str]): List of string values
            max_clusters (int): Maximum number of clusters to try
            
        Returns:
            Dict[str, Any]: Dictionary with embedding analysis results
        """
        if not values:
            return {}
            
        # Load the Universal Sentence Encoder if not already loaded
        if self._sentence_encoder is None:
            try:
                self._sentence_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            except Exception as e:
                return {"error": f"Failed to load Universal Sentence Encoder: {str(e)}"}
        
        # Generate embeddings
        try:
            embeddings = self._sentence_encoder(values).numpy()
        except Exception as e:
            return {"error": f"Failed to generate embeddings: {str(e)}"}
        
        # Perform dimensionality reduction for visualization (using PCA)
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Perform clustering to identify patterns
        # Find optimal number of clusters
        silhouette_scores = []
        min_clusters = min(2, len(values))
        max_clusters = min(max_clusters, len(values))
        
        if min_clusters < max_clusters:
            for n_clusters in range(min_clusters, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                if len(set(cluster_labels)) > 1:  # Ensure we have at least 2 clusters
                    score = silhouette_score(embeddings, cluster_labels)
                    silhouette_scores.append((n_clusters, score))
            
            # Select optimal number of clusters
            if silhouette_scores:
                optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
            else:
                optimal_clusters = min_clusters
        else:
            optimal_clusters = min_clusters
        
        # Perform final clustering with optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Prepare results
        clusters = {}
        for i, label in enumerate(cluster_labels):
            label_str = str(label)
            if label_str not in clusters:
                clusters[label_str] = []
            
            if len(clusters[label_str]) < 5:  # Limit examples per cluster
                clusters[label_str].append(values[i])
        
        # Count items per cluster
        cluster_counts = Counter(map(str, cluster_labels))
        
        return {
            "embedding_dim": embeddings.shape[1],
            "optimal_clusters": optimal_clusters,
            "silhouette_scores": silhouette_scores,
            "cluster_distribution": [{"cluster": k, "count": v, "percentage": (v / len(values)) * 100} 
                                    for k, v in cluster_counts.items()],
            "cluster_examples": clusters,
            "reduced_embeddings": reduced_embeddings.tolist(),
            "cluster_labels": cluster_labels.tolist()
        }
