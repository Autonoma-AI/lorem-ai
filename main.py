#!/usr/bin/env python3
import sys
import json
import os
import sqlite3
import re
import random
import copy

from src.db import DatabaseExtractor
from src.data_generator import DataGenerator
from src.models import call_ollama


def build_graph(table_name: str, columns: list, sample_data: list, feedback: str = None) -> str:
    """
    Generate a graph of faker keys for each column in a table.
    
    Args:
        table_name (str): Name of the table
        columns (list): List of column dictionaries with column information
        sample_data (list): List of dictionaries containing sample data rows
        feedback (str): Optional feedback from validation to improve the graph
        
    Returns:
        str: JSON string with the generated graph
    """
    print(f"  - Preparing to build graph for table '{table_name}'")
    
    generator = DataGenerator()
    key_descriptions = generator.get_key_descriptions()
    
    # Check if key_descriptions is a dictionary or string
    if isinstance(key_descriptions, dict):
        print(f"  - Loaded {len(key_descriptions)} faker key descriptions")
        # Convert dictionary to string for the prompt
        key_descriptions_str = json.dumps(key_descriptions, indent=2)
    else:
        # If it's already a string, count the number of keys by splitting on commas
        print(f"  - Loaded {len(key_descriptions.split(','))} faker key descriptions")
        key_descriptions_str = key_descriptions

    system = f"""You are a database analyzer. Your task is to analyze information from a table and build a graph to recreate it similarly. You will be using faker to generate data.

    The data that you will be generating has a graph field. The graph is a combination of keys from the key_descriptions. The list will create those keys in that order concatenated for that column name.

    This are the keys you could use to generate the data:
    ```
    {key_descriptions_str}
    ```

    For numeric columns, you should use the 'number', 'integer', 'decimal', or 'float' keys. These keys can generate realistic numeric data based on statistical patterns.

    Examples: 
    - if the value of a column is Thomas, then you should chooose the ["first_name"] key and titlecase modifier
    - if the value of a column is smith, then you should chooose the ["last_name"] key and lowercase modifier
    - if the value of a column is SMITH, then you should chooose the ["last_name"] key and uppercase modifier
    - if the value of a column is tom.smith@gmail.com, then you should chooose the ["email"] key and lowercase modifier
    - if the value of a column is Mr. Thomas Smith, then you should chooose the ["prefix", "first_name", "last_name"] key and titlecase modifier
    - if the value of a column is password123!, then you should chooose the ["password"] key and none modifier
    - if the value of a column is a number like 42, then you should choose the ["integer"] key
    - if the value of a column is a decimal like 42.75, then you should choose the ["decimal"] or ["float"] key

    RETURN THE RESULT ONLY IN THE FOLLOWING JSON FORMAT:
    `{{"columns": [{{"name": "column_name", "graph": ["key1", "key2", ...], "mapping": "choose eithernone, uppercase, lowercase or titlecase", "type": "string or numeric"}}]}}`"""

    # Add feedback to the system prompt if provided
    if feedback:
        system += f"""
        
    IMPORTANT FEEDBACK FROM VALIDATION:
    {feedback}
    
    Please adjust your graph based on this feedback to ensure the generated data matches the patterns in the original data."""
        print(f"  - Added validation feedback to the prompt")

    # Create a description of the table structure and sample data
    user_prompt = f"I need to generate synthetic data for a table named '{table_name}' with the following structure:\n\n"
    
    # Add column information
    user_prompt += "Columns:\n"
    for column in columns:
        col_name = column.get('column_name', '')
        data_type = column.get('data_type', '')
        nullable = "NULL" if column.get('is_nullable', 'YES') == 'YES' else "NOT NULL"
        user_prompt += f"- {col_name} ({data_type}, {nullable})\n"
    
    # Add sample data if available
    if sample_data:
        user_prompt += "\nHere are some sample rows from the table:\n"
        
        # Get column names for the header
        header = [col.get('column_name', f'column_{i}') for i, col in enumerate(columns)]
        user_prompt += "| " + " | ".join(header) + " |\n"
        user_prompt += "| " + " | ".join(["---" for _ in header]) + " |\n"
        
        # Add sample rows
        for row in sample_data[:5]:  # Limit to 5 sample rows to keep prompt size reasonable
            row_values = []
            for col_name in header:
                value = row.get(col_name, '')
                # Truncate long values
                if isinstance(value, str) and len(value) > 50:
                    value = value[:47] + "..."
                row_values.append(str(value))
            user_prompt += "| " + " | ".join(row_values) + " |\n"
    
    print(f"  - Created prompt with {len(columns)} columns and {min(5, len(sample_data))} sample rows")
    print(f"  - Sending request to LLM...")
    
    # Choose which model to use
    use_ollama = True  # Set to False to use Azure OpenAI
    
    if use_ollama:
        print(f"  - Using Ollama for graph generation")
        result = call_ollama(system, user_prompt)
    else:
        print(f"  - Using Azure OpenAI for graph generation")
        result = call_azure_openai(system, user_prompt)
    
    print(f"  - Received response from LLM ({len(result)} characters)")
    return result

def print_json(data, indent=2):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=indent, default=str))

def get_db_path():
    """Get the path to the SQLite database file"""
    db_dir = 'db'
    db_file = 'test_db.sqlite'
    db_path = os.path.join(db_dir, db_file)
    
    # Check if the database file exists
    if not os.path.exists(db_path):
        print(f"Error: Database file {db_path} does not exist.")
        print("Please run generate_test_db.py first to create the test database.")
        return None
    
    return db_path

def test_build_graph(table_name: str):
    """
    Test the build_graph function with actual database information.
    
    Args:
        table_name (str): Name of the table to analyze
    """
    print(f"\n=== Testing build_graph with table: {table_name} ===")
    
    # Get SQLite database path
    db_path = get_db_path()
    if db_path is None:
        return None
    
    # Create DatabaseExtractor instance
    extractor = DatabaseExtractor(db_path=db_path)
    
    try:
        # Get table information
        table_info = extractor.get_table_info(table_name)
        
        # Call build_graph with the table information
        print("Calling build_graph with actual database information...")
        graph_result = build_graph(
            table_name=table_info['table_name'],
            columns=table_info['columns'],
            sample_data=table_info['sample_data']
        )
        
        # Parse and display the result
        try:
            graph_data = json.loads(graph_result)
            print("\n=== Generated Graph ===")
            print_json(graph_data)
            
            # Print a summary of the graph
            print("\n=== Graph Summary ===")
            for column in graph_data.get('columns', []):
                col_name = column.get('name', '')
                graph_keys = column.get('graph', [])
                mapping = column.get('mapping', 'none')
                col_type = column.get('type', 'string')
                print(f"Column: {col_name}")
                print(f"  - Graph keys: {', '.join(graph_keys)}")
                print(f"  - Mapping: {mapping}")
                print(f"  - Type: {col_type}")
                print()
                
            return graph_data
        except json.JSONDecodeError:
            print("Error: Could not parse LLM response as JSON")
            print("Raw response:")
            print(graph_result)
            return None
    except Exception as e:
        print(f"Error testing build_graph: {e}")
        return None

def is_numeric_type(data_type: str) -> bool:
    """
    Check if a SQL data type is numeric.
    
    Args:
        data_type (str): SQL data type
        
    Returns:
        bool: True if numeric, False otherwise
    """
    numeric_types = [
        'int', 'integer', 'tinyint', 'smallint', 'mediumint', 'bigint',
        'decimal', 'numeric', 'float', 'double', 'real'
    ]
    
    # Convert to lowercase and remove any parameters (e.g., decimal(10,2))
    base_type = re.sub(r'\(.*\)', '', data_type.lower()).strip()
    
    return any(base_type == t or base_type.startswith(t) for t in numeric_types)

def generate_synthetic_data(graph_data: dict, table_name: str, num_rows: int, output_db_path: str, 
                           original_db_path: str = None) -> bool:
    """
    Generate synthetic data based on graph information and create a new SQLite database.
    
    Args:
        graph_data (dict): Graph data generated by build_graph
        table_name (str): Name of the table to create
        num_rows (int): Number of rows to generate
        output_db_path (str): Path to the output SQLite database
        original_db_path (str): Path to the original database for extracting numeric stats
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create data generator
        generator = DataGenerator()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_db_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Connect to SQLite database
        conn = sqlite3.connect(output_db_path)
        cursor = conn.cursor()
        
        # Create table
        columns = graph_data.get('columns', [])
        if not columns:
            print("Error: No columns found in graph data")
            return False
        
        # If original database is provided, extract numeric column statistics
        numeric_columns = {}
        if original_db_path:
            extractor = DatabaseExtractor(db_path=original_db_path)
            
            # Get table schema to identify numeric columns
            try:
                schema = extractor.get_table_schema(table_name)
                
                # Find numeric columns
                for column_info in schema:
                    col_name = column_info.get('column_name', '')
                    data_type = column_info.get('data_type', '')
                    
                    if is_numeric_type(data_type):
                        print(f"Extracting statistics for numeric column: {col_name}")
                        try:
                            stats = extractor.get_numeric_column_stats(table_name, col_name)
                            numeric_columns[col_name] = stats
                            
                            # Store stats in the generator for later use
                            generator.set_numeric_stats(col_name, stats)
                            
                            print(f"  - Stats: min={stats.get('min')}, max={stats.get('max')}, mean={stats.get('mean')}, stddev={stats.get('stddev')}")
                        except Exception as e:
                            print(f"  - Error extracting stats: {e}")
            except Exception as e:
                print(f"Warning: Could not extract numeric column statistics: {e}")
        
        # Validate graph keys against available faker methods
        available_keys = set(generator.key_map.keys())
        invalid_keys_found = False
        
        for column in columns:
            col_name = column.get('name', '')
            graph_keys = column.get('graph', [])
            col_type = column.get('type', 'string')
            
            # Check for invalid keys
            invalid_keys = []
            for key in graph_keys:
                # Clean up key (remove spaces, etc.)
                clean_key = key.strip().lower()
                if clean_key not in available_keys:
                    invalid_keys.append(key)
            
            if invalid_keys:
                invalid_keys_found = True
                print(f"Warning: Column '{col_name}' contains invalid faker keys: {', '.join(invalid_keys)}")
                # Remove invalid keys
                column['graph'] = [key for key in graph_keys if key.strip().lower() in available_keys]
                print(f"  - Using only valid keys: {', '.join(column['graph'])}")
                
                # If no valid keys remain, add a default key based on column type
                if not column['graph']:
                    if col_type == 'numeric':
                        default_key = 'number'
                    else:
                        default_key = 'word'
                    column['graph'] = [default_key]
                    print(f"  - No valid keys remain, using default key: {default_key}")
        
        if invalid_keys_found:
            print("\nContinuing with valid keys only...\n")
        
        # Build CREATE TABLE statement
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
        column_defs = []
        
        for column in columns:
            col_name = column.get('name', '')
            col_type = column.get('type', 'string')
            
            # Determine SQL type based on column type
            if col_type == 'numeric':
                sql_type = "NUMERIC"
            else:
                sql_type = "TEXT"
                
            column_defs.append(f"  {col_name} {sql_type}")
        
        create_table_sql += ",\n".join(column_defs)
        create_table_sql += "\n);"
        
        # Execute CREATE TABLE
        print(f"Creating table {table_name}...")
        cursor.execute(create_table_sql)
        
        # Generate and insert data
        print(f"Generating {num_rows} rows of synthetic data...")
        
        # Prepare INSERT statement
        column_names = [column.get('name', '') for column in columns]
        placeholders = ", ".join(["?" for _ in column_names])
        insert_sql = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({placeholders})"
        
        # Generate and insert rows in batches
        batch_size = 10000
        for batch_start in range(0, num_rows, batch_size):
            batch_end = min(batch_start + batch_size, num_rows)
            batch_rows = []
            
            for _ in range(batch_start, batch_end):
                row_data = []
                
                for column in columns:
                    col_name = column.get('name', '')
                    graph_keys = column.get('graph', [])
                    mapping = column.get('mapping', 'none')
                    col_type = column.get('type', 'string')
                    
                    # Handle numeric columns differently
                    if col_type == 'numeric' and col_name in numeric_columns:
                        # Use numeric generation with stats from the original database
                        stats = numeric_columns[col_name]
                        
                        # Determine if we should generate an integer or float
                        is_int = False
                        for key in graph_keys:
                            if key.lower() == 'integer':
                                is_int = True
                                break
                        
                        # Generate the numeric value
                        try:
                            value = generator.generate_number(
                                stats=stats,
                                distribution='auto' if 'mean' in stats and 'stddev' in stats else 'uniform',
                                as_int=is_int,
                                round_to=0 if is_int else 2
                            )
                        except Exception as e:
                            print(f"Error generating numeric value for {col_name}: {e}")
                            # Fallback to a simple random number
                            value = random.randint(0, 100) if is_int else round(random.uniform(0, 100), 2)
                        
                        row_data.append(value)
                    else:
                        # Generate value by concatenating results from each graph key
                        value_parts = []
                        for key in graph_keys:
                            try:
                                # Clean up key (remove spaces, etc.)
                                clean_key = key.strip().lower()
                                
                                # Check if this is a numeric key
                                if clean_key in ['number', 'integer', 'decimal', 'float']:
                                    # Generate a numeric value
                                    if col_name in numeric_columns:
                                        # Use stats from the original database
                                        stats = numeric_columns[col_name]
                                        value_part = generator.get_for_key(
                                            clean_key, 
                                            column_name=col_name,
                                            as_int=clean_key == 'integer',
                                            round_to=0 if clean_key == 'integer' else 2
                                        )
                                    else:
                                        # Use default parameters
                                        value_part = generator.get_for_key(
                                            clean_key,
                                            as_int=clean_key == 'integer',
                                            round_to=0 if clean_key == 'integer' else 2
                                        )
                                else:
                                    # Use standard faker method
                                    value_part = generator.get_for_key(clean_key)
                                
                                value_parts.append(str(value_part))
                            except ValueError as e:
                                print(f"Unexpected error with key '{key}' for column '{col_name}': {e}")
                                value_parts.append(f"[{key}]")
                        
                        # Concatenate the parts
                        value = " ".join(value_parts)
                        
                        # Apply mapping if specified
                        if mapping == 'uppercase':
                            value = value.upper()
                        elif mapping == 'lowercase':
                            value = value.lower()
                        elif mapping == 'titlecase':
                            value = value.title()
                        
                        row_data.append(value)
                
                batch_rows.append(row_data)
            
            # Insert batch
            cursor.executemany(insert_sql, batch_rows)
            conn.commit()
            print(f"  Inserted rows {batch_start+1} to {batch_end}")
        
        # Close connection
        conn.close()
        
        print(f"Successfully generated {num_rows} rows of synthetic data in {output_db_path}")
        return True
    
    except Exception as e:
        print(f"Error generating synthetic data: {e}")
        return False

def validate_graph(graph_data: dict, table_name: str, original_db_path: str, sample_size: int = 5) -> tuple:
    """
    Validate the generated graph by creating a small sample of synthetic data
    and comparing it to the original data to check for correctness.
    
    Args:
        graph_data (dict): Graph data generated by build_graph
        table_name (str): Name of the table to validate
        original_db_path (str): Path to the original database
        sample_size (int): Number of sample rows to generate for validation
        
    Returns:
        tuple: (is_valid, validation_report, improved_graph, feedback)
    """
    print(f"\n=== Validating graph with {sample_size} sample rows ===")
    
    # Create a temporary database for validation
    temp_db_path = "temp_validation.sqlite"
    
    try:
        # Generate a small sample of synthetic data
        print("  - Generating synthetic sample data...")
        success = generate_synthetic_data(
            graph_data=graph_data,
            table_name=table_name,
            num_rows=sample_size,
            output_db_path=temp_db_path,
            original_db_path=original_db_path
        )
        
        if not success:
            print("  - Failed to generate validation data")
            return False, "Failed to generate validation data", None, None
        
        print("  - Successfully generated sample data")
        print("  - Comparing with original data...")
        
        # Connect to both databases
        original_conn = sqlite3.connect(original_db_path)
        original_cursor = original_conn.cursor()
        
        synthetic_conn = sqlite3.connect(temp_db_path)
        synthetic_cursor = synthetic_conn.cursor()
        
        # Get column names from graph data
        columns = [col.get('name', '') for col in graph_data.get('columns', [])]
        
        # Get sample rows from original database
        original_cursor.execute(f"SELECT {', '.join(columns)} FROM {table_name} LIMIT {sample_size}")
        original_rows = original_cursor.fetchall()
        
        # Get all rows from synthetic database
        synthetic_cursor.execute(f"SELECT {', '.join(columns)} FROM {table_name}")
        synthetic_rows = synthetic_cursor.fetchall()
        
        # Close connections
        original_conn.close()
        synthetic_conn.close()
        
        print(f"  - Retrieved {len(original_rows)} original rows and {len(synthetic_rows)} synthetic rows")
        
        # Compare data patterns
        validation_report = {
            "columns": {},
            "overall_match": True,
            "issues": []
        }
        
        # Track columns that need improvement
        columns_to_improve = []
        
        print("  - Analyzing data patterns by column:")
        
        for i, col_name in enumerate(columns):
            # Get original and synthetic values for this column
            original_values = [row[i] for row in original_rows]
            synthetic_values = [row[i] for row in synthetic_rows]
            
            # Basic type validation
            original_types = set(type(val) for val in original_values if val is not None)
            synthetic_types = set(type(val) for val in synthetic_values if val is not None)
            
            type_match = len(original_types.intersection(synthetic_types)) > 0
            
            # Pattern validation (for strings)
            pattern_match = True
            pattern_issues = []
            
            # Find the corresponding column in the graph data
            graph_column = next((col for col in graph_data.get('columns', []) if col.get('name', '') == col_name), None)
            
            if graph_column:
                col_type = graph_column.get('type', 'string')
                
                if col_type == 'string':
                    # Check for case consistency (uppercase, lowercase, titlecase)
                    original_case_patterns = []
                    for val in original_values:
                        if val and isinstance(val, str):
                            if val.isupper():
                                original_case_patterns.append('uppercase')
                            elif val.islower():
                                original_case_patterns.append('lowercase')
                            elif val.istitle():
                                original_case_patterns.append('titlecase')
                            else:
                                original_case_patterns.append('mixed')
                    
                    # If there's a consistent case pattern in the original data
                    if original_case_patterns and len(set(original_case_patterns)) == 1:
                        dominant_case = original_case_patterns[0]
                        
                        # Check if synthetic data follows the same pattern
                        synthetic_case_matches = 0
                        for val in synthetic_values:
                            if val and isinstance(val, str):
                                if (dominant_case == 'uppercase' and val.isupper()) or \
                                   (dominant_case == 'lowercase' and val.islower()) or \
                                   (dominant_case == 'titlecase' and val.istitle()):
                                    synthetic_case_matches += 1
                        
                        case_match_percentage = synthetic_case_matches / len(synthetic_values) if synthetic_values else 0
                        
                        if case_match_percentage < 0.8:  # Less than 80% match
                            pattern_match = False
                            pattern_issues.append(f"Case pattern mismatch: original={dominant_case}, synthetic match rate={case_match_percentage:.2f}")
                            
                            # Add to columns to improve
                            columns_to_improve.append({
                                "name": col_name,
                                "issue": "case_pattern",
                                "original_pattern": dominant_case,
                                "current_mapping": graph_column.get('mapping', 'none')
                            })
                
                # For numeric columns, check range consistency
                elif col_type == 'numeric':
                    original_nums = [float(val) if val is not None else None for val in original_values]
                    synthetic_nums = [float(val) if val is not None else None for val in synthetic_values]
                    
                    original_nums = [val for val in original_nums if val is not None]
                    synthetic_nums = [val for val in synthetic_nums if val is not None]
                    
                    if original_nums and synthetic_nums:
                        orig_min, orig_max = min(original_nums), max(original_nums)
                        synth_min, synth_max = min(synthetic_nums), max(synthetic_nums)
                        
                        # Check if ranges are significantly different
                        if orig_min > 0 and synth_min < 0:
                            pattern_match = False
                            pattern_issues.append(f"Sign mismatch: original values are positive, synthetic contains negative")
                            columns_to_improve.append({
                                "name": col_name,
                                "issue": "numeric_sign",
                                "original_range": f"{orig_min} to {orig_max}",
                                "synthetic_range": f"{synth_min} to {synth_max}"
                            })
                        
                        # Check if integers vs decimals
                        original_integers = all(val.is_integer() for val in original_nums if val is not None)
                        synthetic_integers = all(val.is_integer() for val in synthetic_nums if val is not None)
                        
                        if original_integers != synthetic_integers:
                            pattern_match = False
                            pattern_issues.append(f"Integer/decimal mismatch: original={'integers' if original_integers else 'decimals'}, synthetic={'integers' if synthetic_integers else 'decimals'}")
                            columns_to_improve.append({
                                "name": col_name,
                                "issue": "numeric_type",
                                "original_type": "integer" if original_integers else "decimal",
                                "current_type": "integer" if synthetic_integers else "decimal"
                            })
            
            # Store validation results for this column
            validation_report["columns"][col_name] = {
                "type_match": type_match,
                "pattern_match": pattern_match,
                "issues": pattern_issues
            }
            
            # Update overall match status
            if not type_match or not pattern_match:
                validation_report["overall_match"] = False
                validation_report["issues"].append(f"Column '{col_name}' has validation issues")
                print(f"    * {col_name}: ❌ Issues detected")
                for issue in pattern_issues:
                    print(f"      - {issue}")
            else:
                print(f"    * {col_name}: ✓ Looks good")
        
        # If there are issues, try to improve the graph
        improved_graph = None
        feedback = None
        
        if not validation_report["overall_match"] and columns_to_improve:
            print("\n  - Attempting to improve graph based on validation results")
            
            # Create a copy of the original graph
            improved_graph = copy.deepcopy(graph_data)
            
            # Generate feedback for rebuilding the graph
            feedback_lines = ["Please fix the following issues in the graph:"]
            
            # Apply fixes to the graph
            for column_issue in columns_to_improve:
                col_name = column_issue["name"]
                issue_type = column_issue["issue"]
                
                # Find the column in the improved graph
                for col in improved_graph.get('columns', []):
                    if col.get('name', '') == col_name:
                        if issue_type == "case_pattern":
                            # Fix case mapping
                            original_pattern = column_issue["original_pattern"]
                            print(f"    * Fixing case pattern for '{col_name}': setting mapping to '{original_pattern}'")
                            col['mapping'] = original_pattern
                            
                            feedback_lines.append(f"- Column '{col_name}' should use '{original_pattern}' mapping instead of '{column_issue['current_mapping']}'")
                        
                        elif issue_type == "numeric_type":
                            # Fix numeric type (integer vs decimal)
                            original_type = column_issue["original_type"]
                            print(f"    * Fixing numeric type for '{col_name}': setting to '{original_type}'")
                            
                            # Update the graph keys
                            if original_type == "integer":
                                # Replace any float/decimal keys with integer
                                col['graph'] = ['integer' if k.lower() in ['float', 'decimal', 'number'] else k for k in col['graph']]
                                feedback_lines.append(f"- Column '{col_name}' should use 'integer' key instead of 'float' or 'decimal' (values are whole numbers)")
                            else:
                                # Replace any integer keys with decimal
                                col['graph'] = ['decimal' if k.lower() in ['integer', 'number'] else k for k in col['graph']]
                                feedback_lines.append(f"- Column '{col_name}' should use 'decimal' or 'float' key instead of 'integer' (values have decimal points)")
                        
                        elif issue_type == "numeric_sign":
                            # Fix numeric sign issues
                            print(f"    * Fixing numeric sign for '{col_name}': ensuring positive values")
                            feedback_lines.append(f"- Column '{col_name}' should only generate positive values (original range: {column_issue['original_range']})")
                            # This will be handled by the generator with the correct stats
                            pass
                        
                        break
            
            # Create feedback string
            feedback = "\n".join(feedback_lines)
        
        # Clean up temporary database
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
            print("  - Cleaned up temporary validation database")
        
        return validation_report["overall_match"], validation_report, improved_graph, feedback
    
    except Exception as e:
        print(f"  - Error validating graph: {e}")
        
        # Clean up temporary database
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
                print("  - Cleaned up temporary validation database")
            except:
                pass
        
        return False, f"Validation error: {str(e)}", None, None

def main():
    # Check if table name is provided as command-line argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <table_name> [<string_column_name>]")
        print("       python main.py <table_name> --test")
        print("       python main.py --test-build-graph <table_name>")
        print("       python main.py --generate <table_name> <num_rows> [<output_db_path>] [--max-retries <num>]")
        return 1
    
    # Check for generate mode
    if sys.argv[1] == "--generate":
        if len(sys.argv) < 4:
            print("Usage: python main.py --generate <table_name> <num_rows> [<output_db_path>] [--max-retries <num>]")
            return 1
        
        table_name = sys.argv[2]
        try:
            num_rows = int(sys.argv[3])
        except ValueError:
            print(f"Error: Number of rows '{sys.argv[3]}' is not a valid integer")
            return 1
        
        # Set default output path if not provided
        output_db_path = "output/synthetic_data.sqlite"
        max_retries = 3  # Default max retries
        
        # Parse additional arguments
        i = 4
        while i < len(sys.argv):
            if i < len(sys.argv) and sys.argv[i] != "--max-retries":
                output_db_path = sys.argv[i]
            elif i < len(sys.argv) and sys.argv[i] == "--max-retries" and i + 1 < len(sys.argv):
                try:
                    max_retries = int(sys.argv[i + 1])
                    i += 1  # Skip the next argument as we've processed it
                except ValueError:
                    print(f"Error: Max retries '{sys.argv[i + 1]}' is not a valid integer")
                    return 1
            i += 1
        
        print(f"Configuration:")
        print(f"  - Table: {table_name}")
        print(f"  - Rows to generate: {num_rows}")
        print(f"  - Output database: {output_db_path}")
        print(f"  - Maximum validation retries: {max_retries}")
        
        # Get SQLite database path
        db_path = get_db_path()
        if db_path is None:
            return 1
        
        # Create DatabaseExtractor instance
        extractor = DatabaseExtractor(db_path=db_path)
        
        try:
            # Get table information
            print(f"\n=== Extracting table information for {table_name} ===")
            table_info = extractor.get_table_info(table_name)
            print(f"  - Found {len(table_info['columns'])} columns")
            print(f"  - Extracted {len(table_info['sample_data'])} sample rows")
            
            # Generate graph using LLM
            print(f"\n=== Generating initial graph for {table_name} ===")
            graph_result = build_graph(
                table_name=table_info['table_name'],
                columns=table_info['columns'],
                sample_data=table_info['sample_data']
            )
            
            # Parse the result
            try:
                graph_data = json.loads(graph_result)
                print("\n=== Generated Graph ===")
                print_json(graph_data)
                
                # Track retry count
                retry_count = 0
                is_valid = False
                
                # Store all versions of the graph for later selection
                graph_versions = {
                    "original": graph_data
                }
                
                while not is_valid and retry_count < max_retries:
                    print(f"\n=== Validation attempt {retry_count + 1}/{max_retries} ===")
                    
                    # Validate the current graph with a small sample
                    print(f"  - Generating {5} sample rows for validation...")
                    is_valid, validation_report, improved_graph, feedback = validate_graph(
                        graph_data=graph_data,
                        table_name=table_name,
                        original_db_path=db_path,
                        sample_size=5  # Generate 5 rows for validation
                    )
                    
                    if is_valid:
                        print("  - Validation successful! The graph produces data that matches the original patterns.")
                        break
                    
                    print("  - Validation found issues with the generated data.")
                    
                    # If we have an improved graph from automatic fixes
                    if improved_graph:
                        print("\n=== Applying automatic fixes to the graph ===")
                        print("  - The system has automatically identified and fixed some issues:")
                        
                        # Print what was fixed
                        for col_name, col_info in validation_report["columns"].items():
                            if col_info["issues"]:
                                for issue in col_info["issues"]:
                                    print(f"    * {col_name}: {issue}")
                        
                        # Store this version
                        graph_versions["auto_fixed"] = improved_graph
                        
                        # Update the current graph
                        graph_data = improved_graph
                        
                        # Validate the improved graph
                        print("\n=== Validating improved graph ===")
                        print(f"  - Generating {5} sample rows for validation...")
                        is_valid, validation_report, _, _ = validate_graph(
                            graph_data=improved_graph,
                            table_name=table_name,
                            original_db_path=db_path,
                            sample_size=5
                        )
                        
                        if is_valid:
                            print("  - Validation successful after automatic fixes!")
                            break
                        else:
                            print("  - Automatic fixes were not sufficient to resolve all issues.")
                    
                    # If we still have issues and have feedback for the LLM
                    if not is_valid and feedback:
                        print("\n=== Rebuilding graph with LLM using validation feedback ===")
                        print("  - Sending the following feedback to the LLM:")
                        for line in feedback.split('\n'):
                            print(f"    {line}")
                        
                        # Rebuild the graph with feedback
                        print("  - Waiting for LLM to generate improved graph...")
                        new_graph_result = build_graph(
                            table_name=table_info['table_name'],
                            columns=table_info['columns'],
                            sample_data=table_info['sample_data'],
                            feedback=feedback
                        )
                        
                        try:
                            new_graph_data = json.loads(new_graph_result)
                            print("\n=== LLM Rebuilt Graph ===")
                            print_json(new_graph_data)
                            
                            # Store this version
                            graph_versions["llm_rebuilt"] = new_graph_data
                            
                            # Update the current graph
                            graph_data = new_graph_data
                            
                            # Validate the new graph
                            print("\n=== Validating LLM rebuilt graph ===")
                            print(f"  - Generating {5} sample rows for validation...")
                            is_valid, validation_report, _, _ = validate_graph(
                                graph_data=new_graph_data,
                                table_name=table_name,
                                original_db_path=db_path,
                                sample_size=5
                            )
                            
                            if is_valid:
                                print("  - Validation successful after LLM rebuilding!")
                                break
                            else:
                                print("  - LLM rebuilding did not resolve all issues.")
                                print("  - Remaining issues:")
                                for issue in validation_report["issues"]:
                                    print(f"    * {issue}")
                        
                        except json.JSONDecodeError:
                            print("  - Error: Could not parse LLM rebuilt graph as JSON")
                            print("  - Raw response:")
                            print(new_graph_result)
                    
                    # Increment retry count
                    retry_count += 1
                    
                    if retry_count < max_retries and not is_valid:
                        print(f"\n=== Retry {retry_count + 1}/{max_retries} ===")
                        print("  - Previous validation attempt failed.")
                        print("  - Attempting another validation cycle...")
                
                # If we've exhausted retries or have validation issues
                if not is_valid:
                    print("\n=== Validation unsuccessful after maximum retries ===")
                    print("  - The system was unable to automatically generate a perfect graph.")
                    print("  - You can choose which version of the graph to use or abort.")
                    
                    # Let user choose which graph to use
                    print("\nAvailable graph versions:")
                    options = []
                    
                    print("1. Original graph (first attempt)")
                    options.append(("original", graph_versions["original"]))
                    
                    option_num = 2
                    if "auto_fixed" in graph_versions:
                        print(f"{option_num}. Auto-fixed graph (with automatic fixes)")
                        options.append(("auto_fixed", graph_versions["auto_fixed"]))
                        option_num += 1
                    
                    if "llm_rebuilt" in graph_versions:
                        print(f"{option_num}. LLM-rebuilt graph (with LLM fixes)")
                        options.append(("llm_rebuilt", graph_versions["llm_rebuilt"]))
                    
                    choice = input("\nWhich graph would you like to use? (enter number or 'abort' to cancel): ")
                    
                    if choice.lower() == 'abort':
                        print("Aborting data generation.")
                        return 1
                    
                    try:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(options):
                            choice_name, graph_data = options[choice_idx]
                            print(f"Using {choice_name} graph for data generation.")
                        else:
                            print(f"Invalid choice. Using the last attempted graph.")
                    except ValueError:
                        print(f"Invalid choice. Using the last attempted graph.")
                
                # Final confirmation before generating full dataset
                response = input(f"\nReady to generate {num_rows} rows of synthetic data. Continue? (y/n): ")
                if response.lower() not in ['y', 'yes']:
                    print("Aborting data generation.")
                    return 1
                
                # Generate synthetic data
                print(f"\n=== Generating {num_rows} rows of synthetic data for {table_name} ===")
                print(f"  - Using the {'validated' if is_valid else 'selected'} graph")
                print(f"  - Output database: {output_db_path}")
                print(f"  - This may take a while for large datasets...")
                
                success = generate_synthetic_data(
                    graph_data=graph_data,
                    table_name=table_name,
                    num_rows=num_rows,
                    output_db_path=output_db_path,
                    original_db_path=db_path  # Pass the original database path for numeric stats
                )
                
                if success:
                    print(f"\n=== Data generation completed successfully ===")
                    print(f"  - Generated {num_rows} rows for table '{table_name}'")
                    print(f"  - Output database: {output_db_path}")
                else:
                    print("\n=== Data generation failed ===")
                    print("  - Check the error messages above for details.")
                    return 1
                
            except json.JSONDecodeError:
                print("Error: Could not parse LLM response as JSON")
                print("Raw response:")
                print(graph_result)
                return 1
                
        except Exception as e:
            print(f"Error: {e}")
            return 1
        
        return 0
    
    # Get table name from command-line arguments
    table_name = sys.argv[1]
    
    # Check if we should run in test mode
    test_mode = False
    if len(sys.argv) >= 3 and sys.argv[2] == "--test":
        test_mode = True
        
    # If in test mode, just run the test function
    if test_mode:
        test_build_graph(table_name)
        return 0
    
    # Get optional string column name
    string_column = None
    if len(sys.argv) >= 3 and not test_mode:
        string_column = sys.argv[2]
    
    # Get SQLite database path
    db_path = get_db_path()
    if db_path is None:
        return 1
    
    # Create DatabaseExtractor instance
    extractor = DatabaseExtractor(db_path=db_path)
    
    try:
        # Test get_table_schema method
        print(f"\n=== Getting schema for table: {table_name} ===")
        schema = extractor.get_table_schema(table_name)
        print(f"Found {len(schema)} columns in {table_name}:")
        for column in schema:
            print(f"  - {column['column_name']} ({column['data_type']})")
        print()
        
        # Find a string column if not provided
        if string_column is None:
            for column in schema:
                if column['data_type'].lower() in ['text', 'varchar', 'character', 'char', 'string']:
                    string_column = column['column_name']
                    print(f"Found string column: {string_column}")
                    break
        
        if not string_column:
            print("No string column found or provided. Exiting.")
            return 1
            
        # Get comprehensive table information
        print(f"\n=== Getting table information for {table_name} ===")
        table_info = extractor.get_table_info(table_name)
        
        # Generate graph using LLM
        print(f"\n=== Generating graph for {table_name} ===")
        graph_result = build_graph(
            table_name=table_info['table_name'],
            columns=table_info['columns'],
            sample_data=table_info['sample_data']
        )
        
        # Parse and display the result
        try:
            graph_data = json.loads(graph_result)
            print("\n=== Generated Graph ===")
            print_json(graph_data)
        except json.JSONDecodeError:
            print("Error: Could not parse LLM response as JSON")
            print("Raw response:")
            print(graph_result)
        
        # Continue with the rest of the analysis if needed
        # Test numeric column stats
        print(f"\n=== Testing numeric column stats ===")
        # Try to find a numeric column from the schema
        numeric_column = None
        for column in schema:
            if column['data_type'].lower() in ['integer', 'int', 'bigint', 'numeric', 'real', 'double', 'float']:
                numeric_column = column['column_name']
                break
                
        if numeric_column:
            print(f"Found numeric column: {numeric_column}")
            try:
                stats = extractor.get_numeric_column_stats(table_name, numeric_column)
                print(f"Statistics for {numeric_column} column:")
                for key, value in stats.items():
                    print(f"  - {key}: {value}")
            except Exception as e:
                print(f"Error getting numeric stats: {e}")
        else:
            print(f"No numeric column found in {table_name} table.")
        
        # Test string column analysis
        sample_size = 100  # Use a smaller sample size for testing
        
        # Run the full string column analysis (without embeddings)
        print(f"\n=== Running string column analysis for {string_column} ===")
        try:
            full_analysis = extractor.get_string_column_analysis(
                table_name, 
                string_column, 
                sample_size=sample_size, 
                do_embedding=False  # Skip embeddings
            )
            print("Analysis completed successfully.")
            print("Summary of results:")
            print(f"  - Length: min={full_analysis['length_distribution']['min_length']}, max={full_analysis['length_distribution']['max_length']}, mean={full_analysis['length_distribution']['mean_length']:.2f}")
            print(f"  - Entropy: value={full_analysis['entropy']['value_entropy']:.4f}, char={full_analysis['entropy']['char_entropy']:.4f}")
            print(f"  - Categorical: unique={full_analysis['categorical_analysis']['unique_values']}, is_categorical={full_analysis['categorical_analysis']['is_likely_categorical']}")
            
            # Ask if user wants to see full JSON output
            response = input("\nDo you want to see the full JSON output? (y/n): ")
            if response.lower() in ['y', 'yes']:
                print("\nFull analysis results:")
                print_json(full_analysis)
        except Exception as e:
            print(f"Error running full analysis: {e}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Check if we want to run the test directly
    if len(sys.argv) >= 2 and sys.argv[1] == "--test-build-graph":
        if len(sys.argv) >= 3:
            table_name = sys.argv[2]
            test_build_graph(table_name)
        else:
            print("Usage: python main.py --test-build-graph <table_name>")
    else:
        sys.exit(main())
