#!/usr/bin/env python3
import sqlite3
import os
from src.data_generator import DataGenerator

def create_test_database():
    """
    Creates a test database with a users table and populates it with 10 rows of sample data.
    Using SQLite and storing the database file in a 'db' directory.
    """
    # Create db directory if it doesn't exist
    os.makedirs('db', exist_ok=True)
    
    # Database file path
    db_path = 'db/test_db.sqlite'
    
    # Connect to the SQLite database (will create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create users table - SQLite can only execute one statement at a time
    # First drop the table if it exists
    cursor.execute("DROP TABLE IF EXISTS users;")
    
    # Then create the new table
    cursor.execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        phone_number TEXT,
        address TEXT,
        city TEXT,
        state TEXT,
        zipcode TEXT,
        country TEXT,
        job_title TEXT,
        company TEXT,
        username TEXT UNIQUE,
        date_of_birth DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Initialize data generator
    generator = DataGenerator(seed=42)  # Set seed for reproducibility
    
    # Generate and insert 10 users
    for i in range(10):
        user_data = {
            'first_name': generator.get_for_key('first_name'),
            'last_name': generator.get_for_key('last_name'),
            'email': generator.get_for_key('email'),
            'phone_number': generator.get_for_key('phone_number'),
            'address': generator.get_for_key('street_address'),
            'city': generator.get_for_key('city'),
            'state': generator.get_for_key('state'),
            'zipcode': generator.get_for_key('zipcode'),
            'country': generator.get_for_key('country'),
            'job_title': generator.get_for_key('job'),
            'company': generator.get_for_key('company'),
            'username': generator.get_for_key('username'),
            'date_of_birth': generator.get_for_key('date')
        }
        
        # Insert the user data
        cursor.execute("""
        INSERT INTO users (
            first_name, last_name, email, phone_number, address, city, state, zipcode, 
            country, job_title, company, username, date_of_birth
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """, (
            user_data['first_name'], user_data['last_name'], user_data['email'],
            user_data['phone_number'], user_data['address'], user_data['city'],
            user_data['state'], user_data['zipcode'], user_data['country'],
            user_data['job_title'], user_data['company'], user_data['username'],
            user_data['date_of_birth']
        ))
    
    # Commit the changes
    conn.commit()
    
    # Verify data was inserted
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    print(f"Successfully created users table with {count} sample records")
    print(f"Database created at: {os.path.abspath(db_path)}")
    
    # Close the connection
    cursor.close()
    conn.close()

if __name__ == "__main__":
    create_test_database()
