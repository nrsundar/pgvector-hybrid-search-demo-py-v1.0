#!/usr/bin/env python3
"""
Module 09: PostgreSQL Advanced Database Operations
Production-ready PostgreSQL administration and optimization
"""

import psycopg2
import pandas as pd
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

class PostgreSQLModule:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'postgres'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'password'),
            cursor_factory=RealDictCursor
        )

    def setup_database_extensions(self):
        """Install and configure PostgreSQL extensions"""
        cursor = self.conn.cursor()
        
        extensions = ['uuid-ossp', 'pgcrypto', 'pg_stat_statements', 'pg_trgm']
        
        for ext in extensions:
            try:
                cursor.execute(f"CREATE EXTENSION IF NOT EXISTS {ext};")
                print(f"✓ Extension {ext} installed")
            except psycopg2.Error as e:
                print(f"✗ Failed to install {ext}: {e}")
        
        self.conn.commit()
        cursor.close()

    def create_application_tables(self):
        """Create sample application tables with proper indexing"""
        cursor = self.conn.cursor()
        
        # Users table with UUID primary key
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR(255) UNIQUE NOT NULL,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                is_active BOOLEAN DEFAULT true
            );
        """)
        
        # Orders table with foreign key relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id SERIAL PRIMARY KEY,
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                order_total DECIMAL(10,2) NOT NULL,
                order_status VARCHAR(50) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Create optimized indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
            CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);
            CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders (user_id);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (order_status);
            CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders (created_at);
        """)
        
        self.conn.commit()
        cursor.close()
        print("✓ Application tables and indexes created")

    def demonstrate_advanced_queries(self):
        """Showcase advanced PostgreSQL query techniques"""
        cursor = self.conn.cursor()
        
        # Database information query
        cursor.execute("""
            SELECT 
                current_database() as database_name,
                current_user as connected_user,
                version() as postgresql_version,
                NOW() as current_timestamp,
                pg_size_pretty(pg_database_size(current_database())) as database_size
        """)
        
        result = cursor.fetchone()
        if result:
            print(f"Database: {result['database_name']}")
            print(f"User: {result['connected_user']}")
            print(f"Version: {result['postgresql_version']}")
            print(f"Size: {result['database_size']}")
        
        cursor.close()
        return result

    def analyze_database_performance(self):
        """Analyze database performance metrics"""
        cursor = self.conn.cursor()
        
        # Database size and table statistics
        cursor.execute("""
            SELECT 
                schemaname,
                tablename,
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as table_size
            FROM pg_stat_user_tables
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
        """)
        
        results = cursor.fetchall()
        
        print("Database Performance Metrics:")
        print("=" * 40)
        for row in results:
            table_name = row['tablename'] or 'unknown'
            table_size = row['table_size'] or '0 bytes'
            inserts = row['inserts'] or 0
            updates = row['updates'] or 0
            print(f"Table: {table_name}")
            print(f"  Size: {table_size}")
            print(f"  Activity: {inserts} inserts, {updates} updates")
            print()
        
        cursor.close()

    def run_database_demo(self):
        """Execute complete database demonstration"""
        print(f"Starting Module 09 - PostgreSQL Advanced Operations")
        print("=" * 60)
        
        self.setup_database_extensions()
        self.create_application_tables()
        self.demonstrate_advanced_queries()
        self.analyze_database_performance()
        
        print(f"Module 09 database operations completed successfully!")

if __name__ == "__main__":
    demo = PostgreSQLModule()
    demo.run_database_demo()
