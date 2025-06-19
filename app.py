#!/usr/bin/env python3
"""
pgvector-hybrid-search-demo-py - Production-Ready Flask API Server
PostgreSQL 16 with Advanced Analytics
"""

import os
import json
from datetime import datetime
from decimal import Decimal
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

def get_db_connection():
    """Create database connection with proper error handling"""
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except psycopg2.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise

def serialize_result(obj):
    """JSON serializer for database results"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

@app.route('/')
def home():
    """API documentation homepage"""
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>pgvector-hybrid-search-demo-py API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 15px 0; border-left: 4px solid #3498db; }
            .method { background: #3498db; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
            pre { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 4px; overflow-x: auto; }
            .status { color: #27ae60; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>pgvector-hybrid-search-demo-py</h1>
            <p><span class="status">✓ ONLINE</span> | PostgreSQL 16 Analytics API</p>
            
            <h2>API Endpoints</h2>
            
            <div class="endpoint">
                <p><span class="method">GET</span> <strong>/health</strong></p>
                <p>Health check and database connectivity status</p>
            </div>

            
            <div class="endpoint">
                <p><span class="method">GET</span> <strong>/api/analytics/summary</strong></p>
                <p>Time-series analytics summary and metrics</p>
            </div>

            <div class="endpoint">
                <p><span class="method">GET</span> <strong>/api/data/recent</strong></p>
                <p>Recent time-series data points</p>
                <pre>?hours=24&metric=temperature&sensor_id=123</pre>
            </div>
            

            <h2>Database Info</h2>
            <pre id="dbInfo">Loading database information...</pre>
            
            <script>
                fetch('/health')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('dbInfo').textContent = JSON.stringify(data, null, 2);
                    })
                    .catch(error => {
                        document.getElementById('dbInfo').textContent = 'Error loading database info: ' + error;
                    });
            </script>
        </div>
    </body>
    </html>
    '''
    return render_template_string(html_template)

@app.route('/health')
def health_check():
    """Comprehensive health check endpoint"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Basic connectivity test
                cur.execute("SELECT version(), current_database(), current_user, NOW();")
                db_info = cur.fetchone()
                
                
                # Check for time-series optimizations
                cur.execute("""
                    SELECT schemaname, tablename, indexname 
                    FROM pg_indexes 
                    WHERE tablename LIKE '%time%' OR tablename LIKE '%data%'
                    LIMIT 5;
                """)
                indexes = cur.fetchall()
                
                
                return jsonify({
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "database": {
                        "version": db_info['version'],
                        "database": db_info['current_database'],
                        "user": db_info['current_user'],
                        "server_time": db_info['now'].isoformat()
                    },
                    
                    "performance": {
                        "indexes_found": len(indexes),
                        "time_series_optimized": len(indexes) > 0
                    }
                    
                })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/api/analytics/summary')
def analytics_summary():
    """Time-series analytics summary and metrics"""
    try:
        hours = int(request.args.get('hours', 24))
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Sample analytics for time-series data
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT sensor_id) as unique_sensors,
                        MIN(recorded_at) as earliest_record,
                        MAX(recorded_at) as latest_record,
                        AVG(value) as avg_value,
                        MIN(value) as min_value,
                        MAX(value) as max_value
                    FROM sensor_data 
                    WHERE recorded_at >= NOW() - INTERVAL '%s hours';
                """, (hours,))
                
                summary = cur.fetchone()
                
                return jsonify({
                    "time_window_hours": hours,
                    "summary": dict(summary) if summary else {},
                    "generated_at": datetime.now().isoformat()
                }, default=serialize_result)
                
    except Exception as e:
        logger.error(f"Analytics summary failed: {e}")
        return jsonify({"error": "Analytics unavailable"}), 500

@app.route('/api/data/recent')
def recent_data():
    """Get recent time-series data points"""
    try:
        hours = int(request.args.get('hours', 24))
        metric = request.args.get('metric')
        sensor_id = request.args.get('sensor_id')
        limit = min(int(request.args.get('limit', 100)), 1000)
        
        conditions = ["recorded_at >= NOW() - INTERVAL '%s hours'"]
        params = [hours]
        
        if metric:
            conditions.append("metric_name = %s")
            params.append(metric)
        if sensor_id:
            conditions.append("sensor_id = %s")
            params.append(sensor_id)
            
        params.append(limit)
        
        query = f"""
            SELECT sensor_id, metric_name, value, tags, recorded_at
            FROM sensor_data 
            WHERE {' AND '.join(conditions)}
            ORDER BY recorded_at DESC
            LIMIT %s;
        """
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                data_points = cur.fetchall()
                
                return jsonify({
                    "filters": {
                        "hours": hours,
                        "metric": metric,
                        "sensor_id": sensor_id
                    },
                    "total_points": len(data_points),
                    "data": [dict(point) for point in data_points]
                }, default=serialize_result)
                
    except Exception as e:
        logger.error(f"Recent data fetch failed: {e}")
        return jsonify({"error": "Data fetch failed"}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Production WSGI server should be used in deployment
    # This is for development only
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting pgvector-hybrid-search-demo-py API server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
