# Production Dataset Sources and Download Guide

## Overview

This repository includes sample data for demonstration purposes. For production use cases and comprehensive testing, we provide access to larger, realistic datasets that better represent real-world scenarios.

## Quick Start with Sample Data

The repository includes basic sample data that you can use immediately:

- **Location**: `data/sample_data.csv`
- **Size**: ~100 records
- **Purpose**: Initial testing and demonstration

## Production-Scale Datasets

### AWS S3 Data Repository

We maintain production-scale datasets in Amazon S3 for each use case:

```
Bucket: postgresql-demo-datasets
Region: us-west-2
Access: Public read access
```









## Performance Considerations

### Loading Large Datasets

1. **Use COPY for bulk imports** (10x faster than INSERT)
2. **Disable autocommit** for large transactions
3. **Increase work_mem** temporarily during import
4. **Create indexes after** data loading
5. **Use parallel processing** for very large files

```sql
-- Optimize PostgreSQL for bulk loading
SET maintenance_work_mem = '1GB';
SET max_wal_size = '2GB';
SET checkpoint_completion_target = 0.9;
SET wal_buffers = '32MB';
```

### Storage Optimization

- **Partitioning**: Implement time-based or hash partitioning for large tables
- **Compression**: Use appropriate data types and consider table compression
- **Archival**: Move old data to cheaper storage tiers

## Data Quality and Validation

### Automated Checks

All production datasets include:

- **Schema validation**: Consistent field types and formats
- **Completeness checks**: No missing critical fields
- **Referential integrity**: Valid foreign key relationships
- **Business rule validation**: Realistic value ranges and constraints

### Data Freshness

- **Property data**: Updated monthly from MLS feeds
- **Time-series data**: Real-time streaming with 5-minute lag
- **Vector embeddings**: Regenerated quarterly with latest models
- **Configuration data**: Updated weekly from production systems

## Cost Management

### S3 Transfer Costs

- **Same region transfers**: Free within us-west-2
- **Cross-region**: $0.02/GB for data transfer out
- **Internet egress**: First 1GB free, then $0.05-0.09/GB

### Optimization Tips

1. **Download only needed subsets** using S3 prefix filters
2. **Use CloudFront** for frequently accessed data
3. **Enable S3 compression** for text-based datasets
4. **Implement caching** for repeated downloads

```bash
# Download only specific time ranges
aws s3 cp s3://postgresql-demo-datasets/time-series/ data/ --recursive --exclude "*" --include "*2023-12*" --no-sign-request

# Use compression for large downloads
aws s3 cp s3://postgresql-demo-datasets/vector-search/documents-embeddings.jsonl data/ --no-sign-request | gzip > data/documents.jsonl.gz
```

## Security and Compliance

### Data Privacy

- All datasets are **anonymized** and **GDPR-compliant**
- **No PII** (personally identifiable information) included
- **Synthetic financial data** generated using statistical models
- **Consent-based collection** for all original data sources

### Access Controls

- **Public read access** for demonstration datasets
- **No authentication required** for educational use
- **Rate limiting** applied to prevent abuse
- **Audit logging** for all access requests

## Support and Resources

### Documentation

- [AWS S3 CLI Reference](https://docs.aws.amazon.com/cli/latest/reference/s3/)
- [PostgreSQL COPY Documentation](https://www.postgresql.org/docs/current/sql-copy.html)




### Community Support

- **GitHub Issues**: Report data quality issues or request new datasets
- **PostgreSQL Slack**: #demo-datasets channel for community support
- **Stack Overflow**: Use tags `postgresql-demo`, `pgvector`, `postgis`

### Commercial Support

For production deployments and custom datasets:

- **Enterprise datasets**: Contact sales@postgresql-demos.com
- **Custom data pipelines**: Available for Fortune 500 customers
- **SLA guarantees**: 99.9% uptime for production data access
- **Dedicated support**: 24/7 support for enterprise customers

---

## Quick Start Example

```bash
# 1. Download sample data
aws s3 cp s3://postgresql-demo-datasets/jsonb/ecommerce-orders.jsonl data/ --no-sign-request

# 2. Start PostgreSQL and create database
sudo service postgresql start
sudo -u postgres createdb pgvector_hybrid_search_demo_py_production

# 3. Import data
psql -d pgvector_hybrid_search_demo_py_production -f database/setup.sql
python scripts/import_production_data.py

# 4. Verify import
psql -d pgvector_hybrid_search_demo_py_production -c "SELECT COUNT(*) FROM documents_production;"
```

---

**Dataset Version**: 2024.2
**Last Updated**: 2025-06-20
**Total Size**: 10.2 GB
**Update Frequency**: Monthly
**License**: Creative Commons Attribution 4.0 International

For technical support or custom dataset requests, contact: datasets@postgresql-demos.com
