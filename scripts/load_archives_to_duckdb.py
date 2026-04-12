#!/usr/bin/env python3
"""
Script to load CSV files from data/ directory into DuckDB.

Requirements:
    pip install duckdb
"""

import duckdb
from pathlib import Path


def load_csv_to_table(conn, csv_path):
    """Load a CSV file into DuckDB as a table."""
    table_name = csv_path.stem.lower().replace('-', '_').replace(' ', '_')

    print(f"\n  Loading: {csv_path.name}")
    print(f"  Table: {table_name}")

    try:
        # Create table from CSV
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{csv_path}', header=true)
        """)

        # Get row count
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  Loaded {count:,} rows")

        # Show first few column names
        cols = conn.execute(f"DESCRIBE {table_name}").fetchall()
        col_names = [col[0] for col in cols[:5]]
        print(f"  Columns: {', '.join(col_names)}{' ...' if len(cols) > 5 else ''}")

    except Exception as e:
        print(f"  ERROR: {e}")


def main():
    """Main function."""
    print("=" * 60)
    print("CSV to DuckDB Loader")
    print("=" * 60)

    # Find all CSV files in data/ directory
    project_root = Path(__file__).parent.parent.resolve()
    data_dir = project_root / 'data'

    csv_files = sorted(data_dir.glob('*.csv'))

    if not csv_files:
        print(f"ERROR: No CSV files found in {data_dir}")
        print("Run the simulate_data.ipynb notebook first to generate data.")
        return

    print(f"\nFound {len(csv_files)} CSV file(s) in {data_dir}:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")

    # Initialize DuckDB
    db_path = project_root / 'healthcare.duckdb'
    print(f"\nConnecting to DuckDB: {db_path}")
    conn = duckdb.connect(str(db_path))
    print("Connected")

    # Load each CSV file
    print("\n" + "-" * 60)
    print("Loading CSV files...")
    print("-" * 60)

    for csv_path in csv_files:
        load_csv_to_table(conn, csv_path)

    # Show summary
    print("\n" + "=" * 60)
    print("Summary - Tables in DuckDB:")
    print("=" * 60)
    tables = conn.execute("SHOW TABLES").fetchall()

    if tables:
        for table in tables:
            table_name = table[0]
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  {table_name}: {count:,} rows")
    else:
        print("  No tables created")

    # Close connection
    conn.close()
    print(f"\nDatabase saved to: {db_path}")
    print("\nQuery the data with:")
    print(f"  duckdb {db_path}")


if __name__ == "__main__":
    main()
