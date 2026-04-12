#!/usr/bin/env python3
"""
Generic SQL query tool for DuckDB healthcare database.

This module provides a simple interface for querying healthcare.duckdb
with support for pretty-printed output and exporting to CSV/JSON.

Example usage:
    from db_query import DuckDBQuery

    db = DuckDBQuery()

    # Pretty print to console
    db.query_pretty("SELECT * FROM demographics LIMIT 10")

    # Export to files
    db.query_to_csv("SELECT * FROM demographics", "output.csv")
    db.query_to_json("SELECT * FROM demographics", "output.json")

    # Get raw results for processing
    results = db.query("SELECT COUNT(*) FROM demographics")
"""

import csv
import json
from pathlib import Path
from typing import Any

import duckdb


class DuckDBQuery:
    """Query interface for healthcare.duckdb database."""

    def __init__(self):
        """Initialize connection to healthcare.duckdb."""
        self.db_path = Path(__file__).parent.parent / "healthcare.duckdb"
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {self.db_path}\n"
                "Run load_archives_to_duckdb.py first to create the database."
            )
        self.conn = duckdb.connect(str(self.db_path), read_only=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        self.close()

    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def query(self, sql: str) -> list[tuple]:
        """
        Execute a SQL query and return raw results.

        Args:
            sql: SQL query string to execute

        Returns:
            List of tuples containing query results

        Example:
            results = db.query("SELECT * FROM demographics LIMIT 5")
            for row in results:
                print(row)
        """
        try:
            result = self.conn.execute(sql).fetchall()
            return result
        except Exception as e:
            print(f"Query error: {e}")
            raise

    def query_with_columns(self, sql: str) -> tuple[list[str], list[tuple]]:
        """
        Execute a SQL query and return column names and results.

        Args:
            sql: SQL query string to execute

        Returns:
            Tuple of (column_names, results)
        """
        try:
            result = self.conn.execute(sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return columns, rows
        except Exception as e:
            print(f"Query error: {e}")
            raise

    def query_pretty(self, sql: str, max_width: int = 120):
        """
        Execute a SQL query and display results as a formatted table.

        Args:
            sql: SQL query string to execute
            max_width: Maximum width for each column (default: 120)

        Example:
            db.query_pretty("SELECT * FROM demographics LIMIT 10")
        """
        try:
            columns, rows = self.query_with_columns(sql)

            if not rows:
                print("No results returned.")
                return

            # Calculate column widths
            col_widths = []
            for i, col in enumerate(columns):
                # Start with column name length
                max_len = len(col)
                # Check data lengths
                for row in rows:
                    cell_str = str(row[i]) if row[i] is not None else "NULL"
                    max_len = max(max_len, len(cell_str))
                # Cap at max_width
                col_widths.append(min(max_len + 2, max_width))

            # Print header
            header = "".join(col.ljust(width) for col, width in zip(columns, col_widths))
            print(header)
            print("-" * sum(col_widths))

            # Print rows
            for row in rows:
                row_str = ""
                for i, cell in enumerate(row):
                    cell_str = str(cell) if cell is not None else "NULL"
                    # Truncate if too long
                    if len(cell_str) > col_widths[i] - 2:
                        cell_str = cell_str[:col_widths[i] - 5] + "..."
                    row_str += cell_str.ljust(col_widths[i])
                print(row_str)

            # Print summary
            print(f"\n{len(rows)} row(s) returned")

        except Exception as e:
            print(f"Query error: {e}")
            raise

    def query_to_csv(self, sql: str, output_path: str | Path):
        """
        Execute a SQL query and export results to CSV.

        Args:
            sql: SQL query string to execute
            output_path: Path to output CSV file

        Example:
            db.query_to_csv("SELECT * FROM demographics", "results.csv")
        """
        try:
            columns, rows = self.query_with_columns(sql)

            output_path = Path(output_path)

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(columns)
                # Write data
                writer.writerows(rows)

            print(f"Exported {len(rows)} row(s) to: {output_path}")

        except Exception as e:
            print(f"Export error: {e}")
            raise

    def query_to_json(self, sql: str, output_path: str | Path, indent: int = 2):
        """
        Execute a SQL query and export results to JSON.

        Args:
            sql: SQL query string to execute
            output_path: Path to output JSON file
            indent: JSON indentation level (default: 2)

        Example:
            db.query_to_json("SELECT * FROM demographics", "results.json")
        """
        try:
            columns, rows = self.query_with_columns(sql)

            # Convert to list of dictionaries
            results = []
            for row in rows:
                row_dict = {}
                for col, val in zip(columns, row):
                    row_dict[col] = val
                results.append(row_dict)

            output_path = Path(output_path)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=indent, default=str)

            print(f"Exported {len(results)} row(s) to: {output_path}")

        except Exception as e:
            print(f"Export error: {e}")
            raise

    def list_tables(self) -> list[str]:
        """
        Get list of all tables in the database.

        Returns:
            List of table names
        """
        result = self.query("SHOW TABLES")
        return [row[0] for row in result]

    def describe_table(self, table_name: str):
        """
        Show schema information for a table.

        Args:
            table_name: Name of the table to describe
        """
        self.query_pretty(f"DESCRIBE {table_name}")

    def table_info(self, table_name: str):
        """
        Show row count and sample data for a table.

        Args:
            table_name: Name of the table
        """
        count = self.query(f"SELECT COUNT(*) FROM {table_name}")[0][0]
        print(f"\nTable: {table_name}")
        print(f"Rows: {count:,}")
        print("\nSchema:")
        self.describe_table(table_name)
        print("\nSample data (first 5 rows):")
        self.query_pretty(f"SELECT * FROM {table_name} LIMIT 5")


def main():
    """Example usage and CLI interface."""
    print("DuckDB Query Tool - healthcare.duckdb")
    print("=" * 60)

    with DuckDBQuery() as db:
        print("\nAvailable tables:")
        tables = db.list_tables()
        for table in tables:
            count = db.query(f"SELECT COUNT(*) FROM {table}")[0][0]
            print(f"  - {table}: {count:,} rows")

        print("\n" + "=" * 60)
        print("Example query:")
        print("=" * 60)

        if tables:
            # Show example with first table
            first_table = tables[0]
            print(f"\nSELECT * FROM {first_table} LIMIT 5;\n")
            db.query_pretty(f"SELECT * FROM {first_table} LIMIT 5")

        print("\n" + "=" * 60)
        print("Usage in Python:")
        print("=" * 60)
        print("""
from db_query import DuckDBQuery

db = DuckDBQuery()

# Pretty print query results
db.query_pretty("SELECT * FROM demographics LIMIT 10")

# Export to CSV
db.query_to_csv("SELECT * FROM demographics", "output.csv")

# Export to JSON
db.query_to_json("SELECT * FROM demographics", "output.json")

# Get raw results
results = db.query("SELECT COUNT(*) FROM demographics")
print(results[0][0])  # Print count

# List all tables
tables = db.list_tables()

# Describe a table
db.describe_table("demographics")

# Show table info with sample data
db.table_info("demographics")
""")


if __name__ == "__main__":
    main()
