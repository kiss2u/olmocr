"""
Split table tests with multiple relationships into individual tests.

For example, if you have a test case with cell=X, up=Y, down=Z, this script
will split it into two cases: one with cell=X, up=Y and another with cell=X, down=Z.

Usage:
    python split_table_tests.py input.jsonl output.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from olmocr.bench.tests import BasePDFTest, TableTest, TestType, load_tests


# The relationship fields that can be split
RELATIONSHIP_FIELDS = ["up", "down", "left", "right", "top_heading", "left_heading"]


def base_test_to_dict(test: BasePDFTest) -> dict:
    """
    Convert any BasePDFTest to a dict by extracting all non-None attributes.
    """
    from dataclasses import fields, is_dataclass

    result = {}
    if is_dataclass(test):
        for field in fields(test):
            value = getattr(test, field.name)
            # Skip None values and empty strings for optional fields
            if value is not None and value != "":
                result[field.name] = value
    return result


def test_to_dict_minimal(test: TableTest) -> dict:
    """
    Convert a TableTest to a dict, removing empty relationship fields.
    """
    result = {
        "pdf": test.pdf,
        "page": test.page,
        "id": test.id,
        "type": test.type,
        "max_diffs": test.max_diffs,
        "cell": test.cell,
    }

    # Only include non-empty relationship fields
    for field in RELATIONSHIP_FIELDS:
        value = getattr(test, field)
        if value and value.strip():
            result[field] = value

    # Include other optional fields if set
    if test.checked is not None:
        result["checked"] = test.checked
    if test.url:
        result["url"] = test.url
    if test.ignore_markdown_tables:
        result["ignore_markdown_tables"] = test.ignore_markdown_tables

    return result


def split_table_test(test: TableTest) -> list[dict]:
    """
    Split a single TableTest with multiple relationships into separate test dicts.

    Each resulting test will have only one relationship field set (plus the cell).

    Args:
        test: The TableTest to split

    Returns:
        A list of dicts, one for each non-empty relationship
    """
    # Find which relationship fields are set (non-empty)
    active_relationships = []
    for field in RELATIONSHIP_FIELDS:
        value = getattr(test, field)
        if value and value.strip():
            active_relationships.append((field, value))

    # If there's 0 or 1 relationship, no splitting needed - return as minimal dict
    if len(active_relationships) <= 1:
        return [test_to_dict_minimal(test)]

    # Create a new test dict for each relationship
    split_tests = []
    for field, value in active_relationships:
        result = {
            "pdf": test.pdf,
            "page": test.page,
            "id": f"{test.id}_{field}",
            "type": test.type,
            "max_diffs": test.max_diffs,
            "cell": test.cell,
            field: value,  # Only include the single relationship field
        }

        # Include other optional fields if set
        if test.checked is not None:
            result["checked"] = test.checked
        if test.url:
            result["url"] = test.url
        if test.ignore_markdown_tables:
            result["ignore_markdown_tables"] = test.ignore_markdown_tables

        split_tests.append(result)

    return split_tests


def main():
    parser = argparse.ArgumentParser(
        description="Split table tests with multiple relationships into individual tests"
    )
    parser.add_argument("input_file", help="Input JSONL file containing tests")
    parser.add_argument("output_file", help="Output JSONL file for split tests")
    parser.add_argument(
        "--keep-non-table",
        action="store_true",
        help="Keep non-table tests in the output (default: only output table tests)"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)

    # Load all tests
    print(f"Loading tests from {input_path}...")
    tests = load_tests(str(input_path))
    print(f"Loaded {len(tests)} tests")

    # Process tests
    output_tests = []
    table_tests_count = 0
    split_count = 0

    for test in tests:
        if test.type == TestType.TABLE.value:
            table_tests_count += 1
            split_dicts = split_table_test(test)
            if len(split_dicts) > 1:
                split_count += 1
                print(f"Split test '{test.id}' into {len(split_dicts)} tests")
            output_tests.extend(split_dicts)
        elif args.keep_non_table:
            # For non-table tests, just convert to dict (keep all fields)
            output_tests.append(base_test_to_dict(test))

    # Save output
    print(f"\nSaving {len(output_tests)} tests to {output_path}...")
    with open(output_path, "w") as f:
        for test_dict in output_tests:
            f.write(json.dumps(test_dict) + "\n")

    print("\nSummary:")
    print(f"  Input tests: {len(tests)}")
    print(f"  Table tests found: {table_tests_count}")
    print(f"  Table tests split: {split_count}")
    print(f"  Output tests: {len(output_tests)}")


if __name__ == "__main__":
    main()
