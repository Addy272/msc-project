"""
Helpers for managing the active symbol catalog.

The catalog can come from a built-in demo list or an uploaded NSE contract CSV.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from io import BytesIO

import pandas as pd

from config import Config


DEFAULT_COMPANY_NAMES = {}

SUPPORTED_ENCODINGS = ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1')
PREFERRED_SERIES = 'EQ'


def _normalize_header(value):
    """Normalize CSV header names for resilient matching."""
    cleaned = str(value or '').replace('\ufeff', ' ').strip().upper()
    return ' '.join(cleaned.split())


def normalize_symbol(value):
    """Normalize a symbol without stripping meaningful punctuation."""
    return str(value or '').replace('\ufeff', '').strip().upper()


def _clean_text(value):
    """Return a cleaned string value or an empty string."""
    return str(value or '').replace('\ufeff', '').strip()


def _read_csv_bytes(file_bytes):
    """Read CSV bytes using a few common encodings."""
    if not file_bytes:
        raise ValueError('The uploaded CSV file is empty.')

    last_error = None
    for encoding in SUPPORTED_ENCODINGS:
        try:
            return pd.read_csv(
                BytesIO(file_bytes),
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc

    raise ValueError('The CSV file could not be read. Please upload a valid exchange CSV.') from last_error


def parse_contract_csv_bytes(file_bytes):
    """Parse an uploaded exchange CSV and extract one row per symbol."""
    df = _read_csv_bytes(file_bytes)

    normalized_columns = {_normalize_header(column): column for column in df.columns}
    symbol_column = normalized_columns.get('SYMBOL')
    if not symbol_column:
        raise ValueError("The CSV file must include a 'SYMBOL' column.")

    company_column = next(
        (original for normalized, original in normalized_columns.items() if normalized.startswith('NAME OF')),
        None,
    )
    series_column = normalized_columns.get('SERIES')

    records = []
    record_positions = {}

    for row in df.to_dict('records'):
        symbol = normalize_symbol(row.get(symbol_column))
        if not symbol:
            continue

        series = normalize_symbol(row.get(series_column)) if series_column else ''
        company_name = _clean_text(row.get(company_column)) or symbol
        candidate = {
            'symbol': symbol,
            'company_name': company_name,
            'series': series or None,
            'source': 'upload',
        }

        existing_index = record_positions.get(symbol)
        if existing_index is None:
            record_positions[symbol] = len(records)
            records.append(candidate)
            continue

        existing_series = records[existing_index].get('series')
        if series == PREFERRED_SERIES and existing_series != PREFERRED_SERIES:
            records[existing_index] = candidate

    if not records:
        raise ValueError('No usable symbols were found in the uploaded CSV file.')

    return records


def _load_metadata():
    """Load persisted metadata for the uploaded catalog."""
    if not os.path.exists(Config.CONTRACT_METADATA_PATH):
        return {}

    try:
        with open(Config.CONTRACT_METADATA_PATH, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_metadata(payload):
    """Persist uploaded catalog metadata."""
    os.makedirs(Config.DATA_CONTRACTS_PATH, exist_ok=True)
    with open(Config.CONTRACT_METADATA_PATH, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def save_uploaded_contract(file_bytes, original_filename):
    """Validate and persist an uploaded contract CSV."""
    records = parse_contract_csv_bytes(file_bytes)

    os.makedirs(Config.DATA_CONTRACTS_PATH, exist_ok=True)
    with open(Config.CONTRACT_SYMBOLS_PATH, 'wb') as handle:
        handle.write(file_bytes)

    metadata = {
        'original_filename': original_filename or 'uploaded_contracts.csv',
        'uploaded_at': datetime.now().isoformat(timespec='seconds'),
        'symbol_count': len(records),
    }
    _write_metadata(metadata)

    return {
        'metadata': metadata,
        'records': records,
    }


def load_uploaded_contract_rows():
    """Load symbol rows from the persisted uploaded CSV, if available."""
    if not os.path.exists(Config.CONTRACT_SYMBOLS_PATH):
        return []

    try:
        with open(Config.CONTRACT_SYMBOLS_PATH, 'rb') as handle:
            return parse_contract_csv_bytes(handle.read())
    except (OSError, ValueError):
        return []


def clear_uploaded_contract():
    """Remove persisted uploaded contract files, if they exist."""
    cleared_files = []
    for path in (Config.CONTRACT_SYMBOLS_PATH, Config.CONTRACT_METADATA_PATH):
        if os.path.exists(path):
            os.remove(path)
            cleared_files.append(path)
    return cleared_files


def get_default_symbol_records():
    """Return the built-in symbol catalog."""
    return []


def get_symbol_catalog():
    """Return the active catalog and related metadata."""
    uploaded_records = load_uploaded_contract_rows()
    metadata = _load_metadata()

    if uploaded_records:
        uploaded_at = metadata.get('uploaded_at')
        if not uploaded_at and os.path.exists(Config.CONTRACT_SYMBOLS_PATH):
            uploaded_at = datetime.fromtimestamp(
                os.path.getmtime(Config.CONTRACT_SYMBOLS_PATH)
            ).isoformat(timespec='seconds')

        return {
            'uploaded': True,
            'source': 'upload',
            'source_label': 'Uploaded NSE contract file',
            'records': uploaded_records,
            'symbol_count': len(uploaded_records),
            'original_filename': metadata.get('original_filename') or os.path.basename(Config.CONTRACT_SYMBOLS_PATH),
            'uploaded_at': uploaded_at,
            'storage_path': Config.CONTRACT_SYMBOLS_PATH,
        }

    default_records = get_default_symbol_records()
    return {
        'uploaded': False,
        'source': 'empty',
        'source_label': 'No uploaded contract loaded',
        'records': default_records,
        'symbol_count': len(default_records),
        'original_filename': None,
        'uploaded_at': None,
        'storage_path': Config.CONTRACT_SYMBOLS_PATH,
    }


def get_company_name_for_symbol(symbol, catalog_records=None):
    """Resolve a display name for a symbol from uploaded data first."""
    normalized_symbol = normalize_symbol(symbol)
    if not normalized_symbol:
        return ''

    records = catalog_records if catalog_records is not None else load_uploaded_contract_rows()
    for record in records:
        if record['symbol'] == normalized_symbol and record.get('company_name'):
            return record['company_name']

    return normalized_symbol


def get_market_data_symbol(symbol, catalog_records=None):
    """Map an app symbol to the Yahoo Finance ticker used for price downloads."""
    normalized_symbol = normalize_symbol(symbol)
    if not normalized_symbol:
        return symbol

    if any(marker in normalized_symbol for marker in ('.', '^', '=')):
        return normalized_symbol

    records = catalog_records if catalog_records is not None else load_uploaded_contract_rows()
    uploaded_symbols = {record['symbol'] for record in records}
    if normalized_symbol in uploaded_symbols:
        return f'{normalized_symbol}.NS'

    return normalized_symbol
