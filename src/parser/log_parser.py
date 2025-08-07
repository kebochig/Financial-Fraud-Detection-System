"""
Robust log parser for handling multiple transaction log formats.
"""
import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from ..utils.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ParsedTransaction:
    """Structured representation of a parsed transaction."""
    timestamp: Optional[datetime] = None
    user_id: Optional[str] = None
    transaction_type: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    amount_gbp: Optional[float] = None  # Amount converted to GBP
    original_currency: Optional[str] = None  # Original currency before conversion
    location: Optional[str] = None
    device: Optional[str] = None
    raw_log: str = ""
    is_parsed: bool = False
    parse_errors: List[str] = None
    
    def __post_init__(self):
        if self.parse_errors is None:
            self.parse_errors = []

class TransactionLogParser:
    """Robust parser for various transaction log formats."""
    
    def __init__(self):
        """Initialize parser with configuration."""
        self.date_formats = config.get('parsing.date_formats', [
            '%Y-%m-%d %H:%M:%S',
            '%d/%m/%Y %H:%M:%S'
        ])
        
        # Exchange rates to convert to GBP (£)
        self.exchange_rates = {
            '$': 0.75,  # 1 USD = 0.75 GBP
            '€': 0.87,  # 1 EUR = 0.87 GBP
            '£': 1.0,   # 1 GBP = 1.0 GBP
            'unknown': 1.0,  # Default to GBP if unknown
            None: 1.0,  # Default to GBP if None
            '': 1.0     # Default to GBP if empty
        }
        
        # Regex patterns for extracting information
        self.patterns = {
            'amount': re.compile(r'([£$€]?)(\d+\.?\d*)', re.IGNORECASE),
            'user': re.compile(r'user\d+', re.IGNORECASE),
            'location': re.compile(r'(London|Glasgow|Birmingham|Liverpool|Cardiff|Leeds|Manchester|None)', re.IGNORECASE),
            'device': re.compile(r'(iPhone 13|Samsung Galaxy S10|Pixel 6|Nokia 3310|Xiaomi Mi 11|Huawei P30|None)', re.IGNORECASE),
            'transaction_type': re.compile(r'(withdrawal|deposit|cashout|purchase|transfer|top-up|debit|refund)', re.IGNORECASE),
            'timestamp1': re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'),
            'timestamp2': re.compile(r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})'),
            'timestamp3': re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})::[^:]+::[^:]+::[^:]+::[^:]+::[^:]+')
        }
        
        # Known log formats for structured parsing
        self.format_parsers = [
            self._parse_format_1,  # Standard format: timestamp | user | txn | amount | location | device
            self._parse_format_2,  # Colon format: timestamp::user::type::amount::location::device
            self._parse_format_3,  # Pipe format with 'usr:' prefix
            self._parse_format_4,  # Arrow format with '>>' 
            self._parse_format_5,  # Triple colon format with :::
            self._parse_format_6,  # Dash format with user=, action=, ATM:
            self._parse_format_7,  # Simple space-separated format
        ]
    
    def parse_log_entry(self, log_entry: str) -> ParsedTransaction:
        """Parse a single log entry using multiple strategies."""
        log_entry = log_entry.strip()
        
        # Handle empty or malformed entries
        if not log_entry or log_entry == '""' or log_entry == 'MALFORMED_LOG':
            return ParsedTransaction(raw_log=log_entry, is_parsed=False, 
                                   parse_errors=['Empty or malformed log'])
        
        transaction = ParsedTransaction(raw_log=log_entry)
        
        # Try structured parsers first
        for parser in self.format_parsers:
            try:
                if parser(log_entry, transaction):
                    transaction.is_parsed = True
                    return transaction
            except Exception as e:
                transaction.parse_errors.append(f"Parser error: {str(e)}")
        
        # Fallback to regex-based extraction
        self._extract_with_regex(log_entry, transaction)
        
        # Validate and clean extracted data
        self._validate_and_clean(transaction)
        
        return transaction
    
    def _parse_format_1(self, log_entry: str, transaction: ParsedTransaction) -> bool:
        """Parse format: 'timestamp | user: userXXX | txn: type of amount from location | device: device'"""
        if '|' not in log_entry or 'user:' not in log_entry:
            return False
        
        try:
            parts = [p.strip() for p in log_entry.split('|')]
            if len(parts) < 3:
                return False
            
            # Extract timestamp
            timestamp_str = parts[0].strip()
            transaction.timestamp = self._parse_timestamp(timestamp_str)
            
            # Extract user
            user_part = parts[1].strip()
            if 'user:' in user_part:
                transaction.user_id = user_part.split('user:')[1].strip()
            
            # Extract transaction details
            txn_part = parts[2].strip()
            if 'txn:' in txn_part:
                txn_details = txn_part.split('txn:')[1].strip()
                # Parse transaction type, amount, location
                self._parse_transaction_details(txn_details, transaction)
            
            # Extract device
            if len(parts) > 3 and 'device:' in parts[3]:
                transaction.device = parts[3].split('device:')[1].strip()
            
            return True
        except Exception:
            return False
    
    def _parse_format_2(self, log_entry: str, transaction: ParsedTransaction) -> bool:
        """Parse format: 'timestamp::user::type::amount::location::device'"""
        if '::' not in log_entry:
            return False
        
        try:
            parts = log_entry.split('::')
            if len(parts) < 4:
                return False
            
            transaction.timestamp = self._parse_timestamp(parts[0])
            transaction.user_id = parts[1]
            transaction.transaction_type = parts[2]
            
            # Parse amount (might include currency)
            amount_str = parts[3]
            amount_match = self.patterns['amount'].search(amount_str)
            if amount_match:
                transaction.currency = amount_match.group(1) or None
                transaction.amount = float(amount_match.group(2))
            
            if len(parts) > 4:
                transaction.location = parts[4] if parts[4] != 'None' else None
            if len(parts) > 5:
                transaction.device = parts[5] if parts[5] != 'None' else None
            
            return True
        except Exception:
            return False
    
    def _parse_format_3(self, log_entry: str, transaction: ParsedTransaction) -> bool:
        """Parse format: 'usr:userXXX|type|amount|location|timestamp|device'"""
        if not log_entry.startswith('usr:'):
            return False
        
        try:
            parts = log_entry.split('|')
            if len(parts) < 4:
                return False
            
            # Extract user
            transaction.user_id = parts[0].split('usr:')[1]
            transaction.transaction_type = parts[1]
            
            # Parse amount with currency
            amount_str = parts[2]
            amount_match = self.patterns['amount'].search(amount_str)
            if amount_match:
                transaction.currency = amount_match.group(1) or None
                transaction.amount = float(amount_match.group(2))
            
            transaction.location = parts[3] if parts[3] != 'None' else None
            
            if len(parts) > 4:
                transaction.timestamp = self._parse_timestamp(parts[4])
            if len(parts) > 5:
                transaction.device = parts[5] if parts[5] != 'None' else None
            
            return True
        except Exception:
            return False
    
    def _parse_format_4(self, log_entry: str, transaction: ParsedTransaction) -> bool:
        """Parse format: 'timestamp >> [userXXX] did action - amt=amount - location // dev:device'"""
        if '>>' not in log_entry or '[' not in log_entry:
            return False
        
        try:
            # Split by main delimiters
            timestamp_part, rest = log_entry.split('>>', 1)
            transaction.timestamp = self._parse_timestamp(timestamp_part.strip())
            
            # Extract user
            user_match = re.search(r'\[([^\]]+)\]', rest)
            if user_match:
                transaction.user_id = user_match.group(1)
            
            # Extract transaction type
            type_match = re.search(r'did (\w+)', rest)
            if type_match:
                transaction.transaction_type = type_match.group(1)
            
            # Extract amount
            amt_match = re.search(r'amt=([£$€]?\d+\.?\d*)', rest)
            if amt_match:
                amount_str = amt_match.group(1)
                amount_match = self.patterns['amount'].search(amount_str)
                if amount_match:
                    transaction.currency = amount_match.group(1) or None
                    transaction.amount = float(amount_match.group(2))
            
            # Extract location, ensuring the correct delimiter
            location_match = re.search(r'- ([^/]+?)( //| dev:)', rest)
            if location_match:
                location = location_match.group(1).strip()
                transaction.location = location if location != 'None' else None
            
            # Extract device
            device_match = re.search(r'dev:([^$]+)$', rest)
            if device_match:
                device = device_match.group(1).strip()
                transaction.device = device if device != 'None' else None
            
            return True
        except Exception:
            return False
    
    def _parse_format_5(self, log_entry: str, transaction: ParsedTransaction) -> bool:
        """Parse format: 'dd/mm/yyyy hh:mm:ss ::: userXXX *** ACTION ::: amt:amount @ location <device>'"""
        if ':::' not in log_entry or '***' not in log_entry:
            return False
        
        try:
            parts = log_entry.split(':::')
            if len(parts) < 3:
                return False
            
            # Parse timestamp
            transaction.timestamp = self._parse_timestamp(parts[0].strip())
            
            # Parse user and action
            user_action = parts[1].strip()
            user_match = re.search(r'(\w+) \*\*\* (\w+)', user_action)
            if user_match:
                transaction.user_id = user_match.group(1)
                transaction.transaction_type = user_match.group(2).lower()
            
            # Parse amount and location
            details = parts[2].strip()
            amt_match = re.search(r'amt:([£$€]?\d+\.?\d*)', details)
            if amt_match:
                amount_str = amt_match.group(1)
                amount_match = self.patterns['amount'].search(amount_str)
                if amount_match:
                    transaction.currency = amount_match.group(1) or None
                    transaction.amount = float(amount_match.group(2))
            
            location_match = re.search(r'@ ([^<]+)', details)
            if location_match:
                location = location_match.group(1).strip()
                transaction.location = location if location != 'None' else None
            
            device_match = re.search(r'<([^>]+)>', details)
            if device_match:
                device = device_match.group(1).strip()
                transaction.device = device if device != 'None' else None
            
            return True
        except Exception:
            return False
    
    def _parse_format_6(self, log_entry: str, transaction: ParsedTransaction) -> bool:
        """Parse format: 'timestamp - user=userXXX - action=type amount - ATM: location - device=device'"""
        if 'user=' not in log_entry or 'action=' not in log_entry:
            return False
        
        try:
            parts = log_entry.split(' - ')
            if len(parts) < 3:
                return False
            
            # Parse timestamp
            transaction.timestamp = self._parse_timestamp(parts[0].strip())
            
            # Parse user
            user_part = next((p for p in parts if 'user=' in p), None)
            if user_part:
                transaction.user_id = user_part.split('user=')[1]
            
            # Parse action and amount
            action_part = next((p for p in parts if 'action=' in p), None)
            if action_part:
                action_details = action_part.split('action=')[1]
                # Extract transaction type and amount
                words = action_details.split()
                if words:
                    transaction.transaction_type = words[0]
                    # Look for amount in the remaining words
                    for word in words[1:]:
                        amount_match = self.patterns['amount'].search(word)
                        if amount_match:
                            transaction.currency = amount_match.group(1) or None
                            transaction.amount = float(amount_match.group(2))
                            break
            
            # Parse location
            atm_part = next((p for p in parts if 'ATM:' in p), None)
            if atm_part:
                location = atm_part.split('ATM:')[1].strip()
                transaction.location = location if location != 'None' else None
            
            # Parse device
            device_part = next((p for p in parts if 'device=' in p), None)
            if device_part:
                device = device_part.split('device=')[1]
                transaction.device = device if device != 'None' else None
            
            return True
        except Exception:
            return False
    
    def _parse_format_7(self, log_entry: str, transaction: ParsedTransaction) -> bool:
        """Parse format: 'userXXX timestamp type amount location device'"""
        parts = log_entry.split()
        if len(parts) < 4:
            return False
        
        try:
            # Check if first part is user
            if not parts[0].startswith('user'):
                return False
            
            transaction.user_id = parts[0]
            
            # Try to parse timestamp (could be in parts[1] and parts[2])
            timestamp_str = f"{parts[1]} {parts[2]}"
            transaction.timestamp = self._parse_timestamp(timestamp_str)
            
            if len(parts) > 3:
                transaction.transaction_type = parts[3]
            if len(parts) > 4:
                try:
                    transaction.amount = float(parts[4])
                except ValueError:
                    pass
            if len(parts) > 5:
                transaction.location = parts[5] if parts[5] != 'None' else None
            if len(parts) > 6:
                transaction.device = parts[6] if parts[6] != 'None' else None
            
            return True
        except Exception:
            return False
    
    def _parse_transaction_details(self, details: str, transaction: ParsedTransaction):
        """Parse transaction details string to extract type, amount, and location."""
        # Extract transaction type
        type_match = self.patterns['transaction_type'].search(details)
        if type_match:
            transaction.transaction_type = type_match.group(1).lower()
        
        # Extract amount and currency
        amount_match = self.patterns['amount'].search(details)
        if amount_match:
            transaction.currency = amount_match.group(1) or None
            transaction.amount = float(amount_match.group(2))
        
        # Extract location
        location_match = self.patterns['location'].search(details)
        if location_match:
            location = location_match.group(1)
            transaction.location = location if location != 'None' else None
    
    def _extract_with_regex(self, log_entry: str, transaction: ParsedTransaction):
        """Extract information using regex patterns as fallback."""
        # Extract timestamp
        for pattern_name in ['timestamp1', 'timestamp2', 'timestamp3']:
            timestamp_match = self.patterns[pattern_name].search(log_entry)
            if timestamp_match:
                transaction.timestamp = self._parse_timestamp(timestamp_match.group(1))
                break
        
        # Extract user
        user_match = self.patterns['user'].search(log_entry)
        if user_match:
            transaction.user_id = user_match.group(0)
        
        # Extract transaction type
        type_match = self.patterns['transaction_type'].search(log_entry)
        if type_match:
            transaction.transaction_type = type_match.group(1).lower()
        
        # Extract amount and currency
        amount_match = self.patterns['amount'].search(log_entry)
        if amount_match:
            transaction.currency = amount_match.group(1) or None
            try:
                transaction.amount = float(amount_match.group(2))
            except ValueError:
                transaction.parse_errors.append(f"Invalid amount: {amount_match.group(2)}")
        
        # Extract location
        location_match = self.patterns['location'].search(log_entry)
        if location_match:
            location = location_match.group(1)
            transaction.location = location if location != 'None' else None
        
        # Extract device
        device_match = self.patterns['device'].search(log_entry)
        if device_match:
            device = device_match.group(1)
            transaction.device = device if device != 'None' else None
        
        # Mark as parsed if we extracted some information
        if any([transaction.timestamp, transaction.user_id, transaction.transaction_type, 
                transaction.amount, transaction.location, transaction.device]):
            transaction.is_parsed = True
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string using multiple formats."""
        timestamp_str = timestamp_str.strip()
        
        for date_format in self.date_formats:
            try:
                return datetime.strptime(timestamp_str, date_format)
            except ValueError:
                continue
        
        # Try additional common formats
        additional_formats = [
            '%Y-%m-%d %H:%M:%S.%f',
            '%d/%m/%Y %H:%M:%S.%f',
            '%Y/%m/%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S'
        ]
        
        for date_format in additional_formats:
            try:
                return datetime.strptime(timestamp_str, date_format)
            except ValueError:
                continue
        
        return None
    
    def _validate_and_clean(self, transaction: ParsedTransaction):
        """Validate and clean extracted transaction data."""
        # Validate amount
        if transaction.amount is not None:
            if transaction.amount < 0:
                transaction.parse_errors.append("Negative amount")
                transaction.amount = abs(transaction.amount)  # Convert to positive
            elif transaction.amount > 1000000:  # Arbitrarily large amount
                transaction.parse_errors.append("Unusually large amount")
        
        # Clean and validate user_id
        if transaction.user_id:
            transaction.user_id = transaction.user_id.strip()
            if not transaction.user_id.startswith('user'):
                transaction.user_id = f"user{transaction.user_id}"
        
        # Standardize transaction type
        if transaction.transaction_type:
            transaction.transaction_type = transaction.transaction_type.lower().strip()
        
        # Validate location
        if transaction.location:
            transaction.location = transaction.location.strip()
        
        # Validate device
        if transaction.device:
            transaction.device = transaction.device.strip()
        
        # Set currency if amount exists but currency doesn't
        if transaction.amount is not None and not transaction.currency:
            transaction.currency = 'unknown'
        
        # Convert currency to GBP and standardize amount
        if transaction.amount is not None and transaction.currency:
            transaction.amount_gbp = self._convert_to_gbp(transaction.amount, transaction.currency)
            transaction.original_currency = transaction.currency
            transaction.currency = 'GBP'  # Standardize to GBP
    
    def _convert_to_gbp(self, amount: float, currency: str) -> float:
        """Convert amount from given currency to GBP using exchange rates."""
        if amount is None:
            return None
        
        # Get exchange rate for the currency
        exchange_rate = self.exchange_rates.get(currency, 1.0)
        
        # Convert to GBP
        amount_gbp = amount * exchange_rate
        
        # Round to 2 decimal places for currency precision
        return round(amount_gbp, 2)
    
    def parse_dataset(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Parse entire dataset and return DataFrame with statistics."""
        logger.info(f"Starting to parse dataset: {file_path}")
        
        # Read the CSV file
        # df_raw = pd.read_csv(file_path, delimiter='|', header=None, names=['line_num', 'raw_log'])
        df_raw = pd.read_csv(file_path)
        print(len(df_raw))
        transactions = []
        parsing_stats = {
            'total_logs': len(df_raw),
            'parsed_successfully': 0,
            'parsing_failed': 0,
            'empty_logs': 0,
            'malformed_logs': 0,
            'parsing_errors': []
        }
        
        logger.info(f"Processing {len(df_raw)} log entries...")
        
        for idx, row in df_raw.iterrows():
            if pd.isna(row['raw_log']):
                parsing_stats['empty_logs'] += 1
                continue
                
            raw_log = str(row['raw_log']).strip()
            
            if raw_log == 'MALFORMED_LOG':
                parsing_stats['malformed_logs'] += 1
                continue
            
            if raw_log == '""' or raw_log == '':
                parsing_stats['empty_logs'] += 1
                continue
            
            # Parse the log entry
            transaction = self.parse_log_entry(raw_log)
            
            if transaction.is_parsed:
                parsing_stats['parsed_successfully'] += 1
            else:
                parsing_stats['parsing_failed'] += 1
                parsing_stats['parsing_errors'].extend(transaction.parse_errors)
            
            # Convert to dictionary for DataFrame
            transaction_dict = {
                'raw_log': transaction.raw_log,
                'timestamp': transaction.timestamp,
                'user_id': transaction.user_id,
                'transaction_type': transaction.transaction_type,
                'amount': transaction.amount,
                'currency': transaction.currency,
                'location': transaction.location,
                'device': transaction.device,
                'is_parsed': transaction.is_parsed,
                'parse_errors': '; '.join(transaction.parse_errors) if transaction.parse_errors else None
            }
            
            transactions.append(transaction_dict)
        
        # Create DataFrame
        df_parsed = pd.DataFrame(transactions)

        df_parsed['location'] = df_parsed['location'].apply(lambda x: x.split('-')[-1].strip() if isinstance(x, str) and '-' in x else x)

        # Currency mapping based on location for None/missing currencies
        location_currency_map = {
            'Leeds': '£',
            'London': '£', 
            'Glasgow': '£',
            'Manchester': '£',
            'Birmingham': '£',
            'Cardiff': '£',
            'Liverpool': '£'
        }
        
        # Fill missing currencies based on location mapping
        mask = (df_parsed['currency'].isnull()) | (df_parsed['currency'] == 'None') | (df_parsed['currency'] == '')
        for idx in df_parsed[mask].index:
            location = df_parsed.loc[idx, 'location']
            if location in location_currency_map:
                df_parsed.loc[idx, 'currency'] = location_currency_map[location]
            elif location is not None:  # Unknown location, default to £
                df_parsed.loc[idx, 'currency'] = '£'
        
        # Convert amount to GBP if currency is known
        df_parsed['amount_raw'] = df_parsed['amount']  # Keep original amount for reference
        df_parsed['amount'] = df_parsed.apply(lambda row: self._convert_to_gbp(row['amount'], row['currency']) if pd.notna(row['amount']) else None, axis=1)

        
        # Add parsing success statistics
        parsing_stats['parsing_success'] = {
            'success': parsing_stats['parsed_successfully'],
            'failed': parsing_stats['parsing_failed'] + parsing_stats['empty_logs'] + parsing_stats['malformed_logs']
        }
        
        logger.info(f"Parsing complete. Successfully parsed: {parsing_stats['parsed_successfully']}, "
                   f"Failed: {parsing_stats['parsing_failed']}, "
                   f"Empty/Malformed: {parsing_stats['empty_logs'] + parsing_stats['malformed_logs']}")
        
        return df_parsed, parsing_stats
    
    def parse_dataset_from_dataframe(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Parse dataset from a DataFrame and return DataFrame with statistics."""
        logger.info(f"Starting to parse dataset from DataFrame with {len(df_raw)} rows")
        
        transactions = []
        parsing_stats = {
            'total_logs': len(df_raw),
            'parsed_successfully': 0,
            'parsing_failed': 0,
            'empty_logs': 0,
            'malformed_logs': 0,
            'parsing_errors': []
        }
        
        logger.info(f"Processing {len(df_raw)} log entries...")
        
        for idx, row in df_raw.iterrows():
            if pd.isna(row['raw_log']):
                parsing_stats['empty_logs'] += 1
                continue
                
            raw_log = str(row['raw_log']).strip()
            
            if raw_log == 'MALFORMED_LOG':
                parsing_stats['malformed_logs'] += 1
                continue
            
            if raw_log == '""' or raw_log == '':
                parsing_stats['empty_logs'] += 1
                continue
            
            # Parse the log entry
            transaction = self.parse_log_entry(raw_log)
            
            if transaction.is_parsed:
                parsing_stats['parsed_successfully'] += 1
            else:
                parsing_stats['parsing_failed'] += 1
                parsing_stats['parsing_errors'].extend(transaction.parse_errors)
            
            # Convert to dictionary for DataFrame
            transaction_dict = {
                'raw_log': transaction.raw_log,
                'timestamp': transaction.timestamp,
                'user_id': transaction.user_id,
                'transaction_type': transaction.transaction_type,
                'amount': transaction.amount,
                'currency': transaction.currency,
                'location': transaction.location,
                'device': transaction.device,
                'is_parsed': transaction.is_parsed,
                'parse_errors': '; '.join(transaction.parse_errors) if transaction.parse_errors else None
            }
            
            transactions.append(transaction_dict)
        
        # Create DataFrame
        df_parsed = pd.DataFrame(transactions)

        df_parsed['location'] = df_parsed['location'].apply(lambda x: x.split('-')[-1].strip() if isinstance(x, str) and '-' in x else x)

        # Currency mapping based on location for None/missing currencies
        location_currency_map = {
            'Leeds': '£',
            'London': '£', 
            'Glasgow': '£',
            'Manchester': '£',
            'Birmingham': '£',
            'Cardiff': '£',
            'Liverpool': '£'
        }
        
        # Fill missing currencies based on location mapping
        mask = (df_parsed['currency'].isnull()) | (df_parsed['currency'] == 'None') | (df_parsed['currency'] == '')
        for idx in df_parsed[mask].index:
            location = df_parsed.loc[idx, 'location']
            if location in location_currency_map:
                df_parsed.loc[idx, 'currency'] = location_currency_map[location]
            elif location is not None:  # Unknown location, default to £
                df_parsed.loc[idx, 'currency'] = '£'
        
        # Convert amount to GBP if currency is known
        df_parsed['amount_raw'] = df_parsed['amount']  # Keep original amount for reference
        df_parsed['amount'] = df_parsed.apply(lambda row: self._convert_to_gbp(row['amount'], row['currency']) if pd.notna(row['amount']) else None, axis=1)

        
        # Add parsing success statistics
        parsing_stats['parsing_success'] = {
            'success': parsing_stats['parsed_successfully'],
            'failed': parsing_stats['parsing_failed'] + parsing_stats['empty_logs'] + parsing_stats['malformed_logs']
        }
        
        logger.info(f"Parsing complete. Successfully parsed: {parsing_stats['parsed_successfully']}, "
                   f"Failed: {parsing_stats['parsing_failed']}, "
                   f"Empty/Malformed: {parsing_stats['empty_logs'] + parsing_stats['malformed_logs']}")
        
        return df_parsed, parsing_stats
    
    def get_parsing_statistics(self, parsing_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive parsing statistics and summary."""
        total_logs = parsing_stats['total_logs']
        successful = parsing_stats['parsed_successfully']
        failed = parsing_stats['parsing_failed']
        empty = parsing_stats['empty_logs']
        malformed = parsing_stats['malformed_logs']
        
        # Calculate rates
        success_rate = (successful / total_logs * 100) if total_logs > 0 else 0
        failure_rate = (failed / total_logs * 100) if total_logs > 0 else 0
        empty_rate = (empty / total_logs * 100) if total_logs > 0 else 0
        malformed_rate = (malformed / total_logs * 100) if total_logs > 0 else 0
        
        # Comprehensive statistics
        statistics = {
            'parsing_summary': {
                'total_logs_processed': total_logs,
                'successfully_parsed': successful,
                'parsing_failures': failed,
                'empty_logs': empty,
                'malformed_logs': malformed,
                'success_rate_percent': round(success_rate, 2),
                'failure_rate_percent': round(failure_rate, 2),
                'empty_rate_percent': round(empty_rate, 2),
                'malformed_rate_percent': round(malformed_rate, 2)
            },
            'quality_metrics': {
                'parsing_efficiency': 'High' if success_rate >= 80 else 'Medium' if success_rate >= 60 else 'Low',
                'data_quality': 'Good' if (empty_rate + malformed_rate) <= 20 else 'Fair' if (empty_rate + malformed_rate) <= 40 else 'Poor',
                'usable_data_ratio': round((successful / total_logs * 100), 2) if total_logs > 0 else 0
            },
            'error_analysis': {
                'total_errors': failed + empty + malformed,
                'error_categories': {
                    'parsing_errors': failed,
                    'empty_entries': empty,
                    'malformed_entries': malformed
                },
                'common_errors': parsing_stats.get('parsing_errors', [])[:10]  # Top 10 errors
            },
            'format_analysis': self._analyze_log_formats(parsing_stats),
            'recommendations': self._generate_recommendations(success_rate, failure_rate, empty_rate, malformed_rate)
        }
        
        return statistics
    
    def _analyze_log_formats(self, parsing_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze which log formats were encountered and their success rates."""
        # This would be enhanced if we tracked format-specific statistics
        # For now, provide general format analysis
        format_analysis = {
            'detected_formats': {
                'colon_separated': 'Format: timestamp::user::type::amount::location::device',
                'pipe_with_usr': 'Format: usr:user|type|amount|location|timestamp|device',
                'arrow_format': 'Format: timestamp >> [user] did action - amt=amount - location // dev:device',
                'triple_colon': 'Format: dd/mm/yyyy hh:mm:ss ::: user *** ACTION ::: amt:amount @ location <device>',
                'dash_format': 'Format: timestamp - user=user - action=type amount - ATM: location - device=device',
                'space_separated': 'Format: user timestamp type amount location device',
                'pipe_standard': 'Format: timestamp | user: user | txn: type of amount from location | device: device'
            },
            'format_complexity': 'Multiple formats detected - parser handles 7+ different log structures',
            'parsing_strategy': 'Sequential format matching with regex fallback'
        }
        
        return format_analysis
    
    def _generate_recommendations(self, success_rate: float, failure_rate: float, 
                                empty_rate: float, malformed_rate: float) -> List[str]:
        """Generate recommendations based on parsing statistics."""
        recommendations = []
        
        if success_rate < 70:
            recommendations.append("Consider adding more format parsers to improve success rate")
            recommendations.append("Review failed logs to identify new patterns")
        
        if failure_rate > 20:
            recommendations.append("High failure rate detected - investigate parsing logic")
            recommendations.append("Consider implementing additional regex patterns")
        
        if empty_rate > 15:
            recommendations.append("High number of empty logs - check data source quality")
            recommendations.append("Consider filtering empty entries at source")
        
        if malformed_rate > 15:
            recommendations.append("High malformed log rate - investigate data generation process")
            recommendations.append("Consider implementing data validation at source")
        
        if success_rate >= 85:
            recommendations.append("Excellent parsing performance - system is production ready")
        
        if len(recommendations) == 0:
            recommendations.append("Parsing performance is within acceptable ranges")
            recommendations.append("Monitor trends over time for any degradation")
        
        return recommendations
    
    def print_parsing_statistics(self, parsing_stats: Dict[str, Any]) -> None:
        """Print formatted parsing statistics to console."""
        stats = self.get_parsing_statistics(parsing_stats)
        
        print("\n" + "="*60)
        print("📊 TRANSACTION LOG PARSING STATISTICS")
        print("="*60)
        
        # Summary
        summary = stats['parsing_summary']
        print(f"\n📋 PARSING SUMMARY:")
        print(f"  Total Logs Processed: {summary['total_logs_processed']:,}")
        print(f"  Successfully Parsed:  {summary['successfully_parsed']:,} ({summary['success_rate_percent']}%)")
        print(f"  Parsing Failures:     {summary['parsing_failures']:,} ({summary['failure_rate_percent']}%)")
        print(f"  Empty Logs:           {summary['empty_logs']:,} ({summary['empty_rate_percent']}%)")
        print(f"  Malformed Logs:       {summary['malformed_logs']:,} ({summary['malformed_rate_percent']}%)")
        
        # Quality Metrics
        quality = stats['quality_metrics']
        print(f"\n🎯 QUALITY METRICS:")
        print(f"  Parsing Efficiency:   {quality['parsing_efficiency']}")
        print(f"  Data Quality:         {quality['data_quality']}")
        print(f"  Usable Data Ratio:    {quality['usable_data_ratio']}%")
        
        # Error Analysis
        errors = stats['error_analysis']
        print(f"\n❌ ERROR ANALYSIS:")
        print(f"  Total Errors:         {errors['total_errors']:,}")
        print(f"  - Parsing Errors:     {errors['error_categories']['parsing_errors']:,}")
        print(f"  - Empty Entries:      {errors['error_categories']['empty_entries']:,}")
        print(f"  - Malformed Entries:  {errors['error_categories']['malformed_entries']:,}")
        
        # Format Analysis
        formats = stats['format_analysis']
        print(f"\n🔧 FORMAT ANALYSIS:")
        print(f"  Strategy: {formats['parsing_strategy']}")
        print(f"  Complexity: {formats['format_complexity']}")
        print(f"  Supported Formats: {len(formats['detected_formats'])} different structures")
        
        # Recommendations
        recommendations = stats['recommendations']
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*60)
        print("📈 PARSING COMPLETE")
        print("="*60)
    
    def export_statistics_report(self, parsing_stats: Dict[str, Any], 
                               output_path: str = "parsing_statistics_report.json") -> None:
        """Export detailed parsing statistics to JSON file."""
        import json
        from datetime import datetime
        
        stats = self.get_parsing_statistics(parsing_stats)
        
        # Add metadata
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'parser_version': '1.0.0',
                'report_type': 'Transaction Log Parsing Statistics'
            },
            'statistics': stats,
            'raw_parsing_data': parsing_stats
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Parsing statistics report exported to: {output_path}")
    
    def get_parsing_summary_table(self, parsing_stats: Dict[str, Any]) -> pd.DataFrame:
        """Generate a summary table of parsing statistics."""
        stats = self.get_parsing_statistics(parsing_stats)
        summary = stats['parsing_summary']
        
        # Create summary table
        summary_data = [
            ['Total Logs', summary['total_logs_processed'], '100.0%'],
            ['Successfully Parsed', summary['successfully_parsed'], f"{summary['success_rate_percent']}%"],
            ['Parsing Failures', summary['parsing_failures'], f"{summary['failure_rate_percent']}%"],
            ['Empty Logs', summary['empty_logs'], f"{summary['empty_rate_percent']}%"],
            ['Malformed Logs', summary['malformed_logs'], f"{summary['malformed_rate_percent']}%"]
        ]
        
        df_summary = pd.DataFrame(summary_data, columns=['Category', 'Count', 'Percentage'])
        return df_summary
