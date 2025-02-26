from typing import Any, Dict, Optional, Union, List, Tuple
from faker import Faker
import random
import numpy as np

class DataGenerator:
    """
    A utility class that generates fake data using the Faker library
    based on specified keys.
    """
    
    def __init__(self, locale: str = 'en_US', seed: Optional[int] = None):
        """
        Initialize the FakerGenerator with a specific locale and optional seed.
        
        Args:
            locale (str): The locale to use for generating data (default: 'en_US')
            seed (Optional[int]): Optional seed for reproducible data generation
        """
        self.faker = Faker(locale)
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        
        # Map of keys to faker methods
        self.key_map: Dict[str, str] = {
            # Person
            'name': 'name',
            'first_name': 'first_name',
            'last_name': 'last_name',
            'middle_name': 'middle_name',
            'full_name': 'name',
            'prefix': 'prefix',
            'suffix': 'suffix',
            'job': 'job',
            'job_title': 'job',
            
            # Contact
            'email': 'email',
            'phone': 'phone_number',
            'phone_number': 'phone_number',
            'cell': 'cell_phone',
            'cell_phone': 'cell_phone',
            
            # Internet
            'username': 'user_name',
            'password': 'password',
            'domain': 'domain_name',
            'url': 'url',
            'ipv4': 'ipv4',
            'ipv6': 'ipv6',
            'mac_address': 'mac_address',
            'user_agent': 'user_agent',
            
            # Address
            'address': 'address',
            'street': 'street_address',
            'street_address': 'street_address',
            'city': 'city',
            'state': 'state',
            'state_abbr': 'state_abbr',
            'zipcode': 'zipcode',
            'zip': 'zipcode',
            'postal_code': 'zipcode',
            'country': 'country',
            'country_code': 'country_code',
            'latitude': 'latitude',
            'longitude': 'longitude',
            
            # Date & Time
            'date': 'date',
            'time': 'time',
            'datetime': 'date_time',
            'timestamp': 'unix_time',
            
            # Finance
            'credit_card': 'credit_card_number',
            'credit_card_number': 'credit_card_number',
            'credit_card_provider': 'credit_card_provider',
            'credit_card_expire': 'credit_card_expire',
            'credit_card_security_code': 'credit_card_security_code',
            'currency': 'currency_code',
            'currency_code': 'currency_code',
            'currency_name': 'currency_name',
            
            # Text
            'text': 'text',
            'paragraph': 'paragraph',
            'sentence': 'sentence',
            'word': 'word',
            
            # Identifiers
            'uuid': 'uuid4',
            'ssn': 'ssn',
            
            # Company
            'company': 'company',
            'company_name': 'company',
            'bs': 'bs',
            'catch_phrase': 'catch_phrase',
            
            # Misc
            'color': 'color_name',
            'hex_color': 'hex_color',
            'rgb_color': 'rgb_color',
            
            # Numeric (special handling for these)
            'number': '_special_number',
            'integer': '_special_integer',
            'decimal': '_special_decimal',
            'float': '_special_float',
        }
        
        # Map of keys to descriptions for LLM assistance
        self.key_descriptions: Dict[str, str] = {
            # Person
            'name': 'A full name (first and last name)',
            'first_name': 'A first name only',
            'last_name': 'A last name only',
            'middle_name': 'A middle name',
            'full_name': 'A complete name including first and last name',
            'prefix': 'A name prefix or title (e.g., Mr., Mrs., Dr.)',
            'suffix': 'A name suffix (e.g., Jr., Sr., MD, PhD)',
            'job': 'A job title or occupation',
            'job_title': 'A professional job title',
            
            # Contact
            'email': 'An email address',
            'phone': 'A phone number with country code',
            'phone_number': 'A formatted phone number',
            'cell': 'A cell phone number',
            'cell_phone': 'A mobile phone number',
            
            # Internet
            'username': 'A username for online accounts',
            'password': 'A random password',
            'domain': 'A domain name',
            'url': 'A complete URL',
            'ipv4': 'An IPv4 address',
            'ipv6': 'An IPv6 address',
            'mac_address': 'A MAC address',
            'user_agent': 'A browser user agent string',
            
            # Address
            'address': 'A complete mailing address',
            'street': 'A street address',
            'street_address': 'A street address with number',
            'city': 'A city name',
            'state': 'A state or province name',
            'state_abbr': 'A state or province abbreviation',
            'zipcode': 'A postal code',
            'zip': 'A ZIP code (US postal code)',
            'postal_code': 'A postal code',
            'country': 'A country name',
            'country_code': 'A two-letter country code',
            'latitude': 'A latitude coordinate',
            'longitude': 'A longitude coordinate',
            
            # Date & Time
            'date': 'A date in various formats',
            'time': 'A time in various formats',
            'datetime': 'A combined date and time',
            'timestamp': 'A Unix timestamp',
            
            # Finance
            'credit_card': 'A credit card number',
            'credit_card_number': 'A valid credit card number',
            'credit_card_provider': 'A credit card provider name (e.g., Visa, Mastercard)',
            'credit_card_expire': 'A credit card expiration date',
            'credit_card_security_code': 'A credit card security code (CVV/CVC)',
            'currency': 'A three-letter currency code',
            'currency_code': 'A three-letter currency code (e.g., USD, EUR)',
            'currency_name': 'A currency name (e.g., Dollar, Euro)',
            
            # Text
            'text': 'A random text paragraph',
            'paragraph': 'A paragraph of text',
            'sentence': 'A single sentence',
            'word': 'A single word',
            
            # Identifiers
            'uuid': 'A UUID (universally unique identifier)',
            'ssn': 'A Social Security Number (US)',
            
            # Company
            'company': 'A company name',
            'company_name': 'A business name',
            'bs': 'Business speak jargon',
            'catch_phrase': 'A company catch phrase or slogan',
            
            # Misc
            'color': 'A color name',
            'hex_color': 'A hex color code',
            'rgb_color': 'An RGB color tuple',
            
            # Numeric
            'number': 'A numeric value that can be customized with statistical parameters',
            'integer': 'An integer value that can be customized with statistical parameters',
            'decimal': 'A decimal number with configurable precision',
            'float': 'A floating-point number that can be customized with statistical parameters',
        }
        
        # Store numeric column stats for reference
        self.numeric_stats: Dict[str, Dict[str, Any]] = {}
    
    def get_for_key(self, key: str, **kwargs: Any) -> Any:
        """
        Generate a fake value based on the provided key.
        
        Args:
            key (str): The type of data to generate
            **kwargs: Additional arguments to pass to the Faker method
            
        Returns:
            Any: The generated fake value
            
        Raises:
            ValueError: If the key is not supported
        """
        # Convert key to lowercase and remove spaces for consistent lookup
        normalized_key = key.lower().replace(' ', '_')
        
        if normalized_key in self.key_map:
            # Get the corresponding faker method name
            method_name = self.key_map[normalized_key]
            
            # Special handling for numeric types
            if method_name.startswith('_special_'):
                # Extract the numeric type from the method name
                numeric_type = method_name.replace('_special_', '')
                
                # Handle numeric generation based on type
                if numeric_type in ['number', 'float', 'decimal']:
                    # Get column name if provided to use stored stats
                    column_name = kwargs.pop('column_name', None)
                    stats = None
                    
                    if column_name and column_name in self.numeric_stats:
                        stats = self.numeric_stats[column_name]
                    
                    # Use provided stats or defaults
                    if not stats:
                        stats = {
                            'min': kwargs.pop('min', 0),
                            'max': kwargs.pop('max', 100),
                            'mean': kwargs.pop('mean', 50),
                            'stddev': kwargs.pop('stddev', 15),
                        }
                    
                    # Get generation parameters
                    distribution = kwargs.pop('distribution', 'normal')
                    round_to = kwargs.pop('round_to', 2 if numeric_type in ['decimal', 'float'] else None)
                    
                    # Set as_int based on the numeric type
                    as_int = False
                    if numeric_type == 'integer':
                        as_int = True
                    else:
                        # Only use as_int from kwargs if not already determined by type
                        as_int = kwargs.pop('as_int', False)
                    
                    # Generate the number
                    return self.generate_number(
                        stats=stats,
                        distribution=distribution,
                        round_to=round_to,
                        as_int=as_int,
                        **kwargs
                    )
                elif numeric_type == 'integer':
                    # Get column name if provided to use stored stats
                    column_name = kwargs.pop('column_name', None)
                    stats = None
                    
                    if column_name and column_name in self.numeric_stats:
                        stats = self.numeric_stats[column_name]
                    
                    # Use provided stats or defaults
                    if not stats:
                        stats = {
                            'min': kwargs.pop('min', 0),
                            'max': kwargs.pop('max', 100),
                            'mean': kwargs.pop('mean', 50),
                            'stddev': kwargs.pop('stddev', 15),
                        }
                    
                    # Generate the integer (always as_int=True for integer type)
                    return self.generate_number(
                        stats=stats,
                        distribution=kwargs.pop('distribution', 'normal'),
                        as_int=True,
                        **kwargs
                    )
            else:
                # Get the method from faker
                faker_method = getattr(self.faker, method_name)
                # Call the method with any additional arguments
                return faker_method(**kwargs)
        else:
            raise ValueError(f"Unsupported key: {key}. Available keys: {', '.join(sorted(self.key_map.keys()))}")
    
    def get_multiple(self, keys: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate multiple fake values based on a dictionary of keys and their parameters.
        
        Args:
            keys (Dict[str, Dict[str, Any]]): Dictionary mapping output keys to 
                                             dictionaries containing 'type' and optional parameters
        
        Returns:
            Dict[str, Any]: Dictionary with generated values
            
        Example:
            >>> faker_gen = FakerGenerator()
            >>> result = faker_gen.get_multiple({
            ...     'user_name': {'type': 'name'},
            ...     'user_email': {'type': 'email'},
            ...     'birth_date': {'type': 'date', 'pattern': '%Y-%m-%d'}
            ... })
        """
        result = {}
        for output_key, config in keys.items():
            if 'type' not in config:
                raise ValueError(f"Missing 'type' for key: {output_key}")
            
            faker_key = config.pop('type')
            result[output_key] = self.get_for_key(faker_key, **config)
        
        return result
    
    def get_key_descriptions(self, category: Optional[str] = None) -> Dict[str, str]:
        """
        Get descriptions for all available keys to help LLMs choose the correct key.
        
        Args:
            category (Optional[str]): Filter keys by category (person, contact, internet, 
                                     address, date, finance, text, identifiers, company, misc)
                                     If None, returns all keys.
        
        Returns:
            Dict[str, str]: Dictionary mapping keys to their descriptions
            
        Example:
            >>> faker_gen = FakerGenerator()
            >>> # Get all key descriptions
            >>> all_descriptions = faker_gen.get_key_descriptions()
            >>> # Get only person-related keys
            >>> person_keys = faker_gen.get_key_descriptions(category="person")
        """
        if category is None:
            return self.key_descriptions
        
        category = category.lower()
        
        # Define which keys belong to which categories
        categories = {
            'person': ['name', 'first_name', 'last_name', 'middle_name', 'full_name', 'prefix', 'suffix', 'job', 'job_title'],
            'contact': ['email', 'phone', 'phone_number', 'cell', 'cell_phone'],
            'internet': ['username', 'password', 'domain', 'url', 'ipv4', 'ipv6', 'mac_address', 'user_agent'],
            'address': ['address', 'street', 'street_address', 'city', 'state', 'state_abbr', 'zipcode', 'zip', 'postal_code', 'country', 'country_code', 'latitude', 'longitude'],
            'date': ['date', 'time', 'datetime', 'timestamp'],
            'finance': ['credit_card', 'credit_card_number', 'credit_card_provider', 'credit_card_expire', 'credit_card_security_code', 'currency', 'currency_code', 'currency_name'],
            'text': ['text', 'paragraph', 'sentence', 'word'],
            'identifiers': ['uuid', 'ssn'],
            'company': ['company', 'company_name', 'bs', 'catch_phrase'],
            'misc': ['color', 'hex_color', 'rgb_color']
        }
        
        if category not in categories:
            valid_categories = list(categories.keys())
            raise ValueError(f"Invalid category: {category}. Available categories: {', '.join(valid_categories)}")
        
        # Filter descriptions by category
        return {key: self.key_descriptions[key] for key in categories[category] if key in self.key_descriptions}

    def generate_number(self, stats: Dict[str, Any], count: int = 1, 
                       distribution: str = 'normal', 
                       round_to: Optional[int] = None,
                       as_int: bool = False) -> Union[float, int, List[Union[float, int]]]:
        """
        Generate numbers based on statistical information.
        
        Args:
            stats (Dict[str, Any]): Dictionary containing statistical information about the data.
                                   Should include at least 'mean' and 'stddev' for normal distribution,
                                   or 'min' and 'max' for uniform distribution.
            count (int): Number of values to generate
            distribution (str): Distribution to use ('normal', 'uniform', 'lognormal')
            round_to (Optional[int]): Number of decimal places to round to (None for no rounding)
            as_int (bool): Whether to return integers
            
        Returns:
            Union[float, int, List[Union[float, int]]]: Generated number(s)
            
        Example:
            >>> generator = DataGenerator()
            >>> # Generate a number similar to those in a database column
            >>> stats = {
            ...     'min': 10.5,
            ...     'max': 100.3,
            ...     'mean': 55.7,
            ...     'median': 54.2,
            ...     'stddev': 15.3,
            ...     'q1': 42.1,
            ...     'q3': 68.9
            ... }
            >>> generator.generate_number(stats, distribution='normal', as_int=True)
        """
        if not stats:
            raise ValueError("Stats dictionary cannot be empty")
        
        # Generate values based on the specified distribution
        if distribution == 'normal':
            if 'mean' not in stats or 'stddev' not in stats:
                raise ValueError("Normal distribution requires 'mean' and 'stddev' in stats")
            
            values = np.random.normal(stats['mean'], stats['stddev'], count)
            
            # Optionally constrain to min/max if provided
            if 'min' in stats and 'max' in stats:
                values = np.clip(values, stats['min'], stats['max'])
                
        elif distribution == 'uniform':
            if 'min' not in stats or 'max' not in stats:
                raise ValueError("Uniform distribution requires 'min' and 'max' in stats")
                
            values = np.random.uniform(stats['min'], stats['max'], count)
            
        elif distribution == 'lognormal':
            if 'mean' not in stats or 'stddev' not in stats:
                raise ValueError("Lognormal distribution requires 'mean' and 'stddev' in stats")
                
            # Convert normal mean/stddev to lognormal parameters
            if stats['mean'] <= 0:
                raise ValueError("Lognormal distribution requires positive mean")
                
            # Calculate mu and sigma for lognormal distribution
            variance = stats['stddev'] ** 2
            mu = np.log(stats['mean'] ** 2 / np.sqrt(variance + stats['mean'] ** 2))
            sigma = np.sqrt(np.log(1 + variance / stats['mean'] ** 2))
            
            values = np.random.lognormal(mu, sigma, count)
            
            # Optionally constrain to min/max if provided
            if 'min' in stats and 'max' in stats:
                values = np.clip(values, stats['min'], stats['max'])
                
        else:
            # Fallback to normal distribution
            # print(f"Warning: Unsupported distribution '{distribution}'. Falling back to normal distribution.")
            
            # Check if we have mean and stddev
            if 'mean' not in stats or 'stddev' not in stats:
                # Use default values if not available
                mean = 50
                stddev = 10
                print(f"Warning: Missing 'mean' or 'stddev' in stats. Using defaults: mean={mean}, stddev={stddev}")
            else:
                mean = stats['mean']
                stddev = stats['stddev']
                
            values = np.random.normal(mean, stddev, count)
            
            # Optionally constrain to min/max if provided
            if 'min' in stats and 'max' in stats:
                values = np.clip(values, stats['min'], stats['max'])
        
        # Apply rounding if specified
        if round_to is not None:
            values = np.round(values, round_to)
            
        # Convert to integers if requested
        if as_int:
            values = np.round(values).astype(int)
        
        # Return single value or list based on count
        if count == 1:
            return values[0]
        else:
            return values.tolist()
            
    def generate_numbers_from_db_stats(self, db_stats: Dict[str, Any], count: int = 1,
                                      distribution: str = 'auto',
                                      round_to: Optional[int] = None,
                                      as_int: bool = False) -> Union[float, int, List[Union[float, int]]]:
        """
        Generate numbers based on database column statistics from DatabaseExtractor.
        
        Args:
            db_stats (Dict[str, Any]): Statistics dictionary from DatabaseExtractor.get_numeric_column_stats()
            count (int): Number of values to generate
            distribution (str): Distribution to use ('auto', 'normal', 'uniform', 'lognormal')
            round_to (Optional[int]): Number of decimal places to round to (None for no rounding)
            as_int (bool): Whether to return integers
            
        Returns:
            Union[float, int, List[Union[float, int]]]: Generated number(s)
            
        Example:
            >>> from db import DatabaseExtractor
            >>> db_extractor = DatabaseExtractor('my_database.db')
            >>> stats = db_extractor.get_numeric_column_stats('users', 'age')
            >>> generator = DataGenerator()
            >>> generator.generate_numbers_from_db_stats(stats, count=5, as_int=True)
        """
        # Auto-detect the best distribution if set to 'auto'
        if distribution == 'auto':
            # Check if we have enough stats to make a decision
            if all(key in db_stats for key in ['min', 'max', 'mean', 'median', 'stddev']):
                # Check for skewness by comparing mean and median
                skewness = abs(db_stats['mean'] - db_stats['median']) / db_stats['stddev'] if db_stats['stddev'] > 0 else 0
                
                if skewness > 0.5:
                    # Data is skewed, use lognormal if mean is positive
                    distribution = 'lognormal' if db_stats['mean'] > 0 else 'normal'
                else:
                    # Data is not significantly skewed, use normal
                    distribution = 'normal'
            else:
                # Default to uniform if we don't have enough stats
                distribution = 'uniform' if all(key in db_stats for key in ['min', 'max']) else 'normal'
        
        # Generate the numbers using the selected distribution
        return self.generate_number(
            stats=db_stats,
            count=count,
            distribution=distribution,
            round_to=round_to,
            as_int=as_int
        )

    def set_numeric_stats(self, column_name: str, stats: Dict[str, Any]) -> None:
        """
        Store statistical information for a numeric column to use in data generation.
        
        Args:
            column_name (str): Name of the column
            stats (Dict[str, Any]): Statistical information about the column
        """
        self.numeric_stats[column_name] = stats
