"""
Configuration Loader Utility
============================

Handles loading and accessing configuration files, particularly SQL queries
from the YAML configuration file.
"""

import yaml
import os
from typing import Dict, Any, Optional
import streamlit as st

class ConfigLoader:
    """Loads and manages configuration settings and SQL queries"""
    
    def __init__(self, config_path: str = "config/queries.yaml"):
        """
        Initialize the configuration loader
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            st.error(f"Configuration file not found: {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            st.error(f"Error parsing configuration file: {e}")
            return {}
    
    def get_query(self, section: str, query_name: str) -> str:
        """
        Get a specific SQL query from the configuration
        
        Args:
            section: The section name in the YAML file
            query_name: The specific query name
            
        Returns:
            The SQL query string
        """
        try:
            return self.config[section][query_name]
        except KeyError:
            st.error(f"Query not found: {section}.{query_name}")
            return ""
    
    def get_section(self, section: str) -> Dict[str, str]:
        """
        Get all queries from a specific section
        
        Args:
            section: The section name in the YAML file
            
        Returns:
            Dictionary of query names and their SQL strings
        """
        try:
            return self.config[section]
        except KeyError:
            st.error(f"Section not found: {section}")
            return {}
    
    def reload_config(self) -> bool:
        """
        Reload the configuration file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.config = self._load_config()
            return True
        except Exception as e:
            st.error(f"Error reloading configuration: {e}")
            return False
    
    def get_all_sections(self) -> list:
        """
        Get list of all available sections
        
        Returns:
            List of section names
        """
        return list(self.config.keys()) 