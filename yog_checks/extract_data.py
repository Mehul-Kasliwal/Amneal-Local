"""
Utility script to extract OCR page data from complete_data JSON files.

This module provides functions to:
1. Load and parse the complete_data JSON structure
2. Extract the 'extract_page_data' array which contains OCR page information
3. Provide data in a format compatible with check_42.py's calculate_sample_quantity_chunked()
"""

import json
from typing import Union, List, Dict, Any


def load_complete_data(json_path: str) -> Dict[str, Any]:
    """
    Load the complete data JSON file.
    
    Args:
        json_path: Path to the complete_data JSON file
        
    Returns:
        The full JSON data as a dictionary
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_page_data(json_path_or_obj: Union[str, dict]) -> List[Dict[str, Any]]:
    """
    Extract the 'extract_page_data' array from the complete data JSON.
    
    This is the main function to get OCR page data that can be used
    with check_42.py's calculate_sample_quantity_chunked().
    
    Args:
        json_path_or_obj: Either a file path (str) or already loaded JSON dict
        
    Returns:
        List of page dictionaries, each containing:
        - page: page number
        - page_content: extracted tables and key-value data
        - markdown_page: HTML/markdown representation
    """
    if isinstance(json_path_or_obj, str):
        data = load_complete_data(json_path_or_obj)
    else:
        data = json_path_or_obj
    
    # The OCR page data is stored under 'steps' -> 'extract_page_data' key
    if "steps" in data and "extract_page_data" in data["steps"]:
        return data["steps"]["extract_page_data"]
    
    # Fallback: check if 'extract_page_data' is at root level
    if "extract_page_data" in data:
        return data["extract_page_data"]
    
    # Fallback: check if it's already a list of pages
    if isinstance(data, list):
        return data
    
    raise ValueError("Could not find 'extract_page_data' in the JSON structure")


def get_filled_master_data(json_path: str) -> List[Dict[str, Any]]:
    """
    Convenience function to get the filled master data (extract_page_data)
    from the complete_data JSON file.
    
    This is the primary function to use when you need OCR data for validation checks.
    
    Args:
        json_path: Path to the complete_data JSON file
        
    Returns:
        List of page dictionaries with OCR content
    """
    return extract_page_data(json_path)


def get_filled_json(json_path_or_obj: Union[str, dict]) -> List[Dict[str, Any]]:
    """
    Extract the 'filled_json' array from the complete data JSON.
    
    filled_json contains processed OCR page data with:
    - page: page number
    - page_content: extracted tables and key-value data  
    - markdown_page: HTML/markdown representation
    
    Args:
        json_path_or_obj: Either a file path (str) or already loaded JSON dict
        
    Returns:
        List of page dictionaries with filled OCR content
    """
    if isinstance(json_path_or_obj, str):
        data = load_complete_data(json_path_or_obj)
    else:
        data = json_path_or_obj
    
    # The filled_json is stored under 'steps' -> 'filled_json' key
    if "steps" in data and "filled_json" in data["steps"]:
        return data["steps"]["filled_json"]
    
    # Fallback: check if 'filled_json' is at root level
    if "filled_json" in data:
        return data["filled_json"]
    
    # Fallback: check if it's already a list of pages
    if isinstance(data, list):
        return data
    
    raise ValueError("Could not find 'filled_json' in the JSON structure")


def get_filled_master_json(json_path_or_obj: Union[str, dict]) -> List[Dict[str, Any]]:
    """
    Extract the 'filled_master_json' array from the complete data JSON.
    
    filled_master_json contains the master OCR page data with:
    - page: page number
    - page_content: extracted tables and key-value data  
    - markdown_page: HTML/markdown representation
    
    Args:
        json_path_or_obj: Either a file path (str) or already loaded JSON dict
        
    Returns:
        List of page dictionaries with filled master OCR content
    """
    if isinstance(json_path_or_obj, str):
        data = load_complete_data(json_path_or_obj)
    else:
        data = json_path_or_obj
    
    # The filled_master_json is stored under 'steps' -> 'filled_master_json' key
    if "steps" in data and "filled_master_json" in data["steps"]:
        return data["steps"]["filled_master_json"]
    
    # Fallback: check if 'filled_master_json' is at root level
    if "filled_master_json" in data:
        return data["filled_master_json"]
    
    # Fallback: check if it's already a list of pages
    if isinstance(data, list):
        return data
    
    raise ValueError("Could not find 'filled_master_json' in the JSON structure")


def get_metadata(json_path: str) -> Dict[str, Any]:
    """
    Extract metadata from the complete_data JSON file.
    
    Args:
        json_path: Path to the complete_data JSON file
        
    Returns:
        Dictionary with batch metadata including:
        - batch_number
        - bmr_number
        - stage_id
        - user_id
        - filled_process_id
    """
    data = load_complete_data(json_path)
    
    return {
        "batch_number": data.get("batch_number"),
        "bmr_number": data.get("bmr_number"),
        "stage_id": data.get("stage_id"),
        "user_id": data.get("user_id"),
        "filled_process_id": data.get("filled_process_id"),
    }


def get_page_count(json_path_or_obj: Union[str, dict]) -> int:
    """
    Get the total number of pages in the extract_page_data.
    
    Args:
        json_path_or_obj: Either a file path or loaded JSON dict
        
    Returns:
        Number of pages
    """
    pages = extract_page_data(json_path_or_obj)
    return len(pages)


if __name__ == "__main__":
    # Example usage
    json_path = "/home/softsensor/Desktop/Amneal/yog_checks/complete_data_61 1.json"
    
    # Get metadata
    print("=== Metadata ===")
    metadata = get_metadata(json_path)
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Get page data
    print("\n=== Page Data ===")
    pages = get_filled_master_data(json_path)
    print(f"  Total pages: {len(pages)}")
    
    # Show first page structure
    if pages:
        print(f"\n=== First Page Structure ===")
        first_page = pages[0]
        print(f"  Page number: {first_page.get('page')}")
        print(f"  Has markdown_page: {'markdown_page' in first_page}")
        print(f"  Has page_content: {'page_content' in first_page}")
        if 'page_content' in first_page:
            print(f"  Number of content items: {len(first_page['page_content'])}")
