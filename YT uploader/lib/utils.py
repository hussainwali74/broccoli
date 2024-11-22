def safe_title(title):
    """
    Create a safe title for Windows OS by removing or replacing invalid characters.
    
    Args:
        title (str): The original title.
    
    Returns:
        str: The safe title.
    """
    x =  "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in title)
    return x.replace(' ', '_')
