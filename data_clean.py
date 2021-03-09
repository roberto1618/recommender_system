def data_clean(data):
    # Remove duplicate rows
    data_cleaned = data.drop_duplicates()
    return data_cleaned