class CsvProcessor:
    def __init__(self):
        pass

    def read_csv(self, file_path, delimiter=',', encoding='utf-8'):
        import pandas as pd
        try:
            data = pd.read_csv(file_path, sep=delimiter, encoding=encoding)
            return data
        except Exception as e:
            return str(e)

    def detect_delimiter(self, file_path):
        """Automatically detect the delimiter in a CSV file"""
        import csv
        with open(file_path, 'r', encoding='utf-8') as file:
            sample = file.read(1024)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            return delimiter

    def read_csv_with_options(self, file_path, delimiter=None, encoding='utf-8', header='infer'):
        """Read CSV with multiple options"""
        import pandas as pd
        try:
            # If no delimiter specified, try to detect it
            if delimiter is None:
                delimiter = self.detect_delimiter(file_path)
            
            # Try different common delimiters if detection fails
            delimiters_to_try = [delimiter, ',', ';', '\t', '|']
            
            for delim in delimiters_to_try:
                try:
                    data = pd.read_csv(file_path, sep=delim, encoding=encoding, header=header)
                    if len(data.columns) > 1:  # If we got multiple columns, likely correct
                        return {
                            'data': data,
                            'delimiter': delim,
                            'encoding': encoding,
                            'success': True
                        }
                except:
                    continue
            
            # If all fail, return error
            return {
                'data': None,
                'delimiter': None,
                'encoding': encoding,
                'success': False,
                'error': 'Could not parse CSV with any common delimiter'
            }
            
        except Exception as e:
            return {
                'data': None,
                'delimiter': delimiter,
                'encoding': encoding,
                'success': False,
                'error': str(e)
            }

    def process_data(self, data):
        # Implement your data processing logic here
        processed_data = data.describe()  # Example: return summary statistics
        return processed_data