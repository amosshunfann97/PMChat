class ChatHandler:
    def __init__(self, csv_processor):
        self.csv_processor = csv_processor

    def handle_message(self, message):
        if message.startswith("upload:"):
            file_path = message.split(":", 1)[1].strip()
            return self.process_upload(file_path)
        else:
            return self.generate_response(message)

    def process_upload(self, file_path):
        try:
            data = self.csv_processor.read_csv(file_path)
            processed_data = self.csv_processor.process_data(data)
            return f"Successfully uploaded and processed the CSV file. Data: {processed_data}"
        except Exception as e:
            return f"Error processing the file: {str(e)}"

    def generate_response(self, message):
        # Placeholder for generating responses based on user input
        return f"You said: {message}"