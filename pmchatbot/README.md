# CSV Chatbot

This project is a chatbot application that allows users to upload CSV files and interact with the data through a conversational interface. The chatbot is built using Chainlit and provides a user-friendly way to process and analyze CSV data.

## Project Structure

```
csv-chatbot
├── src
│   ├── app.py               # Entry point of the chatbot application
│   ├── chat_handler.py      # Manages conversation flow
│   ├── csv_processor.py     # Handles CSV file processing
│   └── utils
│       └── file_handler.py  # Utility functions for file operations
├── uploads                  # Directory for storing uploaded CSV files
├── requirements.txt         # Project dependencies
├── chainlit.md             # Chainlit specific documentation
└── README.md                # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/csv-chatbot.git
   cd csv-chatbot
   ```

2. **Install dependencies:**
   Make sure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application:**
   Start the chatbot by executing:
   ```
   python src/app.py
   ```

4. **Access the chatbot:**
   Open your web browser and navigate to `http://localhost:8501` to interact with the chatbot.

## Usage

- Upload a CSV file using the provided interface.
- Ask questions or request analyses based on the data in the uploaded CSV.
- The chatbot will process your requests and respond accordingly.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.