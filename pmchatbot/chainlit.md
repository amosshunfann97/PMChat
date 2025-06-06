# Chainlit Chatbot Documentation

## Overview
This document provides instructions on how to set up and run the Chainlit-based chatbot that allows users to upload CSV files for processing.

## Prerequisites
Before running the chatbot, ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package installer)

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd csv-chatbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Chatbot
To start the Chainlit chatbot, run the following command in your terminal:
```
chainlit run src/app.py
```

## Uploading CSV Files
Once the chatbot is running, you can interact with it through the web interface. To upload a CSV file:
1. Click on the upload button in the chatbot interface.
2. Select the CSV file you wish to upload.
3. The chatbot will process the file and respond based on its contents.

## Customization
You can customize the chatbot's behavior by modifying the `src/chat_handler.py` and `src/csv_processor.py` files. Adjust the conversation flow and data processing logic as needed.

## Troubleshooting
If you encounter any issues:
- Ensure all dependencies are installed correctly.
- Check the console for error messages and debug accordingly.

## Contribution
Feel free to contribute to the project by submitting issues or pull requests. Your feedback and improvements are welcome!