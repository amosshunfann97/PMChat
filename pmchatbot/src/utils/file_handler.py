class FileHandler:
    def __init__(self, upload_directory='uploads'):
        self.upload_directory = upload_directory

    def save_file(self, file):
        file_path = self.get_file_path(file.filename)
        with open(file_path, 'wb') as f:
            f.write(file.read())
        return file_path

    def get_file_path(self, filename):
        return f"{self.upload_directory}/{filename}"