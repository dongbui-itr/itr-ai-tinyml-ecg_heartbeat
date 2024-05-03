class TextLogging:
    def __init__(self, file_path, mode):
        self.file_path = file_path
        self.mode = mode

        with open(file_path, mode) as test_file:
            pass

    def write_mylines(self, lines):
        with open(self.file_path, self.mode) as log_file:
            log_file.writelines(lines)
