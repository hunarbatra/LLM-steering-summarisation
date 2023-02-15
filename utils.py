def gen_hash_key(text):
    return str(hash(text))

def read_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()
    
def save_file(file_name: str, data):
    with open(file_name, 'w') as outfile:
        outfile.write(data)