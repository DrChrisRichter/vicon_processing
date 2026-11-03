def read_enf(file_path):

    data = {}
    current_section = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Identify section headers like [Node Information]
            if line.startswith('[') and line.endswith(']'):
                current_section = line.strip('[]')
                data[current_section] = {}

            # Extract key-value pairs under the current section
            elif '=' in line and current_section is not None:
                # Split only on the first '=' in case the value contains one
                key, value = line.split('=', 1)

                # Try to convert numeric strings to actual integers (optional but helpful)
                value = value.strip()
                if value.isdigit():
                    value = int(value)

                data[current_section][key.strip()] = value

    return data


def write_enf(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for section, params in data.items():
            # Write the section header
            f.write(f"[{section}]\n")

            # Write each key-value pair under the section
            for key, value in params.items():
                # str(value) ensures integers are safely converted back to text
                f.write(f"{key}={str(value)}\n")