# Define the input and output file paths
input_file = 'data.csv'  # Replace with your file name
output_file = 'data1.csv'  # Replace with your desired output file name

# Open the input file for reading
with open(input_file, 'r') as file:
    # Read the content of the input file
    content = file.read()

# Replace tab characters with spaces
content = content.replace('\t', ' ')

# Open the output file for writing
with open(output_file, 'w') as file:
    # Write the modified content (now space-separated) to the output file
    file.write(content)

