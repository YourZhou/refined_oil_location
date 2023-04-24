import json

# 请确保文件名和路径正确，这里假设文件名为 "json_package" 且与 Python 脚本在同一目录下
file_name = "location_file/points"

with open(file_name, "r", encoding="utf-8") as file:

    json_string = file.read()
    parsed_json = json.loads(json_string)

print(parsed_json)
