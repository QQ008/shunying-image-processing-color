import pickle


# 从文件中读取对象
with open('features.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

print(loaded_data)  # {'name': 'Alice', 'age': 30}