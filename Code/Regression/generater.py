import json
import random

# 打开已有的文件，生成 1000w 条已有文件中不存在的数据
def generate_unique_data(filename, existing_data):
    # 获取现有数据中的"aa"字段集合
    existing_aa_set = set(entry["aa"] for entry in existing_data)

    # 定义aa字符串的可能字符

    aa_characters = "KLMNPQRSTVWYACDEFGHI"

    # 生成1000w条数据，保证"aa"字段不重复
    new_data = []
    while len(new_data) < 1000000:
        # 随机生成一个7位长度的字符串
        random_aa = "".join(random.choice(aa_characters) for _ in range(7))
        if random_aa not in existing_aa_set:
            data_entry = {"aa": random_aa, "nor_package": 0}
            new_data.append(data_entry)
            existing_aa_set.add(random_aa)

    # 将新生成的数据合并到现有数据中
    combined_data = new_data

    # 将数据保存到json文件
    with open(filename, 'w') as f:
        json.dump(combined_data, f)


if __name__ == '__main__':
    get_pre_data = json.load(open('PreData&Res/Data/1000wData.json', 'r'))
    #pre_dataset = Amino(get_pre_data)
    filename  = 'PreData&Res/Data/100wData.json'
    generate_unique_data(filename,get_pre_data)