
import os
import cv2
import json



def generate_tree(source, parent):
    tree = []
    for item in source:
        if item["parent"] == parent:
            item["child"] = generate_tree(source, item["id"])
            tree.append(item)
    return tree


f = open("./data/category.txt")
lines = f.readlines()

permission_source = []
for line in lines:
    per_line = line.strip().split('    ')
    per_item_dict = {}
    per_item_dict["id"] = per_line[0]
    if per_line[1] != "null":
        per_item_dict["parent"] = per_line[1]
    else:
        per_item_dict["parent"] = 0
    per_item_dict["name"] = per_line[2]
    permission_source.append( per_item_dict )


permission_tree = generate_tree(permission_source, 0)

print(json.dumps(permission_tree, ensure_ascii=False))


