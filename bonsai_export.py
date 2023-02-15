import json
import time

from utils import gen_hash_key, save_file


def create_root_node(root_text: str, 
                     parent_data: list = [], 
                     children_data: list = [], 
                     path: bool = False, 
                     root: bool = False,
                     path_nodes_list: list = []) -> dict:
    id = gen_hash_key(root_text)
    text = ' ' + root_text if root == False else root_text
    parent_ids = [gen_hash_key(parent_data[0])] if len(parent_data) else []
    children_ids = [gen_hash_key(i) for i in children_data] if len(children_data) else []
    last_visited = str(int(time.time()))
    curr_node_dict = {
        "id": id,
        "text": text,
        "parentIds": parent_ids,
        "childrenIds": children_ids,
        "lastVisited": last_visited
    }
    if path == True:
        curr_node_dict["group"] = "path"
        path_nodes_list.append(id) # add to path nodes list

    return curr_node_dict, path_nodes_list

def create_edges_dict(node: str, 
                      parent_node: str) -> dict:
    curr_edges_dict = {
        "from": gen_hash_key(parent_node),
        "to": gen_hash_key(node),
        "relation": "parentId"
    }
    
    return curr_edges_dict

def bonsai_export_runner(init_prompt: str, 
                         generated_data: list[list], 
                         ai_choices: list, 
                         json_name: str):
    root = init_prompt

    tree_data = [[root]] + generated_data

    nodes_dict_list, path_nodes_list, edges_dict_list = [], [], []

    # process root
    curr_node_dict, path_nodes_list = create_root_node(root_text = root, 
                                                       children_data = tree_data[1], 
                                                       path = True,
                                                       root = True,
                                                       path_nodes_list = path_nodes_list)
    nodes_dict_list.append(curr_node_dict)

    prev_root = root

    # process tree level 1..n
    for i in range(1, len(tree_data)):
        curr_level = tree_data[i]
        children_level = tree_data[i+1] if i < len(tree_data)-1 else []
        selected_node = curr_level[ai_choices[i-1]]
        for node in curr_level:
            path = True if node == selected_node else False
            curr_node_children = children_level if path == True else []
            curr_node_parents = [prev_root]
            curr_node_dict, path_nodes_list = create_root_node(node,
                                                               curr_node_parents,
                                                               curr_node_children,
                                                               path = path,
                                                               path_nodes_list = path_nodes_list)
            nodes_dict_list.append(curr_node_dict)
            curr_edges_dict = create_edges_dict(node, 
                                                prev_root)
            edges_dict_list.append(curr_edges_dict)

        prev_root = selected_node

    data_dict = {
        "nodes": nodes_dict_list,
        "edges": edges_dict_list,
        "name": "test",
        "pathNodes": path_nodes_list,
        "focusedId": gen_hash_key(tree_data[-1][ai_choices[-1]])
    }

    json_object = json.dumps(data_dict) 
    save_file('json_exports/' + json_name + '.json', json_object)
