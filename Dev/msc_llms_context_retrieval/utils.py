
import pandas as pd
import re


NUM_NODE_TYPES = 14
NUM_CONTEXTS = 4
from CodeSplitter import CodeSplitter

def match_action(label : str) -> list[list[str, str]]:
    if label is None:
        return None

    action_pattern = re.compile(r'(\w+)\s*=\s*(.+?)(?=,\s*\w+\s*=|$)')
    matches = action_pattern.findall(label)
    action_list = []
    for match in matches:
        if(match[1] != ""):
            tuple_match = [match[0], match[1]]
        else:
            tuple_match = [match[0], match[2]]  
        action_list.append(tuple_match)
    return action_list

def process_labels(label : str, output_columns : str =['node_type', 'left_action']) -> pd.Series:
    """Function to process the labels from the dataset and the generated labels. ~
    The labels are in the format node_type(action1=action1Text, action2=action2Text, ...)

    Args:
        label (str): Label to Process
        output_columns (list, optional): Resulting collumn names types from that processing. Defaults to ['node_type', 'left_action'].

    Returns:
        pd.Series: A pandas series with the resulting columns, node_type and action from each part of the label
    """
    label_pattern = re.compile(r'(\w+)\((.*)\)')
    end_pattern = re.compile(r'<END>')
    action_pattern = re.compile(r'(\w+)\s*=\s*(.+?)(?=,\s*\w+\s*=|$)')
    # If the label is None, return None for both columns  
    if label is None:
        return pd.Series({output_columns[0]: None, output_columns[1]: None})
    
    
    match_label_action = label_pattern.match(label)

    if match_label_action:
        node_type = match_label_action.group(1)
        if node_type == "ISQLNode":
            return pd.Series({output_columns[0]: node_type, output_columns[1]: [('', match_label_action.group(2))]})
        action_list = match_action(match_label_action.group(2))
        
        return pd.Series({output_columns[0]: node_type, output_columns[1]: action_list})
    # If the label is <END>, return <END> for the node_type and None for the action
    elif end_pattern.match(label):
        return pd.Series({output_columns[0]: label, output_columns[1]: None})
    else:
        return pd.Series({output_columns[0]: None, output_columns[1]: None})


def label2id() -> dict[str, int]:
    return {
        # Node type Part
        "<END>": 0,
        "IIfNode": 1,
        "IJSONSerializeNode": 2,
        "INRSendEmailNode": 3,
        "IAssignment": 4,
        "ISendEmailNode": 5,
        "IRaiseExceptionNode": 6,
        "IRecordListToExcelNode": 7,
        "IForEachNode": 8,
        "IExcelToRecordListNode": 9,
        "ISQLNode": 10,
        "IJSONDeserializeNode": 11,
        "IAggregateNode": 12,
        "IExecuteServerActionNode": 13,

        # Context Part
        "full_context": 14,
        "flow_only": 15,
        "flow_and_imports": 16,
        "flow_and_dataclasses": 17
    }

def nodetype2id() -> dict[str, int]:
    return {
        "<END>": 0,
        "IIfNode": 1,
        "IJSONSerializeNode": 2,
        "INRSendEmailNode": 3,
        "IAssignment": 4,
        "ISendEmailNode": 5,
        "IRaiseExceptionNode": 6,
        "IRecordListToExcelNode": 7,
        "IForEachNode": 8,
        "IExcelToRecordListNode": 9,
        "ISQLNode": 10,
        "IJSONDeserializeNode": 11,
        "IAggregateNode": 12,
        "IExecuteServerActionNode": 13
    }

def id2nodetype() -> dict[int, str]:
    return {
        0: "<END>",
        1: "IIfNode",
        2: "IJSONSerializeNode",
        3: "INRSendEmailNode",
        4: "IAssignment",
        5: "ISendEmailNode",
        6: "IRaiseExceptionNode",
        7: "IRecordListToExcelNode",
        8: "IForEachNode",
        9: "IExcelToRecordListNode",
        10: "ISQLNode",
        11: "IJSONDeserializeNode",
        12: "IAggregateNode",
        13: "IExecuteServerActionNode"
    }


def id2label() -> dict[int, str]:
    return {
        # Node type Part
        0: "<END>",
        1: "IIfNode",
        2: "IJSONSerializeNode",
        3: "INRSendEmailNode",
        4: "IAssignment",
        5: "ISendEmailNode",
        6: "IRaiseExceptionNode",
        7: "IRecordListToExcelNode",
        8: "IForEachNode",
        9: "IExcelToRecordListNode",
        10: "ISQLNode",
        11: "IJSONDeserializeNode",
        12: "IAggregateNode",
        13: "IExecuteServerActionNode",
        
        # Context Part
        14: "full_context",
        15: "flow_only",
        16: "flow_and_imports",
        17: "flow_and_dataclasses"
    }

def label_to_bin_node_only(node_type : int) -> list:
    node_type_list = [0] * NUM_NODE_TYPES
    node_type_list[node_type] = 1
   
    return node_type_list

def get_dataclass_name(class_text : str) -> str:
    if class_text.find("class") == 5:
        class_text = class_text[11:] 
    pos_begin = class_text.find("class") + 6
    pos_end = class_text.find(":")
    class_name = class_text[pos_begin:pos_end]

    class_name = class_name.split("(")[0]
    return class_name


def get_dataclass_names(text: str) -> list[str]:
    dataclasses_list = []
    cd = CodeSplitter()
    dataclasses = cd.get_classes(text)
    for dataclass in dataclasses:
        dataclasse_name = get_dataclass_name(dataclass)
        dataclasses_list.append(dataclasse_name)
    entities = set(dataclasses_list)
    if len(entities) == 0:
        return None
    entities_str = ", ".join(entities)

    return entities_str

def get_dataclass_names_and_text(text: str) -> list[list[str]]:
    dataclasses_list = []
    cd = CodeSplitter()
    dataclasses = cd.get_classes(text)
    for dataclass in dataclasses:
        dataclasse_name = get_dataclass_name(dataclass)
        dataclasses_list.append(dataclasse_name)
  
    

    return dataclasses_list, dataclasses

def node_type_context_to_label(node_type : str, context : str) -> str:
    return f"{node_type}({context})"

def label_to_bin(node_type : int, context : int) -> list:
    node_type_list = [0] * NUM_NODE_TYPES
    context_list = [0] * NUM_CONTEXTS
    node_type_list[node_type] = 1
    context_list[context] = 1
    return node_type_list + context_list

def bin_to_label(bin_list : list) -> tuple:
    node_type = bin_list.index(1)
    context = bin_list.index(1, 14) - 14
    return (node_type, context)


def label_to_str(label: tuple[int]) -> str:
    return (id2label()[label[0]],id2label()[label[1] + 14])


def context_indices():
    return range(14, 18)

def node_type_indices():
    return range(0, 14)

