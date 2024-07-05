import re
import pandas as pd


class CodeSplitter:

    def __init__(self, token_pattern = r'<extra_id_\d+>') -> None:
        self.token_pattern = token_pattern
        pass
    
        
    def get_classes(self, code: str) -> pd.DataFrame:
        class_start = r"@dataclass|\nclass "
        indexes = [match.start() for match in re.finditer(class_start, code)]
        indexes.append(code.find("var1"))
        class_list = []
        for i in range(len(indexes)):
            if i + 1 < len(indexes):
                class_list.append(code[indexes[i]:indexes[i+1]])
        new_class_list = []
        ignore_next = False
        for i in range(len(class_list)):
            if ignore_next:
                ignore_next = False
                continue
            if class_list[i].startswith("@dataclass"):
                class_item = class_list[i] + class_list[i+1]
                new_class_list.append(class_item)
                ignore_next = True
            else:
                new_class_list.append(class_list[i])
        return new_class_list
    
    def get_import(self, code: str) -> pd.DataFrame:
        index = code.find("from ")
        finish = code.find("@dataclass", index) 
        finish_other = code.find("class", index)
        if finish_other != -1 and finish_other < finish:
            finish = finish_other
        if finish == -1:
            finish = code.find("var1", index)
        return code[index:finish]

    def get_flow(self, code: str) -> pd.DataFrame:
        flow_start = "var1"
        index = code.find(flow_start)
        return code[index:]

    
        
    def get_dataclasses_and_flow(self, code: str) -> pd.DataFrame:
        classes = self.get_classes(code)
        flow = self.get_flow(code)
        item = ""
        for i in range(len(classes)):
            item += classes[i]
        item = item + flow
        return item
    
    def get_imports_and_flow(self, code: str) -> pd.DataFrame:
        imports = self.get_import(code)
        flow = self.get_flow(code)

        return imports + flow


    def split_code(self, code: str) -> pd.DataFrame:
        imports = self.get_import(code)
        flow = self.get_flow(code)
        classes = self.get_classes(code)
        return imports, flow, classes
