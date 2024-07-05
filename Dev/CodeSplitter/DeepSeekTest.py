from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

input_text = """<｜fim▁begin｜>
from ORD_G_PROCESS_CS.ServerActions import ProcessControl_GetBySwitchNumber, ProcessControlHistory_UpdateProcessHistoryStatus, ProcessControl_UpdateProcessStatus, ProcessControlHistory_GetByMessageQueueId
from MARTE_COMMON_BL.ServerActions import ExceptionHandler, Generate_ComputeHash
from (System).ServerActions import CommitTransaction
from ORD_G_PROCESS_BL.ServerActions import ReceiveH1, ProcessControl_SetSwitchDate, Message_GetReferenceInfoQ00, ProcessControl_History_SetStatus
from REN_QUEUE_CS.ServerActions import MessageQueuePayload_GetXML, MessageQueue_UpdateStatus, MessageQueue_Update, MessageQueue_UpdateComputeHash
from REN_QUEUE_DISPATCHER_BL.ServerActions import DispatchOutboundQueueEntries, MessageHandler, CheckDuplicatedPayload, DispatchInboundQueueEntries, MessageQueue_SetHash, MessageQueue_SetDataQ00, MessageInboundRouting, Validate_MessageQueue, Reprocessing_MessageQueue, Reprocessing_SetStatus
from SiteProperties import TenantId, TenantName, Timeout_REN_Queue_Inbound, Timeout_REN_Queue_Outbound
class MessageQueueStatus(Enum):
        \"\"\"MessageQueueStatus\"\"\"
        Error = 1
        Delivered = 2
        NotImplemented = 3
        Processing = 4
        Duplicated = 5
        New = 6
        OnHold = 7
class Flow(Enum):
        M1 = 1
        E4 = 2
        J1 = 3
        Q00 = 4
        H2 = 5
        D7 = 6
        E1 = 7
        D6 = 8
        E2 = 9
        H11 = 10
        H3 = 11
        E5 = 12
        H1 = 13
class EnergyType(Enum):
        \"\""EnergyType\"\""
        Power = 1
        Gas = 2
var1 = IServerAction(Name=MessageOutboundRouting, InputParameters={"FlowId": Flow, "Payload": str}, OutputParameters={"MessageQueueStatusId": MessageQueueStatus, "Ok": bool})
var2 = IStartNode()
var1.add_subflow(var2)
var3 = ISwitchNode()
var2.sequence = var3
var4 = IAssignment(Variable=MessageQueueStatusId, Value=MessageQueueStatus.NotImplemented)
var3.otherwise = var4
var5 = <｜fim▁hole｜>
var4.sequence = var5
var6 = IEndNode()
var5.undefined = var6
var7 = IAssignment(Variable=Ok, Value=True)
var3.condition = var7
var8 = IEndNode()
var7.undefined = var8<｜fim▁end｜>"""

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_text):])