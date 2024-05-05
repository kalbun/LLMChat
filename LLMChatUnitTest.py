import unittest
from unittest.mock import patch, MagicMock
from typing import List

# Assuming the LLMChat.py is imported and available as LLMChat
from LLMChat import LLM, Models

class TestLLM(unittest.TestCase):

    testRole: str = "From now, echo back exactly what I write"
    testMessage: str = "*A* this is a test *B*"

    def setUp(self):
        # Patching the API client setup directly within the LLM class
        self.mock_client = MagicMock()
#        self.patcher = patch('LLMChat.LLM.LLMClient', new_callable=lambda: self.mock_client)
#        self.patcher.start()

    def tearDown(self):
        pass
#        self.patcher.stop()

    def test_initialization(self):
        # Test initialization with different models
        for model in Models:
            llm: LLM = LLM(model)
            self.assertEqual(llm.model, model)

    def test_define_role(self):
        for model in Models:
            llm: LLM = LLM(model)
            self.assertEqual(llm.roleInitMessage, "")  # not yet defined
            llm.DefineRole(self.testRole)
            self.assertEqual(llm.roleInitMessage, self.testRole)

    def test_send_user_message_GPT(self):
        queue: List[str] = []
        answer: str = ""
        llm: LLM = LLM(Models.GPT)
        llm.DefineRole(self.testRole)
        # new conversation appended to queue
        answer = llm.LLMCompletion(self.testMessage,queue,True)
        self.assertEqual(self.testMessage, answer)
        self.assertEqual(len(queue),3)
        queue.clear()
        # new conversation not appended to queue
        answer = llm.LLMCompletion(self.testMessage,queue,False)
        self.assertEqual(self.testMessage, answer)
        self.assertEqual(len(queue),0)

    def test_send_user_message_CLAUDE(self):
        queue: List[str] = []
        answer: str = ""
        llm: LLM = LLM(Models.CLAUDE)
        llm.DefineRole(self.testRole)
        # new conversation appended to queue
        answer = llm.LLMCompletion(self.testMessage,queue,True)
        self.assertEqual(self.testMessage, answer)
        self.assertEqual(len(queue),2)  # role message not put in queue
        queue.clear()
        # new conversation not appended to queue
        answer = llm.LLMCompletion(self.testMessage,queue,False)
        self.assertEqual(self.testMessage, answer)
        self.assertEqual(len(queue),0)

if __name__ == '__main__':
    unittest.main(verbosity=0)
