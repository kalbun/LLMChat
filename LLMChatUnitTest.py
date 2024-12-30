import unittest
from unittest.mock import MagicMock
from typing import List

# Assuming the LLMChat.py is imported and available as LLMChat
from LLMChat import LLM, Models

class TestLLM(unittest.TestCase):

    testOneShot1: str = "My name is Tom"
    testOneShot2: str = "What is my name? Answer one single word with no punctuation"
    testOneShotAnswer: str = "Tom"
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

        # one-shot conversation with queue
        llm.LLMCompletion(self.testOneShot1,queue)
        answer = llm.LLMCompletion(self.testOneShot2,queue)
        self.assertEqual(self.testOneShotAnswer, answer)
        self.assertEqual(len(queue),5)  # role + 2 messages + 2 answers
        queue.clear()

        # loss of memory test
        llm.LLMCompletion(self.testOneShot1)
        answer = llm.LLMCompletion(self.testOneShot2)
        self.assertNotEqual(self.testOneShotAnswer, answer)

        # zero-shot conversation
        answer = llm.LLMCompletion(self.testMessage)
        self.assertEqual(self.testMessage, answer)
        self.assertEqual(len(queue),0)

    def test_send_user_message_CLAUDE(self):
        queue: List[str] = []
        answer: str = ""
        llm: LLM = LLM(Models.CLAUDE)
        llm.DefineRole(self.testRole)

        # one-shot conversation with queue
        llm.LLMCompletion(self.testOneShot1,queue)
        answer = llm.LLMCompletion(self.testOneShot2,queue)
        self.assertEqual(self.testOneShotAnswer, answer)
        self.assertEqual(len(queue),4)  # role message not put in queue
        queue.clear()

        # loss of memory test
        llm.LLMCompletion(self.testOneShot1)
        answer = llm.LLMCompletion(self.testOneShot2)
        self.assertNotEqual(self.testOneShotAnswer, answer)

        # zero-shot conversation
        answer = llm.LLMCompletion(self.testMessage)
        self.assertEqual(self.testMessage, answer)
        self.assertEqual(len(queue),0)

if __name__ == '__main__':
    print("Running unit tests for LLMChat.py")
    print("This version tests GPT and CLAUDE, but not yet GEMINI")
    print("If all the tests succeed, you will see a message saying 'OK'")
    unittest.main(verbosity=1)
