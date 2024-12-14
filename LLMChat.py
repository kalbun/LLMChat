#LLMChat.py
#
# This unit allows to run the application via GPT APIs or Claude APIs
#
from typing import List
from enum import Enum
# File key.ph contains the OpenAI key and is needed to execute the test.
# In case the import fails, shows instruction for creation of the file.
try:
    import key
except():
    raise ImportError(
            "\n***KEY UNDEFINED!\n\n" +\
            "Please create a file named key.py in the app directory " +\
            "and put your LLM key(s) there.\n\n" +\
            "According to the LLM you enabled, the application expects" +\
            " to find:"
            "\n\tGPTAIKey = \'<your key>\'" +\
            "\nand/or" +\
            "\n\tClaudeAIKey = \'<your key>\'" +\
            "\nand/or" +\
            "\n\tGeminiAIKey = \'<your key>\'" +\
            "\nwhere <your key> is the text you got from the vendor site."
    )

class Models(Enum): 
    GPT = 0
    CLAUDE = 1
    GEMINI = 2

class LLM():
    """
    Purpose: This class initializes an instance of the Language Learning Model (LLM) for conversational AI tasks.

    Usage:
    Instantiate the LLM class with the desired model (Models.GPT, Models.CLAUDE, or Models.GEMINI).
        Example: llm_instance = LLM(Models.GPT)
    Optionally, you can also pass a string to define the LLM role. The default is an empty string (no role).
    Call method DefineRole() to set it later.
        Example: llm_instance = LLM(Models.CLAUDE,"You are a personal shopper")

    Notes:
    - Ensure the required libraries are installed: openai for GPT, anthropic for CLAUDE, and google.generativeai for GEMINI.
    - If libraries are missing, the program will exit and prompt the user to install them.
    - Ensure the necessary API keys are defined in a file named key.py in the application directory.
    - For GPT, the key.py file should contain GPTAIKey.
    - For CLAUDE, the key.py file should contain ClaudeAIKey.
    - For GEMINI, the key.py file should contain GeminiAIKey.
    """
    model: Models = None   # LLM to use, no default
    exact_model: str = ""

    def __init__(self,_model: Models, _role: str = "", _exact_model: str = "") -> None:
        """
        Initializes the LLM instance with the specified model.

        Parameters:
        - model (Models): The LLM model to use (GPT, CLAUDE, or GEMINI).
        - exact_model: exact definition of the model to use, e.g. gpt-4o-mini.
          If omitted, uses defaults:
            gpt-4o-mini for GPT
            claude-3-5-sonnet-20241022 for CLAUDE
            gemini-1.0-pro for GEMINI
        """
        super().__init__()
        self.roleInitMessage = _role
        # Load libraries and throw an error if they are not installed
        try:
            if (_model == Models.GPT):
                from openai import OpenAI
            elif (_model == Models.CLAUDE):
                import anthropic
            elif (_model == Models.GEMINI):
                import google.generativeai
        except Exception as ex:
            raise ImportError(
                    "\n***MISSING LIBRARIES!\n\n" +\
                    "According to the LLM you use, you may need to install\n" +\
                    "openai and/or anthropic and/or google.generativeai\n"
                    "\nApplication will now exit."
            )
        # Dependencies are loaded, create the client
        if (_model == Models.GPT):
            if (_exact_model != ""):
                self.exact_model = _exact_model
            else:
                self.exact_model = "gpt-4o-mini"
            self.LLMClient = OpenAI(api_key = key.GPTAIKey())
        elif (_model == Models.CLAUDE):
            if (_exact_model != ""):
                self.exact_model = _exact_model
            else:
                self.exact_model="claude-3-5-sonnet-20241022"
            self.LLMClient = anthropic.Client(api_key= key.ClaudeAIKey)
        elif (_model == Models.GEMINI):
            if (_exact_model != ""):
                self.exact_model = _exact_model
            else:
                self.exact_model="gemini-1.0-pro"
            google.generativeai.configure(api_key=key.GeminiAIKey)
            # Set up the model
            generation_config = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 0,
            "max_output_tokens": 256,
            }
            safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            ]
            self.LLMClient = google.generativeai.GenerativeModel(
                model_name=self.exact_model,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

        self.model = _model

    def DefineRole(self, role : str):
        """
        Set the LLM "role". Note that Gemini currently does not offer this feature.

        Parameters:
        - role (str): defines the LLM role, like in:
            DefineRole("you are a medieval poet") 
        """
        self.roleInitMessage = role

    def LLMCompletion(self, userMessage: str,queue: List[str], append: bool = True) :
        """
        Call LLM in completion mode.

        Parameters:
        - userMessage (str): Message to send to LLM.
        - queue (List[str]): List containing the conversation with LLM.
          If parameter <append> is True, then user message and answer are added to
          the queue. Otherwise, the queue remains unchanged.
        - append (bool): True (default) if the pair user message/LLM answer must be
          added to <queue>. False otherwise.

        Returns:
        - str: LLM's response to the user's message.
        """
        LLMmessage : str = ""
        internalQueue : List[str] = []

        # Reload the key at each run. This allow to have multiple keys randomly selected.
        # It is not granted that this may have any practical effect, for example changing
        # LLM server or instance.
        if (self.model == Models.GPT):
            self.LLMClient.api_key = key.GPTAIKey()
        elif (self.model == Models.CLAUDE):
            self.LLMClient.api_key = key.ClaudeAIKey
        elif (self.model == Models.GEMINI):
            self.LLMClient.api_key = key.GeminiAIKey

        if (append):
            internalQueue = queue
        else:
            internalQueue = queue.copy()

        # If LLM is GPT, then add the system role as first item in the list.
        # Claude has a different method for setting it.
        if ( (len(internalQueue) == 0) and (self.model == Models.GPT) ):
            internalQueue.append({"role": "system", "content":self.roleInitMessage})

        # Now invoke the completion task with the new user message
        if (self.model == Models.GPT):
            internalQueue.append({"role":"user","content":userMessage})
            response = self.LLMClient.chat.completions.create(
            model=self.exact_model,
            messages= internalQueue,
#            temperature=1.065,
            max_tokens=256,
#            top_p=1,
#            frequency_penalty=0.05,
#            presence_penalty=0
            )
            LLMmessage = response.choices[0].message.content
            internalQueue.append({"role":"assistant","content":LLMmessage})
        elif (self.model == Models.CLAUDE):
            internalQueue.append({"role":"user","content":userMessage})
            response = self.LLMClient.messages.create(
            model=self.exact_model,
            system=self.roleInitMessage, # Claude has a specific parameter for role definition
            messages= internalQueue,
#            temperature=0.065,
            max_tokens=256,
#            top_p=1,
            )
            LLMmessage = response.content[0].text
            internalQueue.append({"role":"assistant","content":LLMmessage})
        elif (self.model == Models.GEMINI):
            internalQueue.append({"role":"user","parts":[userMessage]})
            convo = self.LLMClient.start_chat(history=internalQueue)
            convo.send_message(userMessage)
            LLMmessage = convo.last.text
            internalQueue.append({"role":"model","content":[LLMmessage]})

        return LLMmessage
