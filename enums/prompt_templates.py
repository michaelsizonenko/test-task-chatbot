from enum import Enum


class PromptTemplates(str, Enum):
    CONTEXT_RETRIEVAL_TEMPLATE = (
        """
        Use the following pieces of context to answer the question at the end.
        If you do not know the answer, just say that you do not know, do not try to make up an answer.
    
        {context}
        """
    )

    CONTEXTUALIZE_SYSTEM_PROMPT = (
        """
        Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is.
        """
    )