# rag_agent.py
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.docstore.document import Document
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class TherapyRAGAgent:
    def __init__(self):
        try:
            self.embeddings = OpenAIEmbeddings()
            self.vector_store = None
            self.chat_model = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7
            )
            # Try to load existing vector store
            if os.path.exists("./chroma_db"):
                logger.info("Loading existing vector store...")
                self.vector_store = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=self.embeddings
                )
        except Exception as e:
            logger.error(f"Error initializing RAG agent: {str(e)}")
            raise
        
    def load_documents(self, directory_path):
        try:
            # Load documents from a directory
            logger.info(f"Loading documents from {directory_path}")
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Create or update vector store
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            logger.info("Documents loaded and indexed successfully")
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
        
    def format_chat_history(self, chat_history):
        formatted_history = []
        try:
            for message in chat_history:
                if isinstance(message, dict) and 'content' in message:
                    # If it's already in the right format, use it as is
                    formatted_history.append((message.get('content', ''), ''))
                else:
                    # Otherwise, try to convert it to the right format
                    formatted_history.append((str(message), ''))
        except Exception as e:
            logger.warning(f"Error formatting chat history: {str(e)}")
            # Return empty history if there's an error
            return []
        return formatted_history

    def chat(self, query, chat_history=[]):
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized")
                return "Please load documents first."
            
            formatted_history = self.format_chat_history(chat_history)
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.chat_model,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True,
                verbose=True
            )
            
            result = qa_chain({
                "question": query,
                "chat_history": formatted_history
            })
            return result["answer"]
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            raise

    def add_chunk(self, chunk: dict):
        """
        Append a single transcript/emotion chunk to the vector store.
        Expects output of generate_rag_json().
        """
        # Build a LangChain Document
        doc = Document(
            page_content=chunk["summary"],
            metadata={
                "id":        chunk["id"],
                "emotion":   chunk["emotion"],
                "source":    chunk["source"],
                "speaker":   chunk["speaker"],
                "timestamp": chunk["timestamp"],
            },
        )
        if self.vector_store is None:
            # first time – create a new DB
            self.vector_store = Chroma.from_documents(
                [doc], self.embeddings, persist_directory="./chroma_db"
            )
        else:
            self.vector_store.add_documents([doc])
        self.vector_store.persist()  # flush to disk

# Usage example
if __name__ == "__main__":
    agent = TherapyRAGAgent()
    agent.load_documents("./therapy_documents")
    response = agent.chat("How can I manage anxiety?")
    print(response)