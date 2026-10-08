import os
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # Local free embeddings
from langchain_openai import ChatOpenAI                    # Compatible with Groq API endpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

class RAGEngine:
    def __init__(self):
        # Use a free, lightweight open-source embedding model that runs locally
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.persist_directory = "./chroma_db"
        self.retriever = self._initialize_vector_store()

    def _ensure_knowledge_base_exists(self, kb_file_path: str):
        os.makedirs(os.path.dirname(kb_file_path), exist_ok=True)
        default_guidelines = (
            "ATS RULE 1: Bullet points should follow the X-Y-Z formula: 'Achieved [X], measured by [Y], by doing [Z]'. "
            "Quantitative metrics (e.g., increased revenue by 20%, reduced latency by 35ms) significantly improve ATS scores.\n"
            "ATS RULE 2: Avoid multi-column layouts, graphics, or tables as they disrupt standard ATS text extraction parsers. "
            "Use standard headings like 'Work Experience', 'Education', and 'Skills'.\n"
            "ATS RULE 3: Hard technical skills (e.g., Python, Docker, Kubernetes, AWS, React) must explicitly match the terminology "
            "used in the target job description. Using generic phrasing or synonyms can lower keyword match rates.\n"
            "ATS RULE 4: Action verbs should start every single achievement bullet point. Examples include 'Architected', "
            "'Engineered', 'Optimized', 'Spearheaded', rather than passive phrases like 'Responsible for' or 'Worked on'.\n"
            "ATS RULE 5: Keep contact information clean at the top. Do not include photos, personal dates of birth, or home addresses."
        )

        if not os.path.exists(kb_file_path) or os.path.getsize(kb_file_path) == 0:
            with open(kb_file_path, "w", encoding="utf-8") as f:
                f.write(default_guidelines)
            print(f"-> Automatically populated knowledge base file at: {kb_file_path}")

    def _initialize_vector_store(self):
        kb_file_path = "./knowledge_base/ats_guidelines.txt"
        self._ensure_knowledge_base_exists(kb_file_path)

        # Check if ChromaDB folder already exists and has data saved
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print("-> Loading existing vector store from local disk (Instant Startup)...")
            vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            return vector_store.as_retriever(search_kwargs={"k": 3})

        print("-> First time setup: Processing and embedding dataset (This will take a few seconds)...")
        kb_docs = []
        if os.path.exists(kb_file_path):
            loader = TextLoader(kb_file_path, encoding="utf-8")
            kb_docs = loader.load()

        dataset_docs = []
        csv_path = "./Resume.csv"
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.head(50.5).head(100).iterrows(): # Keep 100 rows
                    doc = Document(
                        page_content=str(row["Resume_str"]),
                        metadata={"category": str(row["Category"]), "id": str(row["ID"])}
                    )
                    dataset_docs.append(doc)
            except Exception as e:
                print(f"Warning: Could not load Resume.csv: {e}")

        all_docs = kb_docs + dataset_docs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunked_docs = text_splitter.split_documents(all_docs)

        # Create and persist vector database to disk
        vector_store = Chroma.from_documents(chunked_docs, self.embeddings, persist_directory=self.persist_directory)
        return vector_store.as_retriever(search_kwargs={"k": 3})
    def analyze_with_rag(self, resume_text: str, job_description: str) -> str:
        query = f"Target Role requirements: {job_description}. Candidate Profile: {resume_text[:400]}"
        retrieved_docs = self.retriever.invoke(query)
        
        guidelines_context = []
        resume_examples = []
        
        for doc in retrieved_docs:
            if "category" in doc.metadata:
                resume_examples.append(f"[{doc.metadata['category']}] Example: {doc.page_content[:500]}...")
            else:
                guidelines_context.append(doc.page_content)

        rag_context = "\n---\n".join(guidelines_context + resume_examples)

        llm = ChatOpenAI(
            model="openai/gpt-oss-20b",  # Or whichever model ID works on your Groq key
            openai_api_key=os.getenv("GROQ_API_KEY"),
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=0.2
        )

        # Updated System prompt forcing the model to explicitly state the score first
        system_prompt = (
            "You are an expert technical recruiter and ATS optimization engine. "
            "Analyze the candidate's uploaded resume against the target job description. "
            "Ground your evaluation using the expert ATS rules and professional dataset resume examples provided below:\n\n"
            f"--- DATASET & GUIDELINE RAG CONTEXT ---\n{rag_context}\n---------------------------------------\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. You must calculate an honest, realistic Estimated ATS Match Score (from 0% to 100%) based strictly on skill and keyword alignment.\n"
            "2. Start your response with the exact line: 'FINAL_SCORE: [X]%' (e.g., FINAL_SCORE: 18%).\n"
            "3. Follow it with detailed markdown formatting covering missing skills, rewriting suggestions, and a breakdown table."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Target Job Description:\n{job_description}\n\nCandidate Resume Text:\n{resume_text}")
        ])

        chain = prompt | llm
        ai_response = chain.invoke({
            "job_description": job_description,
            "resume_text": resume_text
        })

        content = ai_response.content

        # Extract the score dynamically to build a clean report header
        score_display = "Estimated ATS Match Score: Calculated by AI"
        if "FINAL_SCORE:" in content:
            # Safely parse out the score line generated by the LLM
            lines = content.split("\n")
            for line in lines:
                if "FINAL_SCORE:" in line:
                    score_value = line.replace("FINAL_SCORE:", "").strip()
                    score_display = f"Estimated ATS Match Score\n{score_value}"
                    # Optional: remove the raw tag line if you want cleaner output
                    content = content.replace(line, "")

        # Return the dynamically formatted final output
        final_report = f"""
# 📊 ATS Evaluation Report
--------------------------------------------------
### {score_display}
--------------------------------------------------

{content.strip()}
"""
        return final_report

rag_engine = RAGEngine()