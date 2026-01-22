# *SmartMedNotes*

*SmartMedNotes* is a *Generative AI-powered clinical documentation system* that answers orthopedic-related queries and automates clinical documentation using state-of-the-art AI techniques.

## *Features*

- *Frontend:* Built using *Vite + Tailwind CSS* for a fast and responsive UI.
- *Backend:* Developed using *Python Flask*, ensuring efficient query processing and model integration.
- *Model:* Fine-tuned *microsoft/phi-3-mini-4k-instruct* for generating accurate medical responses.
- *Embeddings:* Utilizes *sentence-transformers/all-MiniLM-L6-v2* for optimized information retrieval.
- *Technique:* Implements *Retrieval-Augmented Generation (RAG)* for improved contextual accuracy.
- *Database:* Uses *Firebase* for storing user queries and relevant data.
- *Deployment:*
  - *Frontend:* Hosted on *Vercel* for seamless accessibility.
  - *Backend:* Deployed on *Hugging Face* for reliable API handling.
## *Technologies*
-	*Python & Flask:* Used for backend API development and server-side logic.
-	*Hugging Face Spaces:* Platform for deploying the backend, including the fine-tuned model and retrieval system.
-	*FAISS (Facebook AI Similarity Search):* Utilized for efficient vector search and similarity retrieval.
-	*Microsoft/Phi-3-Mini-4K-Instruct:* Fine-tuned model for generating responses based on orthopedic queries.
-	*Sentence-Transformers/All-MiniLM-L6-V2:* Used for embedding generation in the retrieval process.
-	*Google-T5/T5-Small:* Used for summarizer.
-	*Firebase:* Used for user authentication and database storage.

## *Project Contributions*

### *Muhammad Abdullah Butt (2021-CS-81)*  🚀

- *Lead Developer & Architect* – Designed and implemented the *backend, model training, deployment, and Git management*.
- Fine-tuned *microsoft/phi-3-mini-4k-instruct* and optimized *RAG-based retrieval*.
- Deployed backend on *Hugging Face* and frontend on *Vercel*, managed *Firebase database*.
- Set up *Git repository* and ensured smooth integration of all components.

### *Chaudhry Ilyas (2021-CS-60)* 🎨

- Developed the *frontend* using *Vite + Tailwind CSS*.
- Designed an intuitive *UI* for better user experience.
- Integrated frontend with backend APIs.

### *Aqsa Mehmood (2021-CS-110)* 📖

- Handled *documentation* and project structuring.
- Assisted in *frontend improvements* and *minor backend debugging*.
- Helped in *data preprocessing* and ensuring consistency in dataset formatting.
- Contributed to *testing and evaluation* of the AI model.

### *Abdullah Afzal (2021-CS-80)* 🔹

- Contributed to *initial project setup* and *deployment testing*.
- Assisted in *frontend debugging* and making minor UI adjustments.
- Helped with *documentation formatting and review*.
- Provided feedback on overall system functionality and user experience.

## *Getting Started*

### *Installation*

bash
# Clone the repository
git clone https://github.com/MuhammadAbdullahButt/smartmednotes.git
cd smartmednotes


### *Backend Setup (Flask)*

bash
cd backend
pip install -r requirements.txt
python app.py


### *Frontend Setup (Vite + Tailwind CSS)*

bash
cd frontend
npm install
npm run dev

## *LIVE AT*
smart-med-notes-git-abdullah-muhammadabdullahbutts-projects.vercel.app
