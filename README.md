# Cyberlawbook ⚖️

**Cyberlawbook** is a Retrieval-Augmented Generation (RAG) Streamlit application designed to answer legal queries strictly adhering to Pakistan's **Prevention of Electronic Crimes Act (PECA 2016)**.

---

## 🚀 Features
- **Automatic Setup:** Automatically downloads official PECA PDF documentation from the National Assembly of Pakistan portal and indexes it into a FAISS vector database on startup.
- **Customizable UI Controls:**
  - **Technicality Level:** Simple (Layman), Balanced, or Legal Expert.
  - **Response Length:** Concise vs Detailed summaries.
  - **Penalties Toggle:** Focuses on fines, section penalties, and imprisonment terms.
- **Source Inspection:** Displays precise retrieved legal context chunks alongside page references.

---

## 🛠️ Run locally or on Streamlit Cloud

1. **Clone & Install Dependencies:**
   ```bash
   pip install -r requirements.txt
