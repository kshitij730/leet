from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse, RedirectResponse
from typing import List, Optional
import tempfile, os, asyncio, io
import markdown
import stripe
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from PyPDF2 import PdfReader, PdfWriter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import bleach
from concurrent.futures import ThreadPoolExecutor
from groq import Groq
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from datetime import datetime, timedelta

app = FastAPI()

# Stripe setup
stripe.api_key = "sk_test_51RgrXWPYPeTVsOaztmyC644IRmtEJ19KjlyE6yph33bOU9xU7dKPGxlDOcICgAO9OP7LDGhhW98DjrRAat9pKyJD00AsUbKrXF"
YOUR_DOMAIN = "http://www.pdfcompare.absoluteintel.xyz"
STRIPE_WEBHOOK_SECRET = "whsec_DiGnsg34MJJ9T1Rb6REv05H8uVhxWGDr"

# Stripe email access control
TEST_EMAIL = "test@example.com"
FREE_MAX_SIZE_MB = 2
PREMIUM_MAX_SIZE_MB = 5
user_payment_status = {}

groq_client = Groq(api_key="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")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
combined_vectorstore = None

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=20000,
    chunk_overlap=2000,
    separators=["\n\n", "\n", ".", "!", "?"]
)

# === Prompt Templates ===
SUMMARY_PROMPT = PromptTemplate.from_template("""
You are a highly accurate and detail-oriented document summarization assistant.

Summarize the following document clearly and concisely, capturing:
- First: A brief and simple **overview** of what the document is about.
- Then: A structured list of **key highlights**, focusing on most important points, findings or facts in a well organized structured way keep in mind the spacing and all also using subpoints do not integrate everything in single points and keep more data oreiented.

Keep the summary short, simple, easy to understand(it should be user friendly easy to understand and short), and well-organized with proper spacing and subpoints (do not merge all content into a single block).

**Guidelines**:
- Focus solely on high-value, verifiable content.
- Omit background, narrative, or commentary.
- Use precise, specific phrasing.
- Do not generalize or summarize beyond what’s present in the document.

Do not keep output in markdown format.                                                                                            
Document:
{context}
""")

COMPARISON_PROMPT = PromptTemplate.from_template("""
You are an expert in comparative document analysis.

Your task is to compare two documents in a clear, structured manner. Focus on extracting and contrasting the most relevant aspects of each document in a well organized structured way keep in mind the spacing and all.

Present the comparison using a table with clear headings.

Below are the summaries and highlights for reference:

**{filename1} – Summary**:
{summary1}

**{filename2} – Summary**:
{summary2}
                                                 
Maintain a professional tone, be concise, and ensure the comparison is objective and insightful.
                                                                                                
Do not make it too long make it short and simple to understand.
                                                 
Do not keep output in markdown format.                                                 
""")

# === Stripe Webhook ===

@app.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid signature: {e}")

    event_type = event["type"]
    data = event["data"]["object"]
    print(f"🔔 Received event: {event_type}")

    if event_type == "checkout.session.completed":
        email = data.get("metadata", {}).get("email")
        if email:
            expiration = datetime.utcnow() + timedelta(days=30)
            user_payment_status[email] = expiration
            print(f"✅ Subscription for {email} active until {expiration}")
    elif event_type == "payment_intent.payment_failed":
        print(f"❌ Payment failed: {data['last_payment_error']['message']}")
    elif event_type == "charge.refunded":
        print(f"💸 Refund issued for charge {data['id']}")
    elif event_type == "customer.subscription.deleted":
        print(f"🛑 Subscription cancelled: {data['customer']}")
    else:
        print(f"📦 Unhandled event type: {event_type}")

    return {"status": "success"}

# These should be your actual Stripe price IDs
MONTH_PRICE_ID = "price_1Ro29BPYPeTVsOazN8RGpKx9"  # <-- replace this with your real monthly price id
YEAR_PRICE_ID = "price_1Ro29BPYPeTVsOazZ8I9y57W"    # <-- replace this with your real yearly price id

@app.post("/create-checkout-session")
async def create_checkout_session(
    email: str = Form(...),
    plan: str = Form(...)  # either 'month' or 'year'
):
    try:
        # Pick the right price ID
        if plan == "month":
            price_id = MONTH_PRICE_ID
        elif plan == "year":
            price_id = YEAR_PRICE_ID
        else:
            raise HTTPException(status_code=400, detail="Invalid plan selection.")

        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            line_items=[{
                "price": price_id,
                "quantity": 1,
            }],
            success_url=f"{YOUR_DOMAIN}/success?email={email}",
            cancel_url=f"{YOUR_DOMAIN}/cancel?email={email}",
            metadata={"email": email, "plan": plan},
        )
        return RedirectResponse(session.url, status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def split_pdf_by_pages(path: str, batch_size: int = 100) -> List[str]:
    reader = PdfReader(path)
    total_pages = len(reader.pages)
    output_paths = []

    for i in range(0, total_pages, batch_size):
        writer = PdfWriter()
        for j in range(i, min(i + batch_size, total_pages)):
            writer.add_page(reader.pages[j])
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        with open(temp_file.name, "wb") as f:
            writer.write(f)
        output_paths.append(temp_file.name)

    return output_paths

async def process_pdf_chunks_in_batches(path: str) -> List[Document]:
    page_batches = split_pdf_by_pages(path)
    results = await asyncio.gather(*[
        asyncio.to_thread(PyPDFLoader(batch).load) for batch in page_batches
    ])

    os.unlink(path)
    for batch in page_batches:
        os.remove(batch)
    all_docs = [doc for batch_docs in results for doc in batch_docs if doc.page_content.strip()]
    return text_splitter.split_documents(all_docs)


def combine_chunks(chunks: List[Document], group_size: int = 5) -> List[Document]:
    def combine_group(group):
        group_text = "\n".join(chunk.page_content for chunk in group if chunk.page_content.strip())
        return Document(page_content=group_text) if group_text else None

    with ThreadPoolExecutor() as executor:
        grouped = [chunks[i:i+group_size] for i in range(0, len(chunks), group_size)]
        results = list(executor.map(combine_group, grouped))

    return [doc for doc in results if doc]

def cluster_chunk_group(group: List[Document], n_clusters: int = 3) -> List[Document]:
    texts = [doc.page_content for doc in group if doc.page_content.strip()]
    if len(texts) <= n_clusters:
        return [Document(page_content=txt) for txt in texts]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    clustered = {}
    for label, text in zip(labels, texts):
        clustered.setdefault(label, []).append(text)
    return [Document(page_content="\n".join(docs)) for docs in clustered.values() if docs]


async def embed_chunks(chunks: List[Document]):
    return await asyncio.to_thread(lambda: FAISS.from_documents(chunks, embedding=embedding_model))


async def call_groq(prompt: str) -> str:
    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1500,
        top_p=0.95,
        stream=False
    )
    return completion.choices[0].message.content


async def process_chunks(chunks: List[Document], prompt: PromptTemplate, group_size: int = 5, clusters_per_group: int = 3):
    grouped_chunks = combine_chunks(chunks, group_size)

    async def cluster_and_summarize(group: Document):
        try:
            clustered = await asyncio.to_thread(cluster_chunk_group, [group], clusters_per_group)

            async def summarize_cluster(cluster_doc: Document):
                return await call_groq(prompt.format(context=cluster_doc.page_content))

            summaries = await asyncio.gather(*[summarize_cluster(cd) for cd in clustered])
            return "\n".join(summaries)
        except Exception as e:
            return f"[Error: {str(e)}]"

    results = await asyncio.gather(*[cluster_and_summarize(group) for group in grouped_chunks])
    return "\n".join(results)




async def download_google_drive_pdf(file_id: str, access_token: str) -> str:
    creds = Credentials(token=access_token)
    service = build("drive", "v3", credentials=creds)
    request = service.files().get_media(fileId=file_id)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    with open(temp_file.name, "wb") as f:
        f.write(fh.getbuffer())
    return temp_file.name


# === Compare Route ===
@app.post("/compare-docs-rag", response_class=HTMLResponse)
async def compare_docs(
    file1: UploadFile = File(None),
    file2: UploadFile = File(None),
    file1_id: str = Form(None),
    file2_id: str = Form(None),
    access_token: str = Form(None),
    email: Optional[str] = Form(None),  # email is optional for free uploads
):
    global combined_vectorstore

    # Save files temporarily and check sizes (in MB)
    if not file1 or not file2:
        raise HTTPException(status_code=400, detail="Both files required.")

    file1_bytes = await file1.read()
    file2_bytes = await file2.read()
    size1_mb = len(file1_bytes) / (1024 * 1024)
    size2_mb = len(file2_bytes) / (1024 * 1024)

    # Determine user status
    is_premium = False
    if email:
        expiration = user_payment_status.get(email)
        if email == TEST_EMAIL or (expiration and expiration > datetime.utcnow()):
            is_premium = True

    # Free users can upload files up to 2MB without email
    # If any file > 2MB, email and premium required for larger uploads
    if size1_mb > FREE_MAX_SIZE_MB or size2_mb > FREE_MAX_SIZE_MB:
        if not (email and is_premium):
            # Show upgrade with email required
            return HTMLResponse(
    content=f"""
<div id="upgrade-overlay"
     role="dialog"
     aria-modal="true"
     aria-labelledby="upgrade-title"
     aria-describedby="upgrade-desc"
     style="
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 9999;
        background: rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        display: flex;
        align-items: center;
        justify-content: center;
     ">
  <div id="upgrade-modal"
       style="
        background: #ffffff !important;
        border-radius: 18px;
        width: 92%;
        max-width: 400px;
        padding: 32px 28px;
        box-shadow: 0 12px 36px rgba(50, 63, 93, 0.3);
        border: 2px solid #e2e8f0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        display: flex;
        flex-direction: column;
        gap: 22px;
        opacity: 1 !important;
        z-index: 10000;
       ">

    <style>
      #button-container {{
        display: flex;
        gap: 16px;
      }}
      #upgrade-modal button[type="submit"] {{
        border: none !important;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 700;
        cursor: pointer;
        color: #fff !important;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 14px 0;
        transition: 
          background 0.25s,
          box-shadow 0.25s;
        flex: 1;
        box-shadow: 0 4px 18px rgba(53,112,247,0.13);
        background-clip: padding-box;
      }}
      #upgrade-modal button[value="month"] {{
        background: linear-gradient(90deg, #41e7b8 0%, #3570f7 100%) !important;
      }}
      #upgrade-modal button[value="month"]:hover {{
        background: linear-gradient(90deg, #21bfa4 0%, #2359d4 100%) !important;
        box-shadow: 0 8px 32px rgba(53,112,247,0.29);
      }}
      #upgrade-modal button[value="year"] {{
        background: linear-gradient(90deg, #ffcb52 0%, #fc3c6a 100%) !important;
      }}
      #upgrade-modal button[value="year"]:hover {{
        background: linear-gradient(90deg, #f7b956 0%, #ea3a7c 100%) !important;
        box-shadow: 0 8px 32px rgba(252,60,106,0.29);
      }}
      #upgrade-modal button small {{
        font-weight: 400;
        font-size: 0.90rem;
        opacity: 0.9;
        margin-top: 3px;
        line-height: 1.2;
      }}
      #upgrade-email {{
        width: 100%;
        box-sizing: border-box; /* Ensure padding and border are inside width */
        padding: 14px 14px;
        font-size: 1rem;
        border-radius: 10px;
        border: 2px solid #7a92ff;
        transition: border-color 0.3s ease;
        margin-bottom: 0;
        margin-top: 0;
        background: #fff;
      }}
      #upgrade-email:focus {{
        border-color: #3570f7;
        outline: none;
        box-shadow: 0 0 8px #3570f7a8;
      }}
      #upgrade-overlay a {{
        user-select: none;
        text-decoration: underline;
      }}
      #upgrade-overlay a:hover {{
        color: #1a4edb;
      }}
    </style>

    <h2 id="upgrade-title"
        style="
          font-size: 1.45rem;
          color: #d43e56;
          font-weight: 800;
          text-align: center;
          margin: 0;
        ">
      🚫 File Size Limit Exceeded
    </h2>

    <p id="upgrade-desc"
       style="
         font-size: 1.05rem;
         color: #333;
         font-weight: 500;
         line-height: 1.5;
         text-align: center;
         margin: 0;
       ">
      Your free limit is <strong style="color: #3570f7;">{FREE_MAX_SIZE_MB}</strong> MB per file.<br/>
      Please upgrade to premium for higher limits.
    </p>

    <form method="POST" action="/create-checkout-session" novalidate
          style="display: flex; flex-direction: column; gap: 18px;">
      <input id="upgrade-email"
             type="email"
             name="email"
             placeholder="you@example.com"
             required
             autocomplete="email"
             aria-required="true" />

      <div id="button-container">
        <button type="submit"
                name="plan"
                value="month"
                aria-label="Subscribe monthly plan for $5">
          💳 Monthly
          <small>$5.00 / month</small>
        </button>
        <button type="submit"
                name="plan"
                value="year"
                aria-label="Subscribe yearly plan for $50">
          💎 Yearly
          <small>$50.00 / year</small>
        </button>
      </div>

      <a href="/"
         style="
           display: block;
           text-align: center;
           margin-top: 4px;
           font-weight: 600;
           color: #3570f7;
           font-size: 0.95em;
           user-select: none;
         ">
        🔙 Back
      </a>
    </form>
  </div> <!-- upgrade-modal -->
</div> <!-- upgrade-overlay -->
""",
    status_code=402,
)


        # if user is premium check max premium size limit
        if size1_mb > PREMIUM_MAX_SIZE_MB or size2_mb > PREMIUM_MAX_SIZE_MB:
            return HTMLResponse(
                content=f"""
                <html>
                <body style="text-align:center; font-family:sans-serif;">
                    <h2>🚫 File Too Large</h2>
                    <p>Your premium upload limit is {PREMIUM_MAX_SIZE_MB} MB per file.</p>
                    <p>Please reduce file size.</p>
                    <p style="margin-top:1rem;"><a href="/">🔙 Back</a></p>
                </body>
                </html>
                """,
                status_code=413
            )
    # else (files under 2MB): no email or payment needed, proceed free

    # Save temp files again with actual content (resetting read pointer for file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp1:
        tmp1.write(file1_bytes)
        path1 = tmp1.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
        tmp2.write(file2_bytes)
        path2 = tmp2.name

    # Process docs as before...
    chunks1, chunks2 = await asyncio.gather(
    process_pdf_chunks_in_batches(path1),
    process_pdf_chunks_in_batches(path2)
    )

    combined_vectorstore = await embed_chunks(chunks1 + chunks2)


    combined_chunks1 = combine_chunks(chunks1)
    combined_chunks2 = combine_chunks(chunks2)

    summary1, summary2 = await asyncio.gather(
        process_chunks(combined_chunks1, SUMMARY_PROMPT, group_size=15, clusters_per_group=2),
        process_chunks(combined_chunks2, SUMMARY_PROMPT, group_size=15, clusters_per_group=2)
    )


    comparison_prompt = COMPARISON_PROMPT.format(
        filename1=getattr(file1, "filename", "Document 1"),
        filename2=getattr(file2, "filename", "Document 2"),
        summary1=summary1,
        summary2=summary2,
    )

    comparison_result = await call_groq(comparison_prompt)

    output_html = f"""
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        h3 {{
            color: #333;
            border-bottom: 2px solid #ccc;
            padding-bottom: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
        }}
        ul, ol {{
            padding-left: 20px;
        }}
    </style>
</head>
<body>
    <h3>Summary of {sanitize_output(getattr(file1, "filename", "Document 1"))}</h3><div>{sanitize_output(summary1)}</div>
    <h3>Summary of {sanitize_output(getattr(file2, "filename", "Document 2"))}</h3><div>{sanitize_output(summary2)}</div>
    <h3>Comparison</h3><div>{sanitize_output(comparison_result)}</div>
</body>
</html>"""

    return HTMLResponse(content=output_html)

@app.get("/success", response_class=HTMLResponse)
async def success(email: str):
    expiration = datetime.utcnow() + timedelta(days=30)
    user_payment_status[email] = expiration
    return f"""
    <html>
    <head>
        <meta http-equiv="refresh" content="1; url=/" />
    </head>
    <body style="text-align:center; font-family:sans-serif; padding:2rem;">
        <h2>✅ Payment Successful</h2>
        <p>Redirecting you back to the app...</p>
    </body>
    </html>
    """

@app.get("/cancel", response_class=HTMLResponse)
async def cancel(email: str = ""):
    return f"""
    <html>
    <body style="text-align:center; padding:2rem;">
        <h2>❌ Payment Cancelled</h2>
        <p>Your payment was not completed.</p>
        <a href="/">🔁 Try Again</a>
    </body>
    </html>
    """

@app.post("/chat", response_class=PlainTextResponse)
async def chat_with_combined_docs(query: str = Form(...)):
    global combined_vectorstore
    if not combined_vectorstore:
        return PlainTextResponse("Documents not uploaded yet.", status_code=400)

    docs = combined_vectorstore.similarity_search(query, k=5)
    context = "\n".join(doc.page_content for doc in docs)
    prompt = f"Answer the following question using the provided context in a well organized structured way keep in mind the spacing and also using subpoints do not integrate everything in single points. Do not make it too long make it short and simple to understand.:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = await call_groq(prompt)
    return StreamingResponse(iter([sanitize_output(response)]), media_type="text/plain")

@app.get("/", response_class=HTMLResponse)
async def upload_ui():
    return open("ui.html", encoding="utf-8").read()

def sanitize_output(text: str) -> str:
    if any(symbol in text for symbol in ["#", "*", "-", "`", "[", "]"]):  # crude Markdown check
        html = markdown.markdown(text, extensions=["extra", "tables", "sane_lists"])
        return bleach.clean(html, tags=["p", "strong", "em", "ul", "ol", "li", "table", "tr", "td", "th", "h1", "h2", "h3", "br"], strip=True)
    else:
        return bleach.clean(text, tags=["br"], strip=True)









