import os
import sys
import oci
import json
import pandas as pd
import gradio as gr
import random
import time
import uuid
from loguru import logger
from gradio_pdf import PDF
import oracledb
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import oraclevs
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI
from langchain_upstage import UpstageLayoutAnalysisLoader
from langchain_text_splitters import (Language, RecursiveCharacterTextSplitter)
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage

# ======== parameter 
import configparser
config = configparser.ConfigParser()
config.read("oci.env")

username = config["DATABASE"]["USERNAME"]
password = config["DATABASE"]["PASSWORD"]
host = config["DATABASE"]["HOST"]
port = config["DATABASE"]["PORT"]
service_name = config["DATABASE"]["SERVICE_NAME"]
table_name = config["DATABASE"]["TABLE_NAME_CV_LANG"]
compartment_id = config["OCI"]["compartment_id"]
config_profile = config["OCI"]["CONFIG_PROFILE"]
endpoint = config["OCI"]["endpoint"]
dsn=host+":"+port+"/"+service_name
directory=config["ENVIRONMET"]["DIRECTORY"]
cohere_api_key=config["APIKEY"]["cohere"]
base64_image_oci_logo=config["LOGO"]["oci_logo"]
base64_image_isv_logo=config["LOGO"]["isv_logo"]
upstage_api_key=config["APIKEY"]["UPSTAGE_API_KEY"]
logger.debug("App to Use LLM on OCI Model Deployment")

os.environ["UPSTAGE_API_KEY"] = upstage_api_key

css = """
#warning {background-color: #FFCCCB}
.feedback textarea {font-size: 24px !important}
"""

#Connection Database
try:
    oracledb.init_oracle_client()
    connection = oracledb.connect(user=username, password=password, dsn=dsn)

    print("\nConnection successful!\n")
except Exception as e:
    print(e)
    print("\nConnection failed!\n")

#==== Oracle Langchain lib
# initial params
distance_strategy=DistanceStrategy.COSINE    #COSINE
table_name_with_strategy = table_name+'_'+distance_strategy+'_UPSTAGE'
print("table Name :",table_name_with_strategy)

vector_store = OracleVS(client=connection, embedding_function=UpstageEmbeddings(model="solar-embedding-1-large"), table_name=table_name_with_strategy)

with_rag = True
def select_rag_option(selected_option):
    global with_rag
    if selected_option == "With RAG":
        with_rag = True
    else:
        with_rag = False
    
def rag_embedding(pdf_file):
    print(pdf_file)

    s1time = time.time()
    
    chunks_with_mdata = []
    
    # UpstageLayoutAnalysisLoader - Start
    layzer = UpstageLayoutAnalysisLoader(pdf_file, split="page")
    
    # For improved memory efficiency, consider using the lazy_load method to load documents page by page.
    docs = layzer.load()  # or layzer.lazy_load()

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        chunk_size=500, chunk_overlap=100, language=Language.HTML
    )
    chunks_with_mdata = text_splitter.split_documents(docs)

    print(f"Number of docs loaded: {len(docs)}")

    for doc in chunks_with_mdata:
        doc.metadata['_file'] = pdf_file
    # UpstageLayoutAnalysisLoader - End
    
    #===count number of documents
    unique_files = set()
    
    for chunk in chunks_with_mdata:
        file_name = chunk.metadata['_file']
        unique_files.add(file_name)

    print("chunks_with_mdata:", chunks_with_mdata)

    vector_store = OracleVS.from_documents(chunks_with_mdata, UpstageEmbeddings(model="solar-embedding-1-large"), client=connection, table_name=table_name_with_strategy, distance_strategy=distance_strategy)

    if vector_store is not None:
        print("\n Documents loading, chunking and generating embeddings are complete.\n")
        result = "Documents loading, chunking and generating embeddings are complete."
    else:
        print("\nFailed to get the VectorStore populated.\n")
        result = "Failed to get the VectorStore populated."
        
    ### Create Oracle HNSW Index
    oraclevs.create_index(client=connection,vector_store=vector_store, params={
        "idx_name": "hnsw"+table_name_with_strategy, "idx_type": "HNSW"
    })

    s2time = time.time()
    print( f"Vectorizing and inserting chunks duration: {round(s2time - s1time, 1)} sec.")
    
    return result

with gr.Blocks(theme=gr.themes.Base(primary_hue="red", secondary_hue="pink")) as demo:
    with gr.Row():
        gr.HTML(
            f"""
            <div style="display: flex; align-items: center;">
                <img src="{base64_image_oci_logo}" alt="Logo" style="height: 40px; margin-right: 10px;">
                <h1>RAG 데모 with Oracle 23ai , Upstage </h1>
            </div>
            """
        )
    with gr.Row():
        with gr.Column(scale=1, visible=False) as sidebar_left:
            gr.Markdown("### OCI GenAI Option")
            with gr.Row():
                max_tokens = gr.Slider(
                    256,
                    4096,
                    value=256,
                    step=16,
                    label="max_tokens",
                    info="The maximum number of tokens",
                )

                temperature = gr.Slider(
                    0.2,
                    2.0,
                    value=1.0,
                    step=0.1,
                    label="temperature",
                    info="Controls randomness in the model. The value is used to modulate the next token probabilities.",
                )

                top_p = gr.Slider(
                    0.1,
                    1.0,
                    value=0.95,
                    step=0.1,
                    label="top_p",
                    info="Nucleus sampling, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
                )
        with gr.Column(scale=3,elem_id="component-0"):
            with gr.Row():
                chatbot = gr.Chatbot(
                    [],
                    show_label=False,
                    elem_id="chatbot",
                    height=330,
                    elem_classes="chatbot-container"
                )
            with gr.Row():
                    msg = gr.Textbox(
                        label="프롬프트:",
                        placeholder="저는 AI 어시스턴트입니다. 무엇이 궁금한가요?",
                        lines=2,
                        autofocus=True,
                        scale=3
                    )
                    with gr.Row():
                        send = gr.Button(value="Generate", variant="primary")
                        clear = gr.Button(value="Clear", variant="secondary")
            with gr.Row():
                examples = [
                    "도깨비에 대해서 설명해줘.",
                    "최초의 AI법은 언제 통과 되었니? 출처나 참고 문서도 같이 알려줘.",
                ]
                
                gr.Examples(
                    examples=examples,
                    inputs=msg,
                )
        with gr.Column(scale=1):
            with gr.Tab("RAG 옵션"):
                rag_option = gr.Radio(["With RAG", "Without RAG"], container=False, value="With RAG", interactive=True)
                rag_option.change(fn=select_rag_option, inputs=rag_option)
                
            with gr.Row():
                pdf = PDF(label="PDF 업로드", height=280, scale=1)
                pdf_file = gr.Textbox(visible=False)
            with gr.Row():
                out = gr.Textbox(value="", label="임베딩 상태", container=True, interactive=False, elem_classes="feedback")
                pdf.upload(lambda f: f, pdf, pdf_file)
                pdf_file.change(rag_embedding, pdf_file, out)
            #    pdf_files = gr.Dropdown(["전체문서", "문서1", "문서2", "문서3"]),

    def remove_newline(text):
        return text.replace('\n', '')
        
    def user(user_message, history):
        return ("", history + [[user_message, None]])
        
    def format_docs(docs):
        return '\n\n'.join([d.page_content for d in docs])

    def groundedness_message_html(groundedness_check_response):
        print("groundedness_check_response:", groundedness_check_response)
        
        if groundedness_check_response == "grounded":
            message = "통과하였습니다."
            color = "green"
        else:
            message = "통과하지 못했습니다."
            color = "red"
            
        return f"""
            <div style="text-align: center;">
            <p><b>-<b> LLM이 생성한 답변은 <b>Upstage</b>의 <b>Groundedness Check</b>를 <span style='color: {color};'>{message}</span> <b>-</b>
            </div>
        """
        
    def response_from_oraclevs(query):

        # "solar-1-mini-chat"
        chat = ChatUpstage()

        # s1time = time.time()
        # print(vector_store.similarity_search(query, 3))
        # s2time = time.time()
        
        # print(f"Search for the user question in the Oracle Database 23ai and return similar chunks duration: {round(s2time - s1time, 1)} sec.")
        
        if with_rag:
            message = [
                (
                    "system",
                    """
                    질문-답변 업무를 돕는 AI 어시스턴트입니다. 
                    문서의 내용을 참고해서 답변해 주세요.:
                    \n\n
                    {context}",
                    """
                ),
                ("human", "{human}"),
            ]
        else:
            message = [
                (
                    "system",
                    """
                    you are an AI assistant.",
                    """
                ),
                ("human", "{human}"),
            ]
        
        prompt = ChatPromptTemplate.from_messages(message)
        
        chain = {
            "context": vector_store.as_retriever() if with_rag else RunnablePassthrough(),
            "human": RunnablePassthrough(),
        } | prompt | chat | StrOutputParser()

        response = chain.invoke(query)

        # Groundedness Check
        if with_rag:
            user_content = format_docs(vector_store.similarity_search(query))
            print("user_content:", user_content)
        else:
            user_content = query

        
        
        upstage_client = OpenAI(
            api_key=upstage_api_key,
            base_url="https://api.upstage.ai/v1/solar"
        )
        
        groundedness_check_response = upstage_client.chat.completions.create(
            model="solar-1-mini-groundedness-check",
            messages=[
                {
                  "role": "user",
                  "content": user_content
                },
                {
                  "role": "assistant",
                  "content": response
                }
            ]
        )
        
        groundedness_message = groundedness_message_html(groundedness_check_response.choices[0].message.content)
        
        return response + groundedness_message
    
    def bot(history, max_tokens, temperature, top_p):
        query = history[-1][0]

        response = response_from_oraclevs(query)
        
        # llm_response = response
        llm_response = response
        
        if history and history[-1][1] is None:
            history[-1][1] = ""  # history[-1][1]이 None일 경우 빈 문자열로 초기화
        
        for character in llm_response:
            history[-1][1] += character
            time.sleep(0.03)
            yield history
        
    send.click(
        user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    ).then(bot, [chatbot, max_tokens, temperature, top_p], chatbot)

    clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    # demo.launch(share=True)
    demo.launch(server_port=7870,server_name="0.0.0.0")