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
from langchain_community.document_loaders import oracleai
from langchain_community.utilities.oracleai import OracleSummary
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.embeddings.oracleai import OracleEmbeddings
from langchain_community.document_loaders.oracleai import OracleTextSplitter 
from langchain_community.document_loaders.oracleai import OracleDocLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI

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
proxy = ''
splitter_params = {"BY" :"words", "MAX": 200, "OVERLAP": 10, "SPLIT": "sentence", "LANGUAGE": "KOREAN", "NORMALIZE": "all"}
embedder_params = {"provider": "ocigenai", 
                   "credential_name": "YH_OCI_CRED", 
                   "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText", 
                   "model": "cohere.embed-multilingual-v3.0"}
splitter = OracleTextSplitter(conn=connection, params=splitter_params)
embedder = OCIGenAIEmbeddings(
            model_id="cohere.embed-multilingual-v3.0",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id= compartment_id)

distance_strategy=DistanceStrategy.COSINE    #COSINE
table_name_with_strategy = table_name+'_'+distance_strategy
print("table Name :",table_name_with_strategy)

# Vector Store Initialization
vector_store = OracleVS(client=connection, embedding_function=embedder, table_name=table_name_with_strategy)

#==== OCI Gen AI Model List
model_list = [
    {"name":"cohere.command-r-16k","mode_id":"ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyawk6mgunzodenakhkuwxanvt6wo3jcpf72ln52dymk4wq"},
    {"name":"cohere.command-r-plus","mode_id":"ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya7ozidbukxwtun4ocm4ngco2jukoaht5mygpgr6gq2lgq"}
    #,{"name":"meta.llama-3-70b-instruct","mode_id":"ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyaycmwwnvu2gaqrffquofgmshlqzcdwpk727n4cykg34oa"}
]

genai_model_id = "cohere.command-r-16k" #cohere.command-r-16k
def select_model_option(selected_model):
    global genai_model_id
    genai_model_id = selected_model
    
with_rag = True
def select_rag_option(selected_option):
    global with_rag
    if selected_option == "With RAG":
        with_rag = True
    else:
        with_rag = False

retriever = "VectorStore"
def select_retriever_option(selected_option):

    global retriever
    retriever = selected_option
    
    if selected_option == "SQL":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
        
def toggle_sidebar(state):
    state = not state
    return gr.update(visible = state), state

def enable_send_button(msg):
    # 버튼을 활성화할지 비활성화할지 결정
    if msg.strip():
        return gr.update(interactive=True)  # 문자가 있으면 버튼 활성화
    else:
        return gr.update(interactive=False)  # 문자가 없으면 버튼 비활성화

def rag_embedding(pdf_file):
    print(pdf_file)

    s1time = time.time()
    
    chunks_with_mdata = []
    doc_origin = Document
    max_lengh_oracle_allow=9000
    counter = 0  
    document_num = 0
    
    # Oracle DocsLoader - Start
    loader_params = {}        
    loader_params['file'] = pdf_file
    # instantiate loader, splitter and embedder
    loader = OracleDocLoader(conn=connection, params=loader_params)
    
    # read the docs, convert blob docs to clob docs
    docs = loader.load()
    print(f"Number of docs loaded: {len(docs)}")

    for id, doc in enumerate(docs, start=1):
        #remove line break from the text document
        doc.page_content = doc.page_content.replace("\n", "")
        doc_origin.page_content = doc.page_content
        # check the doc
        if len(doc.page_content)>max_lengh_oracle_allow :
            #reduce the text to max_lengh_oracle_allow
            doc.page_content = doc.page_content[:9000]
        # get the summary
        # summ = summary.get_summary(doc) # total number of tokens exceeded by 2587.
        document_num += 1
        
        # chunk the doc
        chunks = splitter.split_text(doc_origin.page_content)
        print(f"Doc {id}: chunks# {len(chunks)}")

    ids = []
    #For each chunk create chunk_metadata with 
    for ic, chunk in enumerate(chunks, start=1):
        counter += 1  
        chunk_metadata = doc.metadata.copy()  
        chunk_metadata['id'] = str(counter)  
        chunk_metadata['document_id'] = str(document_num)
        # chunk_metadata['document_summary'] = str(summ[0])
        chunks_with_mdata.append(Document(page_content=str(chunk), metadata=chunk_metadata))
        ids.append(str(uuid.uuid4()))
        print(f"Doc {id}: metadata: {doc.metadata}")
    # Oracle DocsLoader - End
    
    #===count number of documents
    unique_files = set()
    
    for chunk in chunks_with_mdata:
        file_name = chunk.metadata['_file']
        unique_files.add(file_name)

    print("chunks_with_mdata:", chunks_with_mdata)

    vector_store = OracleVS.from_documents(chunks_with_mdata, embedder, client=connection, table_name=table_name_with_strategy, distance_strategy=distance_strategy)
    
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
                <h1>RAG 데모 with Oracle 23ai , OCI Gen AI </h1>
            </div>
            """
        )
    with gr.Row(): 
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
                        placeholder="저는 OCI GenAI 어시스턴트입니다. 무엇이 궁금한가요?",
                        lines=2,
                        autofocus=True,
                        scale=3
                    )
                    with gr.Row():
                        send = gr.Button(value="Generate", variant="primary", interactive=False)
                        clear = gr.Button(value="Clear", variant="secondary")
                    msg.change(fn=enable_send_button, inputs=msg, outputs=send)
                
            with gr.Row():
                examples = [
                    "도깨비에 대해서 설명해줘.",
                    "최초의 AI법은 언제 통과 되었니? 출처나 참고 문서도 같이 알려줘.",
                    "자동차가 침수되었는데 보험금을 수령할수 있나요?"
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
            
            with gr.Tab("Gen AI 모델"):
                model_options = {model['name']: model['mode_id'] for model in model_list}
                selected_model = gr.Dropdown(label="Model 선택", container=False, choices=list(model_options.keys()), value="cohere.command-r-16k")
                selected_model.change(fn=select_model_option, inputs=selected_model)
                with gr.Row():
                    max_tokens = gr.Slider(
                        256,
                        4096,
                        value=500,
                        step=16,
                        label="max_tokens",
                        info="The maximum number of tokens",
                    )
                    
                    temperature = gr.Slider(
                        0.2,
                        2.0,
                        value=0.7,
                        step=0.1,
                        label="temperature",
                        info="Controls randomness in the model. The value is used to modulate the next token probabilities.",
                    )
    
                    top_p = gr.Slider(
                        0.1,
                        1.0,
                        value=0.6,
                        step=0.1,
                        label="top_p",
                        info="Nucleus sampling, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
                    )

            with gr.Tab("검색기"):
                retriever_option = gr.Radio(["VectorStore", "SQL"], container=False, value="VectorStore", interactive=True)
                with gr.Row():
                    top_k = gr.Slider(
                        1,
                        100,
                        value=3,
                        step=1,
                        label="top_k",
                        info="",
                    )
                with gr.Row():
                    code_input = gr.Code(language="sql", lines=10, label="Enter SQL", visible=False, interactive=True)

                retriever_option.change(fn=select_retriever_option, inputs=retriever_option, outputs=code_input)
                
    def remove_newline(text):
        return text.replace('\n', '')
        
    def user(user_message, history):
        return ("", history + [[user_message, None]])
    
    def format_docs(docs):
        return '\n\n'.join([d.page_content for d in docs])

    def response_from_oraclevs(query, max_tokens, temperature, top_p, top_k):
        print("genai_model_id", genai_model_id)
        chat = ChatOCIGenAI(
            model_id=genai_model_id,
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id=compartment_id,
            model_kwargs={"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
        )

        # vector_store = OracleVS(client=connection, embedding_function=embedder, table_name=table_name_with_strategy)
        # print(vector_store.similarity_search(query, 3))
        
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
            "context": vector_store.as_retriever(search_kwargs={'k':top_k}) if with_rag else RunnablePassthrough(),
            "human": RunnablePassthrough(),
        } | prompt | chat | StrOutputParser()

        response = chain.invoke(query)
        return response

    def response_from_oraclesql(query, max_tokens, temperature, top_p, top_k, code_input):

        if code_input == "":
            return ""
            
        sql_query = f"""
           {code_input.format(table_name_with_strategy=table_name_with_strategy, top_k=top_k)}
        """

        print("sql_query:", sql_query)

        context = ""

        if with_rag:
            cur = connection.cursor()
            cur.execute(sql_query, {'query': query, 'embedder_params': json.dumps(embedder_params)})
            rows = cur.fetchall()

            # 보험 약관 관련 시나리오에서 사용하기 위해 두 가지 형태로 구성 (보험 약관의 경우 문자열 반환으로 read() 함수 사용 불가)
            documents = [Document(page_content=row[0]) if isinstance(row[0], str) else Document(page_content=row[0].read(), metadata=json.loads(row[1].read())) for row in rows]

            if "보험" in query:
                system_message_for_insurance_scenario = """보험과 관련된 질문인 경우에만 검색된 정보와 고객정보, 그리고 다음 내용을 참조하여 질문에 답변해 주세요.:
                - 참조 문서: 만약 답변에 특정 문서의 정보를 사용했다면, 해당 문서의 번호와 청크 번호를 명시해주세요.
                - 답변 어조 : 질문에 대해서 사실 기반으로 답변해주세요. 답변 작성시 친숙한 어조로 작성해주세요. 
                - 언어: 답변은 한국어로 작성하며, 명확하고 간결하게 요약하여 500자 이내로 제한합니다.
                - 답변 형식: 답변은 '답변 내용'과 '참고 문서'로 구분해서 작성해주세요. 답변 내용에는 이름, 가입된 보험이름을 포함하여 답변해주세요. 참고문서는 약관메타 정보를 이용하여 약관ID, 청크ID, 약관항목를 포함하여 답변해주세요.:"""

            else:
                system_message_for_insurance_scenario = ""
            
            print("documents:", documents)
            context = format_docs(documents)

        # print("context:", context)
        chat = ChatOCIGenAI(
            model_id=genai_model_id,
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id="ocid1.compartment.oc1..aaaaaaaal7ipgtkkohxxjdbgxmqap4jx3gloyd52f33ujv3thz45uwopjmna",
            model_kwargs={"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p},
        )

        message = [
            (
                "system",
                """
                질문-답변 업무를 돕는 AI 어시스턴트입니다. 
                문서의 내용을 참고해서 답변해 주세요.:
                {system_message_for_insurance_scenario}
                \n\n
                {context}",
                """
            ),
            ("human", "{human}"),
        ]
        prompt = ChatPromptTemplate.from_messages(message)
        
        chain = {
            "context": RunnablePassthrough(),
            "human": RunnablePassthrough(),
            "system_message_for_insurance_scenario": RunnablePassthrough()
        } | prompt | chat | StrOutputParser()
        
        response = chain.invoke({
            "human": query,
            "context": context if with_rag else RunnablePassthrough(),  # pass context as a string here
            "system_message_for_insurance_scenario": system_message_for_insurance_scenario
        })

        return response

    def bot(history, max_tokens, temperature, top_p, top_k, code_input):

        query = history[-1][0]

        if retriever == "VectorStore":
            response = response_from_oraclevs(query, max_tokens, temperature, top_p, top_k)
        else:
            response = response_from_oraclesql(query, max_tokens, temperature, top_p, top_k, code_input)

            if response == "":
                response = "SQL 검색기를 선택한 경우에는 문서 검색을 위한 SQL 구문을 제공해 주셔야 합니다."
                
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
    ).then(bot, [chatbot, max_tokens, temperature, top_p, top_k, code_input], chatbot)

    clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    # demo.launch(share=True)
    demo.launch(server_port=7860,server_name="0.0.0.0")